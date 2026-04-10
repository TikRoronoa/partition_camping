/*
 * partition_camping_3080.cu
 * 
 * 验证 RTX 3080 (384-bit / 12-partition) 上的 partition camping 行为
 * 编译: nvcc -O3 -arch=sm_86 -o camping partition_camping_3080.cu
 *
 * 实验分五组:
 *   EXP1 - Stride 扫描: 带宽 vs 步长关系
 *   EXP2 - 临界步长分析: 3072B 附近精细测量
 *   EXP3 - 占用分区数量 vs 带宽
 *   EXP4 - L2 cache 效应的排除
 *   EXP5 - Nsight 可观测指标基线
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/* ─── 常量 ─────────────────────────────────────────────────── */
// GA102 物理参数
static const int  NUM_PARTITIONS    = 12;     // 3080 有 12 个 32-bit 分区
static const int  CACHE_LINE_BYTES  = 128;    // L2 cache line
static const int  SECTOR_BYTES      = 32;     // DRAM sector (GDDR6X)
static const int  PARTITION_PERIOD  = 3072;   // 256 * 12 字节 = 一个完整映射周期
static const long BUF_BYTES         = 2048L * 1024 * 1024; // 2 GB, 远大于 L2 (6 MB), 适配 12G 显存
static const int  WARMUP            = 5;
static const int  ITERS             = 20;

/* ─── EXP1: stride 读取核函数 ─────────────────────────────── */
// 每个线程以固定步长访问 float 元素
// active_n: 实际触碰的元素个数 (控制总流量)
__global__ void stride_read(const float* __restrict__ buf,
                             int stride_floats,
                             long active_n,
                             float* sink)
{
    long tid  = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long step = (long)gridDim.x  * blockDim.x;
    float acc = 0.f;
    for (long i = tid; i < active_n; i += step) {
        long addr = (i * stride_floats) % (BUF_BYTES / sizeof(float));
        acc += buf[addr];
    }
    if (acc != 0.f) *sink = acc;  // 防止编译器优化掉
}

/* ─── EXP3: 只访问 k 个分区的核函数 ─────────────────────── */
// 通过让步长 = 256/sizeof(float) * (12/k) 来控制占用的分区数
__global__ void k_partition_read(const float* __restrict__ buf,
                                  int stride_floats,
                                  long active_n,
                                  float* sink)
{
    long tid  = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long step = (long)gridDim.x  * blockDim.x;
    float acc = 0.f;
    for (long i = tid; i < active_n; i += step) {
        long addr = (i * stride_floats) % (BUF_BYTES / sizeof(float));
        acc += buf[addr];
    }
    if (acc != 0.f) *sink = acc;
}

/* ─── 计时工具 ──────────────────────────────────────────── */
typedef struct { float ms; double gbps; } TimedResult;

static TimedResult time_kernel_stride(const float* d_buf, int stride_floats,
                                       float* d_sink, long total_bytes_accessed)
{
    // 计算访问的元素数量
    // 确保访问范围不超出 buf，同时足够大绕过 L2
    long active_n = (long)(BUF_BYTES / sizeof(float)) / stride_floats;
    // 最少保证 64 MB 流量
    while (active_n * sizeof(float) < 64L * 1024 * 1024) active_n *= 2;

    int threads = 256;
    int blocks  = 1024;  // 足够多 SM 都忙碌

    // warmup
    for (int i = 0; i < WARMUP; i++)
        stride_read<<<blocks, threads>>>(d_buf, stride_floats, active_n, d_sink);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < ITERS; i++)
        stride_read<<<blocks, threads>>>(d_buf, stride_floats, active_n, d_sink);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= ITERS;

    // 实际字节流量: active_n * 4 bytes 读
    double bytes = (double)active_n * sizeof(float);
    double gbps  = bytes / (ms * 1e-3) / 1e9;

    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));
    return (TimedResult){ms, gbps};
}

/* ─── 推算当前步长映射到多少个分区 ─────────────────────── */
static int partitions_touched(int stride_bytes)
{
    // 追踪前 NUM_PARTITIONS 次访问各自落在哪个分区
    int seen[NUM_PARTITIONS] = {0};
    int count = 0;
    long addr = 0;
    for (int i = 0; i < 1000; i++) {
        int p = (int)((addr / 256) % NUM_PARTITIONS);
        if (!seen[p]) { seen[p] = 1; count++; }
        if (count == NUM_PARTITIONS) break;
        addr += stride_bytes;
    }
    return count;
}

/* ─── EXP1: stride 扫描 ─────────────────────────────────── */
static void exp1_stride_sweep(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════\n");
    printf("EXP1: Stride 扫描  (理论峰值 ~912 GB/s)\n");
    printf("%-18s %-12s %-14s %-10s %-8s\n",
           "stride_bytes", "partitions", "BW(GB/s)", "利用率%", "camping?");
    printf("────────────────────────────────────────────────────\n");

    // 测试从 4B (单元素) 到 192KB 的步长
    int strides_bytes[] = {
        4,      // 连续访问 → 最优
        32,     // 每分区 2 transaction, 间隔一个sector
        64,     // 每分区 2 transaction,2个sector，半 cache line
        96,     // 每分区 2 transaction
        128,    // 每分区 2.67 transaction, 4个sector,一个 cache line, 
        160,    // 每分区 3.33 transaction
        192,    // 每分区 4 transaction
        224,    // 
        256,    // partition granularity
        512,
        768,
        1024,
        1536,
        2048,
        3072,   // = 256*12 → camping! 所有线程落同一分区
        3072+256,
        3072+256*2,
        3072+256*3,
        4096,   // = 3072+1024, 公因子大
        6144,   // 2 * 3072 → camping
        9216,   // 3 * 3072 → camping
        12288,  // 4 * 3072 → camping
        3076,   // 3072+4 → 轻微偏移, 应该改善
        3080,   // 3072+8
        3200,   // 接近但不整除
        4096+3072, // 最坏叠加
        65536,  // 64 KB, gcd(65536,3072)=1024 → 3分区
        65537,
        196611,
        
        65536 + 4,   // 已有 → 30 GB/s  ✓
        65536 + 8,   // 预期 30 GB/s（8B偏移，float对齐后仍有偏移）
        65536 + 1,   // 已有(65537) → 466 GB/s（1B偏移被float对齐吃掉）✓
        65536 + 2,   // 预期高带宽（2B偏移也被float对齐吃掉）
        65536 + 3,   // 预期高带宽（3B偏移也被float对齐吃掉）

        65540,   // 65536+4，应该是12分区，预期高带宽
        65024,   // 65536-512，gcd变化，看分区数是否改变
        63488,   // 62×1024，整除但非3072倍数，看row buffer hit是否触发
        196608, // 192 KB = 64 * 3072 → camping
        197632, // 193 KB
        
    };

    double peak_bw = 912.0;  // RTX 3080 12G 理论峰值 GB/s (384-bit × 19 Gbps / 8)

    for (int si = 0; si < (int)(sizeof(strides_bytes)/sizeof(strides_bytes[0])); si++) {
        int sb = strides_bytes[si];
        int sf = sb / (int)sizeof(float);
        if (sf < 1) sf = 1;

        int parts = partitions_touched(sb);
        TimedResult r = time_kernel_stride(d_buf, sf, d_sink, 0);

        // camping 判定: 分区利用率 < 33% (4 of 12 or fewer)
        const char* tag = (parts <= 4) ? "*** CAMPING" : (parts <= 8 ? "partial" : "OK");
        printf("%-18d %-12d %-14.1f %-10.1f %s\n",
               sb, parts, r.gbps, r.gbps / peak_bw * 100.0, tag);
    }
}

/* ─── EXP2: 临界步长精细分析 ────────────────────────────── */
static void exp2_critical_stride(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════\n");
    printf("EXP2: 3072B 附近精细扫描 (步长 = 3072 ± delta)\n");
    printf("%-18s %-12s %-14s %-8s\n",
           "stride_bytes", "partitions", "BW(GB/s)", "delta");
    printf("────────────────────────────────────────────────────\n");

    for (int delta = -128; delta <= 128; delta += 4) {
        int sb = 3072 + delta;
        if (sb <= 0) continue;
        int sf = sb / (int)sizeof(float);
        if (sf < 1) sf = 1;

        int parts = partitions_touched(sb);
        TimedResult r = time_kernel_stride(d_buf, sf, d_sink, 0);
        printf("%-18d %-12d %-14.1f %+d\n", sb, parts, r.gbps, delta);
    }
}

/* ─── EXP3: 分区数量 vs 带宽 ────────────────────────────── */
static void exp3_partition_count(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════\n");
    printf("EXP3: 手动控制使用的分区数量\n");
    printf("%-16s %-16s %-14s %-10s\n",
           "target_parts", "stride_bytes", "BW(GB/s)", "利用率%");
    printf("────────────────────────────────────────────────────\n");

    // 构造访问 k 个分区的步长:
    // 使步长为 (12/k) * 256 字节, 则 gcd=3072/(12/k) → k 个分区
    // k=1: stride=3072 (访问分区0)
    // k=2: stride=1536
    // k=3: stride=1024
    // k=4: stride=768
    // k=6: stride=512
    // k=12: stride=256

    struct { int k; int stride_bytes; } cases[] = {
        {1,  3072},
        {2,  1536},
        {3,  1024},
        {4,   768},
        {6,   512},
        {12,  256},
    };

    double peak_bw = 912.0;
    for (int ci = 0; ci < sizeof(cases)/sizeof(cases[0]); ci++) {
        int k  = cases[ci].k;
        int sb = cases[ci].stride_bytes;
        int sf = sb / (int)sizeof(float);
        int real_parts = partitions_touched(sb);
        TimedResult r = time_kernel_stride(d_buf, sf, d_sink, 0);
        printf("%-16d %-16d %-14.1f %.1f%%\n",
               real_parts, sb, r.gbps, r.gbps / peak_bw * 100.0);
    }
}

/* ─── EXP4: 排除 L2 cache 效应 ─────────────────────────── */
// 同样步长但分别用 32MB / 64MB / 256MB / 512MB buffer
// 如果 < L2(5MB) 的数据被 cache 命中，带宽会虚高
__global__ void stride_read_sized(const float* __restrict__ buf,
                                   int stride_floats,
                                   long buf_floats,
                                   long active_n,
                                   float* sink)
{
    long tid  = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long step = (long)gridDim.x  * blockDim.x;
    float acc = 0.f;
    for (long i = tid; i < active_n; i += step) {
        long addr = (i * stride_floats) % buf_floats;
        acc += buf[addr];
    }
    if (acc != 0.f) *sink = acc;
}

static void exp4_cache_exclusion(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════\n");
    printf("EXP4: 排除 L2 cache 效应  (camping stride = 3072)\n");
    printf("%-16s %-14s %-8s\n", "buf_size_MB", "BW(GB/s)", "来源");
    printf("────────────────────────────────────────────────────\n");

    // RTX 3080 12G: L2 = 6 MB (GA102 完整配置), 总显存 ~12 GB
    long sizes_mb[] = {1, 4, 8, 16, 64, 256, 1024, 2048};
    int stride_bytes = 3072;
    int stride_floats = stride_bytes / (int)sizeof(float);

    for (int si = 0; si < sizeof(sizes_mb)/sizeof(sizes_mb[0]); si++) {
        long mb      = sizes_mb[si];
        long buf_f   = mb * 1024 * 1024 / sizeof(float);
        long active_n = buf_f / stride_floats;
        while (active_n * sizeof(float) < 32L * 1024 * 1024) active_n *= 2;

        int threads = 256, blocks = 1024;

        for (int w = 0; w < WARMUP; w++)
            stride_read_sized<<<blocks, threads>>>(d_buf, stride_floats, buf_f, active_n, d_sink);
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));
        CHECK(cudaEventRecord(t0));
        for (int it = 0; it < ITERS; it++)
            stride_read_sized<<<blocks, threads>>>(d_buf, stride_floats, buf_f, active_n, d_sink);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms /= ITERS;
        double gbps = (double)active_n * sizeof(float) / (ms * 1e-3) / 1e9;
        CHECK(cudaEventDestroy(t0));
        CHECK(cudaEventDestroy(t1));

        const char* src = (mb <= 4) ? "L2 hit (无效)" : (mb <= 8 ? "L2 partial" : "DRAM (有效)");
        // GA102 RTX 3080 12G: L2 = 6 MB
        printf("%-16ld %-14.1f %s\n", mb, gbps, src);
    }
}

/* ─── EXP5: 预期 Nsight 指标 ────────────────────────────── */
static void exp5_nsight_guidance()
{
    printf("\n════════════════════════════════════════════════════\n");
    printf("EXP5: 如何在 Nsight Compute 中观测 camping\n");
    printf("────────────────────────────────────────────────────\n");
    printf("\n关键指标 (Nsight Compute → Memory Workload Analysis):\n\n");
    printf("  l2_global_load_bytes          → 实际产生的 L2 读请求\n");
    printf("  dram_read_throughput          → DRAM 端实测带宽\n");
    printf("  lts__t_sectors_srcunit_tex_op_read → per-partition sector 计数\n");
    printf("    (12 个子计数器, camping 时某几个远高于其他)\n\n");

    printf("  camping 时典型观测:\n");
    printf("    dram_read_throughput     ≈  75-120 GB/s  (峰值 912 的 8-13%%)\n");
    printf("    sm__sass_l1tex_m_xbar2l1tex_read_sectors_pipe_lsu_mem_global_op_ld.avg\n");
    printf("    → per-warp L2 sector 请求数激增 (cache miss 率 100%%)\n\n");

    printf("  ncu 命令示例:\n");
    printf("    ncu --metrics \\\n");
    printf("      l2_global_load_bytes,\\\n");
    printf("      dram__bytes_read.sum,\\\n");
    printf("      lts__t_sectors_srcunit_tex_op_read.sum,\\\n");
    printf("      lts__t_sector_hit_rate.pct \\\n");
    printf("      ./camping\n\n");

    printf("  理想 (stride=4): 所有12个分区负载均衡, hit_rate ~60%%\n");
    printf("  camping (stride=3072): 1个分区 100%% 负载, hit_rate ~0%%\n");
}

/* ─── main ──────────────────────────────────────────────── */
int main()
{
    // 打印设备信息
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);
    printf("SM count: %d  |  L2 cache: %d MB  |  VRAM: %.0f MB\n",
           prop.multiProcessorCount,
           prop.l2CacheSize / (1024 * 1024),
           (double)prop.totalGlobalMem / (1024 * 1024));
    printf("\nGA102 内存分区参数:\n");
    printf("  bus width        = 384 bits  = 12 × 32-bit\n");
    printf("  partition period = 256 × 12  = %d bytes\n", PARTITION_PERIOD);
    printf("  GDDR6X peak BW   ≈ 912 GB/s  (384-bit × 19 Gbps / 8)\n");
    printf("  camping stride   = N × 3072  → 仅命中 1 个分区\n");

    // 分配 2GB buffer (远超 L2 的 6MB, 适配 12G 显存)
    float* d_buf;
    CHECK(cudaMalloc(&d_buf, BUF_BYTES));
    CHECK(cudaMemset(d_buf, 0, BUF_BYTES));

    float* d_sink;
    CHECK(cudaMalloc(&d_sink, sizeof(float)));

    exp1_stride_sweep   (d_buf, d_sink);
    exp2_critical_stride(d_buf, d_sink);
    exp3_partition_count(d_buf, d_sink);
    exp4_cache_exclusion(d_buf, d_sink);
    exp5_nsight_guidance();

    CHECK(cudaFree(d_buf));
    CHECK(cudaFree(d_sink));
    printf("\n[完成]\n");
    return 0;
}



/**
 * 非常完整的实测数据！让我逐组深度分析，有几个现象非常有趣，甚至超出了理论预期。

---

## EXP1 — 整体解读

**第一个异常：stride=4 时 869 GB/s，却没到 912 GB/s**

这是正常的。连续访问时 L1/L2 prefetch 介入，实际 DRAM 利用率受 warp 调度和 memory pipeline 饱和度限制，能到 95% 已经非常好。

**第二个异常，也是最关键的发现：**

```
stride=64   → 12分区，但带宽只有 108 GB/s
stride=256  → 12分区，但带宽只有  59 GB/s
stride=512  → 6分区，  带宽        58 GB/s
```

这说明：**分区数量不是带宽的唯一决定因素**。stride=64 虽然命中 12 个分区，带宽却只有 stride=4 的 1/8。原因是 **cache line 利用率（sector utilization）**的差异：

| stride | 每次 load 触碰的 sector                     | 128B cache line 利用率 | 带宽      |
|--------|-------------------------------------------|------------------------|----------|
| 4B     | 连续32线程填满4 sector                      | ~100%                  | 869 GB/s |
| 64B    | 每线程跨 2 sector                          | ~50%                   | 108 GB/s  |
| 128B   | 每线程独占 1 cache line，但 warp 32线程=32行 | ~3%                    | 101 GB/s  |
| 256B   | 同上但更稀疏                                | ~1.5%                  | 59 GB/s  |

本质上，从 stride=64 开始就已经是严重的 cache line 浪费，camping 只是在这个基础上再叠加分区拥塞。

第三个关键数字：camping 时的带宽底部

```
3072B → 30.5 GB/s
1536B → 30.7 GB/s
1024B → 30.9 GB/s
```

所有 camping 情况都收敛在 **~30-31 GB/s**，这不是偶然。这是单个 GDDR6X 颗粒的实测极限：

```
912 GB/s ÷ 12 partitions = 76 GB/s（单分区理论上限）
实测 30 GB/s ≈ 单分区上限的 40%
```

为什么只有 40%？因为 camping 时不只是带宽瓶颈，还有 **DRAM bank conflict** 和 **行激活（row activation）开销**叠加——所有请求挤在一个颗粒，tRAS/tRP 等时序延迟无法被其他分区的请求掩盖。

**第四个异常：stride=65536 和 196608 带宽高达 460-466 GB/s**

```
65536  → 3分区，  466 GB/s
196608 → 1分区，  460 GB/s   ← 只用1个分区却有460 GB/s！
```

这非常反直觉——camping 反而带宽高？原因是 **DRAM row buffer hit rate 极高**。stride=196608 = 192KB，正好是 GDDR6X 一个 row 的整数倍（典型 row size = 1KB~4KB），大步长访问时相邻访问落在同一 DRAM row，触发 **row buffer hit**，延迟从 ~300 cycle 降至 ~30 cycle，吞吐量反而暴增。这是 camping 研究里经常被忽视的"假性高带宽"陷阱。

---

## EXP2 — 临界步长分析

```
3068  → 12分区，30.7 GB/s
3072  →  1分区，30.5 GB/s   ← camping
3076  → 12分区，30.7 GB/s
```

**带宽差异只有 0.2 GB/s！**

这揭示了一个深层事实：**在这个 stride 范围内，带宽早就被 cache line 浪费打爆了，分区 camping 的额外惩罚几乎被淹没**。3072 附近所有步长的带宽都是 ~30.6 GB/s，瓶颈不是 camping，而是**每个 warp 的 128B cache line 里只用了 4 字节（float），97% 的 DRAM 带宽全都浪费在搬运不需要的数据**。

换句话说，EXP2 的结论是：**在大 stride（~3KB）场景下，sector 浪费的危害远大于 partition camping 本身**。

---

## EXP3 — 分区数 vs 带宽

```
1分区  → 30.5 GB/s
2分区  → 30.7 GB/s
3分区  → 30.9 GB/s
4分区  → 31.4 GB/s
6分区  → 58.1 GB/s   ← 突变
12分区 → 59.5 GB/s
```

注意 **4→6 分区之间有一个明显的跳变**（31→58 GB/s），而 6→12 分区几乎没有增益（58→59 GB/s）。这与 GA102 的物理拓扑吻合：

GA102 的 12 个内存分区分布在芯片两侧，每侧 6 个，对应两个独立的 **memory subsystem**。步长选到 512B（6分区）时恰好把两侧各 6 个分区都激活，跨越了 subsystem 边界，带宽翻倍。而 6→12 没有再翻倍，说明两侧 subsystem 已经都饱和，再加分区也只是分散负载，没有新的物理通道可用。

---

## EXP4 — L2 cache 效应排除

```
1 MB   → 442 GB/s   (L2 hit)
4 MB   → 334 GB/s   (L2 hit)
8 MB   → 311 GB/s   (L2 partial)
16 MB  → 297 GB/s   (DRAM)
64 MB  → 102 GB/s   (DRAM)
256 MB →  39 GB/s   (DRAM)
1024MB →  30.6 GB/s (DRAM)
2048MB →  30.6 GB/s (DRAM, 稳定)
```

有两个值得深挖的现象：

**1. 16MB 时已经是 DRAM，但带宽还有 297 GB/s**

这是 **hardware prefetcher 的作用域**。16MB buffer + camping stride 时，prefetcher 还能在一定范围内预测访问模式，提前发出请求。到 64MB 时 prefetch 深度不够，带宽跌到 102 GB/s。

**2. 256MB→1024MB 带宽从 39 降到 30.6，之后稳定**

256MB 时还有少量 TLB 覆盖带来的地址转换加速。超过 1GB 后 TLB 完全失效，每次访问都需要 page walk，这就是真实的 camping 底线：**30.6 GB/s**。

---

## 综合结论

| 现象 | 根因 |
|------|------|
| camping 底线 30.6 GB/s | 单分区 DRAM 带宽上限 + bank conflict 延迟惩罚 |
| EXP2 camping vs 非camping 差异仅 0.2 GB/s | 大 stride 下 sector 浪费已是主瓶颈 |
| 4→6 分区带宽跳变 | GA102 双侧 memory subsystem 边界 |
| 65536/196608 camping 却高带宽 | DRAM row buffer hit 掩盖分区拥塞 |
| 真实 camping 底线需 >1GB buffer | TLB、prefetcher、L2 层层掩盖，512MB 以下均不可信 |

最后一点对实际工程最有价值：**如果你的 kernel 工作集 < 256MB，即使发生 camping，测出来的数字也是虚高的**——真正上线大模型或大 batch 时才会踩到 30 GB/s 的坑。

“这非常反直觉——camping 反而带宽高？原因是 DRAM row buffer hit rate 极高。stride=196608 = 192KB，
正好是 GDDR6X 一个 row 的整数倍（典型 row size = 1KB~4KB），大步长访问时相邻访问落在同一 DRAM row，
触发 row buffer hit，延迟从 ~300 cycle 降至 ~30 cycle，吞吐量反而暴增。
这是 camping 研究里经常被忽视的"假性高带宽"陷阱。”
请详细解释这句话

这是一个非常有深度的问题，涉及 DRAM 架构的底层机制。让我一步步拆解这个"假性高带宽陷阱"。

核心：DRAM Row Buffer 的工作原理
DRAM 的读取并不是随机寻址的——它有一套三阶段协议，性能取决于当前行是否已经"开着"。

这里有一个图，见目录文件:dram_row_buffer_states.svg

关键结论：Row Hit 的延迟只有 Row Empty/Conflict 的 1/10，吞吐量因此可以高出一个数量级。

为什么大步长 stride=192KB 反而触发 Row Hit？
这里有一个反直觉的几何关系：

这里有一个图，见目录文件:stride_row_mapping.svg

地址映射的数学关系
DRAM 地址解码方式决定了这一切。以典型 GDDR6X 为例：
物理地址 → [Channel | Rank | Bank | Row | Column]

Row 地址 = addr >> log2(row_size)   // 例如 addr >> 10（1KB row）

当 stride = 192KB = 192 × 1KB 时：

addr_N = addr_0 + N × 192 × 1024
Row(addr_N) = (addr_0 + N × 192 × 1024) >> 10
            = (addr_0 >> 10) + N × 192
每次访问的 row 编号以 192 为步长递增，但在同一个 bank 里，row buffer 里装的是同一行数据——只要前后两次访问打到同一个 bank 的同一行，就是 Row Hit，无需重新 Activate。
关键条件：stride 必须是 row_size 的整数倍，且连续访问命中同一 bank 的同一行（地址映射中 bank 选择位没有变化）。

"假性高带宽"的陷阱是什么？
这里有一个重要的误解链：
camping 实验的目的 是测量 GPU 对显存的持续施压能力——通常用"有多少 warp 在同时争抢同一块显存"来衡量访存拥塞。
当研究者看到"stride=192KB 时带宽最高"，可能得出错误结论：

"这个 stride 下 GPU 的显存访问效率最高，说明此时显存带宽利用率最饱和"

但实际上：

表面现象                    真实原因
吞吐量高                    Row buffer hit rate 极高，延迟从 300→30 cycles
看起来"带宽高"               每次读取几乎无等待，时间窗口内完成了更多事务
误以为是 camping 压力大      其实是 DRAM 地址对齐的"意外福利"
这是延迟掩盖了带宽的典型情形：你测量的是"单位时间传输了多少数据"，但分母（时间）因为延迟降低而变小了，分子（数据量）并没有真正增加——显存总线的物理带宽利用率未必更高，甚至可能更低（因为访问的数据集更稀疏）。
真正的 camping 压力测试需要刻意制造 Row Conflict，才能真实反映 warp 争抢显存的拥塞延迟。如果恰好踩到 row-aligned stride，反而让每个 warp 都"轻松过关"，测到的是最优情况而非最差情况，与 camping 实验的初衷背道而驰。
 * 
 */


 /**
  * 
  * 
  * 
  * 
  * 

一些重要的结论：

## 重新看数据

```
stride=32:  108.6 GB/s
stride=64:  108.5 GB/s
stride=96:  107.7 GB/s
stride=128: 101.7 GB/s
```

stride=32 到 stride=96 几乎完全相同，stride=128 才开始下降。

## 先算清楚每种情况 DRAM 实际收到了什么

```
stride=32, warp 32线程：
  地址：0, 32, 64, ..., 992
  覆盖范围：[0, 1024)
  128B 对齐的 transaction 数 = 1024/128 = 8 个

stride=64, warp 32线程：
  地址：0, 64, 128, ..., 1984
  覆盖范围：[0, 2048)
  128B transaction 数 = 2048/128 = 16 个

stride=128, warp 32线程：
  地址：0, 128, 256, ..., 3968
  覆盖范围：[0, 4096)
  128B transaction 数 = 4096/128 = 32 个
```

transaction 数量是 8、16、32，翻倍增长。但带宽 108、108、101，几乎不变。

**这说明 transaction 数量翻倍，DRAM 完成这些 transaction 的时间也翻倍了。带宽 = 有效字节/时间，分子（有效字节 = active_n × 4B）不变，分母（时间）也不变，所以带宽不变。**

但为什么时间不变？transaction 多了一倍，不应该花更多时间吗？

---

## 关键：DRAM 的并发处理能力

DRAM 不是串行处理 transaction 的，而是靠多个 bank 并发处理：

```
stride=32:  8 个 transaction，覆盖 [0, 1024)
  这 8 个 transaction 打到 1024/256 ≈ 4 个分区
  每个分区处理 2 个 transaction
  4 个分区并行 → 实际耗时 ≈ 2 个 transaction 的时间

stride=64: 16 个 transaction，覆盖 [0, 2048)
  打到 2048/256 ≈ 8 个分区
  每个分区处理 2 个 transaction
  8 个分区并行 → 实际耗时 ≈ 2 个 transaction 的时间
```

两种情况下每个分区的负载相同，耗时相同，所以总带宽相同。

---

## 用这个模型验证 stride=128 为什么开始下降

```
stride=128: 32 个 transaction，覆盖 [0, 4096)
  打到 4096/256 = 16 个分区，但 GA102 只有 12 个分区
  → 有些分区需要处理多于 2 个 transaction
  → 部分分区成为瓶颈，整体带宽开始下降
```

验证：

```
stride=32:  每分区负载 = 8 transaction / 4 分区 = 2 个
stride=64:  每分区负载 = 16 transaction / 8 分区 = 2 个  ← 相同！
stride=96:  每分区负载 = 24 transaction / 12 分区 = 2 个 ← 相同！
stride=128: 每分区负载 = 32 transaction / 12 分区 = 2.67 个 ← 增加了！
```

**stride=32、64、96 时每个分区的负载完全相同（2 个 transaction），所以带宽完全相同。stride=128 开始每分区负载超过 2，带宽开始下降。**

这也预测了一个新的现象：

```
stride=96 之后带宽应该持续下降
stride=32×N 时如果每分区负载恰好是整数，带宽应该有局部平台
```

回看实验数据：

```
stride=32:  108.6 GB/s  每分区 2 transaction
stride=64:  108.5 GB/s  每分区 2 transaction
stride=96:  107.7 GB/s  每分区 2 transaction
stride=128: 101.7 GB/s  每分区 2.67 transaction ← 开始下降 ✓
stride=160:  87.2 GB/s  每分区 3.33 transaction
stride=192:  73.9 GB/s  每分区 4 transaction
stride=256:  59.3 GB/s  每分区 5.33 transaction
stride=512:  57.9 GB/s  每分区 ← 分区数变成6了，另一个效应
```

stride=192（每分区恰好 4 个 transaction）应该是一个局部低点，之后 stride=224 每分区 4.67，stride=256 每分区 5.33，带宽应该继续下降但有轻微波动。

实测：

```
stride=192: 73.9 GB/s
stride=224: 66.3 GB/s
stride=256: 59.3 GB/s
```

单调下降，没有明显的局部平台，说明每分区负载不是整数时的带宽变化是平滑的，不像我预测的那样有阶跃。这可能是因为不同分区的负载并不是均匀分布的，实际情况更复杂。

---

## 最终结论

**stride=32 和 stride=64 带宽相同的真实原因：**

```
两种 stride 下，warp 的访问地址恰好分散到相同数量的分区
每个分区收到的 transaction 数量相同
各分区并行处理，总耗时相同
→ 有效带宽相同

具体：
  stride=32: 8 transaction，分布到 4 分区，每分区 2 个
  stride=64: 16 transaction，分布到 8 分区，每分区 2 个
  每分区负载相同 → 带宽相同

stride=128 开始下降：
  32 transaction 超过了 12 分区的均匀承载能力（12×2=24）
  部分分区过载 → 整体带宽下降
```
  * 
  * 
  * 
  * 
  * 
  * 
  * 
  */