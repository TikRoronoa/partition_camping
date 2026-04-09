/*
 * page_size_verify_v2.cu
 *
 * 验证 GDDR6X DRAM page size 假说
 *
 * 与 v1 的关键区别：
 *   - 核函数以字节为单位计算 stride，但以 float 为单位读取
 *   - 这样可以做到字节级精度的 stride，同时保持与原始实验相同的访问模式
 *   - active_n 的计算修复为基于实际访问地址数，而不是流量字节数
 *
 * 编译: nvcc -O3 -arch=sm_86 -o page_size_verify_v2 page_size_verify_v2.cu
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
                cudaGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/* ─── 常量 ──────────────────────────────────────────────── */
static const long   BUF_BYTES  = 2048L * 1024 * 1024;  // 2GB
static const int    WARMUP     = 5;
static const int    ITERS      = 30;
static const int    THREADS    = 256;
static const int    BLOCKS     = 1024;
static const double PEAK_BW    = 912.0;

/* ─── 核心核函数：字节级 stride，float 读取 ────────────── */
//
// 与原始实验的唯一差别：stride 从 float 单位改为 byte 单位
// 这样 stride=65537 和 stride=65536 就是真正不同的访问模式
//
// 原始: float_idx = i * stride_floats          (stride 只能是4的倍数)
// 新版: float_idx = (i * stride_bytes) / 4     (stride 可以是任意字节数)
//
// stride_bytes=65536: float_idx = i * 16384         (与原始完全相同)
// stride_bytes=65537: float_idx = i*16384 + i/4     (真正不同)
// stride_bytes=65538: float_idx = i*16384 + i/2     (真正不同)
// stride_bytes=65539: float_idx = i*16384 + 3*i/4   (真正不同)
// stride_bytes=65540: float_idx = i*16385            (与原始相同)
//
__global__ void stride_read_byteprec(const float* __restrict__ buf,
                                      long stride_bytes,
                                      long active_n,
                                      float* sink)
{
    long tid      = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long step     = (long)gridDim.x  * blockDim.x;
    long buf_floats = BUF_BYTES / sizeof(float);
    float acc = 0.f;
    for (long i = tid; i < active_n; i += step) {
        // 字节地址 → float 索引（向下对齐到4字节边界）
        long byte_addr  = (i * stride_bytes) % BUF_BYTES;
        long float_idx  = (byte_addr >> 2) % buf_floats;  // >>2 等价于 /4
        acc += buf[float_idx];
    }
    if (acc != 0.f) *sink = acc;
}

/* ─── 计时函数 ──────────────────────────────────────────── */
static double measure_bw(const float* d_buf, long stride_bytes, float* d_sink)
{
    // active_n：覆盖整个 2GB buffer 至少一遍
    // 这保证了：
    //   1. 访问的地址数 = BUF_BYTES / stride_bytes（足够多的唯一地址）
    //   2. 总流量 = active_n * 4 bytes（实际读取量）
    long active_n = BUF_BYTES / stride_bytes;

    // 对于极大 stride（如 stride > 256MB），active_n 会很小
    // 补足到至少能让所有 SM 都忙碌：BLOCKS * THREADS * 若干轮
    long min_active = (long)BLOCKS * THREADS * 64;
    if (active_n < min_active) active_n = min_active;

    // warmup
    for (int i = 0; i < WARMUP; i++)
        stride_read_byteprec<<<BLOCKS, THREADS>>>(d_buf, stride_bytes,
                                                   active_n, d_sink);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < ITERS; i++)
        stride_read_byteprec<<<BLOCKS, THREADS>>>(d_buf, stride_bytes,
                                                   active_n, d_sink);
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    ms /= ITERS;

    CHECK(cudaEventDestroy(t0));
    CHECK(cudaEventDestroy(t1));

    // 实际读取字节数 = active_n * sizeof(float)
    double bytes = (double)active_n * sizeof(float);
    return bytes / (ms * 1e-3) / 1e9;
}

/* ─── 推算分区数 ─────────────────────────────────────────── */
static int partitions_touched(long stride_bytes)
{
    const int NUM_PARTITIONS = 12;
    int seen[12] = {0};
    int count = 0;
    long addr = 0;
    for (int i = 0; i < 10000 && count < NUM_PARTITIONS; i++) {
        int p = (int)((addr / 256) % NUM_PARTITIONS);
        if (!seen[p]) { seen[p] = 1; count++; }
        addr += stride_bytes;
    }
    return count;
}

/* ─── EXP D: 对照组，验证新核函数与原始结果一致 ────────── */
static void expD_sanity_check(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP D: 对照组 (新核函数 vs 原始结果)\n");
    printf("%-18s %-12s %-14s %-24s\n",
           "stride_bytes", "partitions", "BW(GB/s)", "原始结果");
    printf("────────────────────────────────────────────────────────\n");

    struct { long sb; const char* expected; } cases[] = {
        {4,      "~869 GB/s"},
        {3072,   "~30  GB/s (camping)"},
        {65536,  "~465 GB/s (高带宽)"},
        {65540,  "~30  GB/s (低带宽)"},
        {196608, "~460 GB/s (高带宽)"},
        {197632, "~30  GB/s (低带宽)"},
    };

    for (int i = 0; i < 6; i++) {
        long   sb    = cases[i].sb;
        int    parts = partitions_touched(sb);
        double bw    = measure_bw(d_buf, sb, d_sink);
        printf("%-18ld %-12d %-14.1f %s\n", sb, parts, bw, cases[i].expected);
    }
    printf("\n对照通过标准: stride=4 应 >800, stride=65536 应 >400\n");
}

/* ─── EXP B: 64KB 附近字节级精细扫描（核心实验）────────── */
static void expB_fine_scan_64k(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP B: 64KB 附近字节级精细扫描 (±32B，步进 1B)\n");
    printf("假说: 跳变点在 65536+1 处（page size = 64KB）\n");
    printf("%-18s %-12s %-14s %-10s\n",
           "stride_bytes", "partitions", "BW(GB/s)", "delta");
    printf("────────────────────────────────────────────────────────\n");

    long center = 65536L;
    for (long delta = -32; delta <= 32; delta++) {
        long sb    = center + delta;
        if (sb <= 0) continue;
        int    parts = partitions_touched(sb);
        double bw    = measure_bw(d_buf, sb, d_sink);
        // 标注跳变
        const char* tag = (bw > 200.0) ? " <-- HIGH" : "";
        printf("%-18ld %-12d %-14.1f %+ld%s\n", sb, parts, bw, delta, tag);
    }
}

/* ─── EXP C: 其他倍数附近字节级扫描 ───────────────────── */
static void expC_other_multiples(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP C: 其他 N×64KB 附近字节级扫描 (±8B)\n");
    printf("预期: 每个 N×64KB 处都有相同的跳变模式\n");
    printf("%-18s %-12s %-14s %-10s\n",
           "stride_bytes", "partitions", "BW(GB/s)", "delta");
    printf("────────────────────────────────────────────────────────\n");

    long centers[] = {
        32768,   // 32KB  = 0.5 × 64KB，如果也有跳变说明更小的page层级
        131072,  // 128KB = 2 × 64KB
        196608,  // 192KB = 3 × 64KB（已知高带宽）
        262144,  // 256KB = 4 × 64KB
    };
    const char* labels[] = {"32KB (0.5×64K)", "128KB (2×64K)",
                             "192KB (3×64K)", "256KB (4×64K)"};

    for (int ci = 0; ci < 4; ci++) {
        long center = centers[ci];
        printf("\n--- %s ---\n", labels[ci]);
        for (long delta = -8; delta <= 8; delta++) {
            long   sb    = center + delta;
            int    parts = partitions_touched(sb);
            double bw    = measure_bw(d_buf, sb, d_sink);
            const char* tag = (bw > 200.0) ? " <-- HIGH" : "";
            printf("%-18ld %-12d %-14.1f %+ld%s\n", sb, parts, bw, delta, tag);
        }
    }
}

/* ─── EXP A: 粗扫描，看整体高带宽分布 ─────────────────── */
static void expA_coarse_scan(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP A: 粗扫描 32KB~256KB (步进 1KB)\n");
    printf("用于发现所有高带宽区间，验证是否只在 N×64KB 处出现\n");
    printf("%-18s %-12s %-14s\n", "stride_bytes", "partitions", "BW(GB/s)");
    printf("────────────────────────────────────────────────────────\n");

    for (long kb = 32; kb <= 256; kb++) {
        long   sb    = kb * 1024L;
        int    parts = partitions_touched(sb);
        double bw    = measure_bw(d_buf, sb, d_sink);
        const char* tag = (bw > 200.0) ? " *** HIGH" : "";
        printf("%-18ld %-12d %-14.1f%s\n", sb, parts, bw, tag);
    }
}

/* ─── main ──────────────────────────────────────────────── */
int main()
{
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device : %s\n", prop.name);
    printf("L2     : %d MB  |  VRAM: %.0f MB\n",
           prop.l2CacheSize / (1024 * 1024),
           (double)prop.totalGlobalMem / (1024 * 1024));
    printf("\n待验证假说:\n");
    printf("  GDDR6X DRAM page size = 64KB\n");
    printf("  stride = N×64KB + [0,3]B → 原始实验看起来高带宽\n");
    printf("               但这是 float 对齐截断的伪影\n");
    printf("  本实验用字节级核函数找到真实跳变点\n");
    printf("  如果跳变点在 65537B，则 page size = 64KB 得到验证\n");
    printf("  如果跳变点在其他位置，由此反推真实 page size\n");

    float* d_buf;
    CHECK(cudaMalloc(&d_buf, BUF_BYTES));
    CHECK(cudaMemset(d_buf, 0, BUF_BYTES));

    float* d_sink;
    CHECK(cudaMalloc(&d_sink, sizeof(float)));

    // 顺序：先对照确认核函数正确，再跑核心实验
    expD_sanity_check    (d_buf, d_sink);
    expB_fine_scan_64k   (d_buf, d_sink);
    expC_other_multiples (d_buf, d_sink);
    expA_coarse_scan     (d_buf, d_sink);

    CHECK(cudaFree(d_buf));
    CHECK(cudaFree(d_sink));
    printf("\n[完成]\n");
    return 0;
}
