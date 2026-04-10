/*
 * worst_case_bw.cu
 *
 * 验证单分区内存带宽的最差情况（row conflict）
 *
 * 最好情况（已验证）：stride = N×8192 → row buffer hit → ~160 GB/s（单分区）
 * 最差情况（本实验）：stride 使每次访问都触发 row conflict → 带宽应低于 30 GB/s
 *
 * row conflict 条件：
 *   同一个 bank，访问不同的 row
 *   → 每次需要 precharge 当前行 + activate 新行
 *   → 延迟 tRP + tRCD ≈ 2× tRAS
 *
 * 实验分三组：
 *   EXP U - camping stride 扫描，在 3072 倍数里找最低带宽点
 *   EXP V - 对比三种状态：row hit / row empty / row conflict
 *   EXP W - buffer size 验证，确认最差点不受 L2/prefetch 污染
 *
 * 编译: nvcc -O3 -arch=sm_86 -o worst_case_bw worst_case_bw.cu
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
static const long   BUF_BYTES = 2048L * 1024 * 1024;
static const int    WARMUP    = 10;
static const int    ITERS     = 50;
static const int    THREADS   = 256;
static const int    BLOCKS    = 1024;
static const double PEAK_BW   = 912.0;

/* ─── 核函数：字节级 stride，float 读取 ─────────────────── */
__global__ void stride_read(const float* __restrict__ buf,
                             long stride_bytes,
                             long active_n,
                             float* sink)
{
    long tid        = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long step       = (long)gridDim.x  * blockDim.x;
    long buf_floats = BUF_BYTES / sizeof(float);
    float acc = 0.f;
    for (long i = tid; i < active_n; i += step) {
        long byte_addr = (i * stride_bytes) % BUF_BYTES;
        long float_idx = (byte_addr >> 2) % buf_floats;
        acc += buf[float_idx];
    }
    if (acc != 0.f) *sink = acc;
}

/* ─── 计时函数（中位数降噪）────────────────────────────── */
static double measure_bw(const float* d_buf,
                          long stride_bytes,
                          long active_n,
                          float* d_sink)
{
    for (int i = 0; i < WARMUP; i++)
        stride_read<<<BLOCKS, THREADS>>>(d_buf, stride_bytes, active_n, d_sink);
    CHECK(cudaDeviceSynchronize());

    float samples[50];
    for (int it = 0; it < ITERS; it++) {
        cudaEvent_t t0, t1;
        CHECK(cudaEventCreate(&t0));
        CHECK(cudaEventCreate(&t1));
        CHECK(cudaEventRecord(t0));
        stride_read<<<BLOCKS, THREADS>>>(d_buf, stride_bytes, active_n, d_sink);
        CHECK(cudaEventRecord(t1));
        CHECK(cudaEventSynchronize(t1));
        CHECK(cudaEventElapsedTime(&samples[it], t0, t1));
        CHECK(cudaEventDestroy(t0));
        CHECK(cudaEventDestroy(t1));
    }
    for (int i = 0; i < ITERS - 1; i++)
        for (int j = i + 1; j < ITERS; j++)
            if (samples[j] < samples[i]) {
                float tmp = samples[i]; samples[i] = samples[j]; samples[j] = tmp;
            }
    float ms = samples[ITERS / 2];
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
    for (int i = 0; i < 100000 && count < NUM_PARTITIONS; i++) {
        int p = (int)((addr / 256) % NUM_PARTITIONS);
        if (!seen[p]) { seen[p] = 1; count++; }
        addr += stride_bytes;
    }
    return count;
}

/* ─── EXP U: camping stride 扫描，找最低带宽点 ─────────── */
//
// 只扫 3072 的倍数（保证 camping 到 1 个分区）
// 在 row hit 底线（30 GB/s）之下寻找 row conflict 造成的更低带宽
//
// 扫描范围：3072 × 1 到 3072 × 256
// 重点关注：
//   3072 × N 其中 N 使得 stride 接近 2KB 的奇数倍（可能触发 row conflict）
//   stride 在 2KB~32KB 之间（row size 到 bank 周期之间）
//
static void expU_camping_scan(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP U: Camping stride 扫描 (3072 的倍数，N=1~256)\n");
    printf("目的: 在单分区内找 row conflict，带宽应低于 30 GB/s\n");
    printf("已知参考点:\n");
    printf("  row hit:    stride=N×8192 → ~160 GB/s (最好)\n");
    printf("  camping底线: stride=3072  → ~30  GB/s\n");
    printf("  目标:        row conflict  → <30  GB/s (最差)\n");
    printf("%-6s %-14s %-12s %-14s %-10s\n",
           "N", "stride_bytes", "partitions", "BW(GB/s)", "状态");
    printf("────────────────────────────────────────────────────────\n");

    long active_n = BUF_BYTES / sizeof(float) / 16384;
    if (active_n < (long)BLOCKS * THREADS * 64)
        active_n = (long)BLOCKS * THREADS * 64;

    double bw_baseline = 0;  // 记录3072的带宽作为基准

    for (int n = 1; n <= 256; n++) {
        long sb    = (long)n * 3072;
        int  parts = partitions_touched(sb);
        double bw  = measure_bw(d_buf, sb, active_n, d_sink);

        if (n == 1) bw_baseline = bw;

        // 状态判断
        const char* status;
        if (bw > 100.0)       status = "*** ROW HIT";
        else if (bw < bw_baseline * 0.85) status = "*** WORSE";
        else                  status = "baseline";

        // 只打印有意义的行：row hit、比基准差、或每16个打印一次
        if (bw > 100.0 || bw < bw_baseline * 0.85 || n % 16 == 0 || n <= 8)
            printf("N=%-4d %-14ld %-12d %-14.2f %s\n",
                   n, sb, parts, bw, status);
    }

    printf("\n--- 重点范围精细扫描：N=1~32（覆盖 3KB~96KB）---\n");
    printf("%-6s %-14s %-12s %-14s %-12s\n",
           "N", "stride_bytes", "partitions", "BW(GB/s)", "vs_baseline");
    printf("────────────────────────────────────────────────────────\n");

    for (int n = 1; n <= 32; n++) {
        long sb    = (long)n * 3072;
        int  parts = partitions_touched(sb);
        double bw  = measure_bw(d_buf, sb, active_n, d_sink);
        double ratio = bw / bw_baseline;

        const char* tag = (bw > 100.0) ? " *** ROW HIT" :
                          (ratio < 0.85) ? " *** WORSE"  : "";
        printf("N=%-4d %-14ld %-12d %-14.2f %.2f%s\n",
               n, sb, parts, bw, ratio, tag);
    }
}

/* ─── EXP V: 三种 row 状态对比 ──────────────────────────── */
//
// 直接对比三种状态：
//
// 状态1 - Row Hit（最好）：
//   stride = 8192×3 = 24576（3072的8倍，同时是8192的3倍）
//   每次访问同一行，row buffer 一直开着
//
// 状态2 - Row Empty（中间）：
//   构造方式：每次访问前先用另一个 stride 把 row buffer 驱逐
//   实际上难以在 GPU 上精确控制，用大 stride 近似
//
// 状态3 - Row Conflict（最差）：
//   stride 使得每次访问打到同一 bank 的不同 row
//   候选：3072 × N，其中 stride 在 2KB~32KB 之间且不是 8KB 倍数
//
// 同时测试多个候选，找真正最低的
//
static void expV_row_state_compare(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP V: Row 状态对比实验\n");
    printf("目的: 直接量化 row hit / row empty / row conflict 的带宽差异\n");
    printf("────────────────────────────────────────────────────────\n");

    long active_n = BUF_BYTES / sizeof(float) / 16384;
    if (active_n < (long)BLOCKS * THREADS * 64)
        active_n = (long)BLOCKS * THREADS * 64;

    // Row Hit 候选：3072 的倍数且是 8192 的倍数
    // 3072 × N = 8192 × M → N/M = 8192/3072 = 8/3 → N=8,M=3; N=16,M=6...
    // 最小公倍数：lcm(3072,8192) = 24576
    printf("\n--- 状态1: Row Hit 候选 (stride = N×24576) ---\n");
    long row_hit_strides[] = {24576, 49152, 73728};
    for (int i = 0; i < 3; i++) {
        long sb = row_hit_strides[i];
        double bw = measure_bw(d_buf, sb, active_n, d_sink);
        printf("  stride=%-10ld parts=%-4d BW=%.2f GB/s\n",
               sb, partitions_touched(sb), bw);
    }

    // Row Conflict 候选：3072 的倍数，stride 在 2KB~32KB 之间，不是 8KB 倍数
    // 这类 stride 的特征：
    //   > row_size(2KB) → 每步跨越不同 row
    //   < bank_period(32KB) → 不完整地遍历 bank
    //   不是 8192 的倍数 → 不触发 row hit
    printf("\n--- 状态3: Row Conflict 候选 ---\n");
    printf("  (stride 在 2KB~32KB 之间，是 3072 倍数但不是 8192 倍数)\n");

    // 3072×1=3072, ×2=6144, ×3=9216, ×4=12288, ×5=15360,
    // ×6=18432, ×7=21504, ×8=24576(8192×3,row hit!), ×9=27648, ×10=30720
    long conflict_candidates[] = {
        3072,   // ×1, 3KB
        6144,   // ×2, 6KB
        9216,   // ×3, 9KB
        12288,  // ×4, 12KB
        15360,  // ×5, 15KB
        18432,  // ×6, 18KB
        21504,  // ×7, 21KB
        27648,  // ×9, 27KB
        30720,  // ×10, 30KB
    };

    double bw_min = 1e9;
    long sb_min = 0;

    for (int i = 0; i < 9; i++) {
        long sb = conflict_candidates[i];
        double bw = measure_bw(d_buf, sb, active_n, d_sink);
        const char* tag = (bw > 100.0) ? " ROW HIT" :
                          (bw < 28.0)  ? " *** CONFLICT?" : "";
        printf("  stride=%-10ld parts=%-4d BW=%.2f GB/s%s\n",
               sb, partitions_touched(sb), bw, tag);
        if (bw < bw_min) { bw_min = bw; sb_min = sb; }
    }

    printf("\n最低带宽点: stride=%ld, BW=%.2f GB/s\n", sb_min, bw_min);

    // 汇总对比
    printf("\n--- 汇总 ---\n");
    double bw_hit      = measure_bw(d_buf, 24576,  active_n, d_sink);
    double bw_baseline = measure_bw(d_buf, 3072,   active_n, d_sink);
    double bw_worst    = measure_bw(d_buf, sb_min, active_n, d_sink);

    printf("  Row Hit     stride=24576:  %.2f GB/s  (单分区最好)\n", bw_hit);
    printf("  Camping底线 stride=3072:   %.2f GB/s  (row hit的camping)\n", bw_baseline);
    printf("  最差候选    stride=%-6ld:  %.2f GB/s  (目标最差)\n", sb_min, bw_worst);
    printf("\n  最好/最差 带宽比: %.1f×\n", bw_hit / bw_worst);
    printf("  底线/最差 带宽比: %.1f×\n", bw_baseline / bw_worst);
}

/* ─── EXP W: buffer size 验证 ───────────────────────────── */
//
// 对最差 stride 候选，用不同 buffer size 验证
// 排除 L2 cache 和 prefetcher 对最差情况的掩盖
// 如果带宽随 buffer size 增大而下降，说明小 buffer 时有 L2 命中掩盖
//
static void expW_buffer_size_verify(const float* d_buf, float* d_sink,
                                     long worst_stride)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP W: Buffer size 验证 (worst stride=%ld)\n", worst_stride);
    printf("目的: 确认最差带宽不受 L2/prefetch 污染\n");
    printf("%-16s %-14s %-10s\n", "buf_size_MB", "BW(GB/s)", "来源");
    printf("────────────────────────────────────────────────────────\n");

    // 与 EXP4 相同的 buffer size 梯度
    long sizes_mb[] = {1, 4, 8, 16, 64, 256, 1024, 2048};
    int  n_sizes    = 8;

    for (int si = 0; si < n_sizes; si++) {
        long mb       = sizes_mb[si];
        long buf_f    = mb * 1024 * 1024 / sizeof(float);
        long active_n = buf_f / (worst_stride / (long)sizeof(float));
        if (active_n < (long)BLOCKS * THREADS * 64)
            active_n = (long)BLOCKS * THREADS * 64;

        // 使用固定的 buf_floats（当前 buffer size 对应）
        // 临时核函数：访问限制在 buf_f 范围内
        // 这里复用 stride_read，通过 % BUF_BYTES 自动限制（不够精确）
        // 对于 buffer size < 2GB 的情况，active_n 较小，访问会自然限制
        double bw = measure_bw(d_buf, worst_stride, active_n, d_sink);

        const char* src = (mb <= 4)  ? "L2 hit" :
                          (mb <= 8)  ? "L2 partial" :
                          (mb <= 64) ? "prefetch区" : "DRAM (有效)";
        printf("%-16ld %-14.2f %s\n", mb, bw, src);
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

    printf("\n实验目标:\n");
    printf("  已知最好情况: stride=N×8192 (row hit)  → ~160 GB/s\n");
    printf("  已知camping底线: stride=3072            → ~30  GB/s\n");
    printf("  寻找最差情况: row conflict              → <30  GB/s ?\n");
    printf("\nrow conflict 条件:\n");
    printf("  stride 是 256 的倍数 (camping 到 1 个分区)\n");
    printf("  stride > row_size(2KB) (每步跨越不同 row)\n");
    printf("  stride 不是 bank_period 的整数倍 (不能整周期遍历)\n");

    float* d_buf;
    CHECK(cudaMalloc(&d_buf, BUF_BYTES));
    CHECK(cudaMemset(d_buf, 0, BUF_BYTES));

    float* d_sink;
    CHECK(cudaMalloc(&d_sink, sizeof(float)));

    expU_camping_scan(d_buf, d_sink);
    expV_row_state_compare(d_buf, d_sink);

    // EXP W 需要知道最差 stride，先跑 U/V 再决定
    // 这里用 6144 作为初始猜测（3072×2，stride=6KB，在 row size 和 bank period 之间）
    // 根据 U/V 结果修改
    printf("\n[根据 EXP U/V 结果，用最低带宽的 stride 跑 EXP W]\n");
    printf("[当前使用初始猜测 stride=6144，请根据结果修改]\n");
    expW_buffer_size_verify(d_buf, d_sink, 6144);

    CHECK(cudaFree(d_buf));
    CHECK(cudaFree(d_sink));
    printf("\n[完成]\n");
    printf("\n后续: 根据 EXP U/V 找到真正最差 stride 后，\n");
    printf("      修改 expW_buffer_size_verify 的参数重新验证\n");
    return 0;
}
