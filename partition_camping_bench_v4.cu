/*
 * page_size_verify_v3.cu
 *
 * 进一步验证 8KB 周期的物理来源
 *
 * 三组实验：
 *   EXP E - 1KB~16KB 细扫描，确认最小高带宽周期
 *   EXP F - 8192 附近字节级精细扫描，测半宽度
 *   EXP G - 所有 N×8KB（N=1~32）完整包络，看次级峰来源
 *
 * 编译: nvcc -O3 -arch=sm_86 -o page_size_verify_v3 page_size_verify_v3.cu
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
static const int    WARMUP    = 5;
static const int    ITERS     = 30;
static const int    THREADS   = 256;
static const int    BLOCKS    = 1024;
static const double PEAK_BW   = 912.0;

/* ─── 核函数（与 v2 完全相同，字节级 stride，float 读取）── */
__global__ void stride_read_byteprec(const float* __restrict__ buf,
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

/* ─── 计时函数 ──────────────────────────────────────────── */
static double measure_bw(const float* d_buf, long stride_bytes, float* d_sink)
{
    long active_n = BUF_BYTES / stride_bytes;
    long min_active = (long)BLOCKS * THREADS * 64;
    if (active_n < min_active) active_n = min_active;

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

/* ─── EXP E: 1KB~16KB 细扫描，找最小高带宽周期 ─────────── */
//
// 关键问题：8KB 以下有没有高带宽点？
//   如果有，说明真实 row size < 8KB
//   如果没有，8KB 是最小触发单位
//
// 步进 256B（分区粒度），覆盖 1KB~16KB
//
static void expE_small_stride_scan(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP E: 1KB~16KB 细扫描 (步进 256B)\n");
    printf("目的: 确认最小高带宽周期，验证 row size\n");
    printf("%-18s %-12s %-14s %-10s\n",
           "stride_bytes", "partitions", "BW(GB/s)", "倍数of256B");
    printf("────────────────────────────────────────────────────────\n");

    for (long sb = 256; sb <= 16384; sb += 256) {
        int    parts = partitions_touched(sb);
        double bw    = measure_bw(d_buf, sb, d_sink);
        long   mult  = sb / 256;
        const char* tag = (bw > 60.0) ? " *** HIGH" : "";
        printf("%-18ld %-12d %-14.1f %-10ld%s\n", sb, parts, bw, mult, tag);
    }
}

/* ─── EXP F: 各个 N×8KB 处的字节级精细扫描 ─────────────── */
//
// 对 N=1,2,3,4 分别做 ±16B 字节级扫描
// 验证：
//   1. 每个 N×8KB 处都有尖峰
//   2. 半宽度是否随 N 线性增大（线性规律验证）
//   3. 8192 处的半宽度是多少（推算 bank_period）
//
static void expF_fine_scan_multiples(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP F: N×8KB 附近字节级精细扫描 (±16B)\n");
    printf("目的: 验证半宽度线性规律，反推 bank_period\n");
    printf("%-18s %-12s %-14s %-10s\n",
           "stride_bytes", "partitions", "BW(GB/s)", "delta");
    printf("────────────────────────────────────────────────────────\n");

    // N=1: 8192, N=2: 16384, N=3: 24576, N=4: 32768
    long centers[] = {8192, 16384, 24576, 32768};
    const char* labels[] = {"1×8KB=8192", "2×8KB=16384",
                             "3×8KB=24576", "4×8KB=32768"};

    for (int ci = 0; ci < 4; ci++) {
        long center = centers[ci];
        printf("\n--- %s ---\n", labels[ci]);
        for (long delta = -16; delta <= 16; delta++) {
            long   sb    = center + delta;
            if (sb <= 0) continue;
            int    parts = partitions_touched(sb);
            double bw    = measure_bw(d_buf, sb, d_sink);
            const char* tag = (bw > 60.0) ? " <-- HIGH" : "";
            printf("%-18ld %-12d %-14.1f %+ld%s\n", sb, parts, bw, delta, tag);
        }
    }
}

/* ─── EXP G: N×8KB 完整包络（N=1~32）────────────────────── */
//
// 测量所有 N×8KB 处的峰值带宽
// 观察包络形状：
//   如果包络周期 = 4（每4个N一个强峰），说明 32KB 是第二层周期
//   如果有更复杂的结构，说明还有更多层
//
static void expG_envelope(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP G: N×8KB 完整包络 (N=1~32)\n");
    printf("目的: 观察包络周期，确认是否有多层周期结构\n");
    printf("%-6s %-18s %-12s %-14s %-10s\n",
           "N", "stride_bytes", "partitions", "BW(GB/s)", "利用率%");
    printf("────────────────────────────────────────────────────────\n");

    for (int n = 1; n <= 32; n++) {
        long   sb    = (long)n * 8192;
        int    parts = partitions_touched(sb);
        double bw    = measure_bw(d_buf, sb, d_sink);
        // 简单的 ASCII 包络图
        int    bars  = (int)(bw / 10.0);
        if (bars > 50) bars = 50;
        const char* tag = (bw > 300.0) ? " *** SUPER" :
                          (bw > 150.0) ? " ** STRONG" :
                          (bw > 80.0)  ? " * MID"     : "";
        printf("N=%-4d %-18ld %-12d %-14.1f %.1f%%%s\n",
               n, sb, parts, bw, bw / PEAK_BW * 100.0, tag);
    }

    // 打印简单包络图
    printf("\n包络示意图 (每格=10 GB/s):\n");
    for (int n = 1; n <= 32; n++) {
        long   sb = (long)n * 8192;
        double bw = measure_bw(d_buf, sb, d_sink);
        int    bars = (int)(bw / 10.0);
        if (bars > 48) bars = 48;
        printf("N=%2d |", n);
        for (int b = 0; b < bars; b++) printf("█");
        printf(" %.0f\n", bw);
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
    printf("\n待验证:\n");
    printf("  1. 8KB 以下是否有高带宽点（确认最小周期）\n");
    printf("  2. 半宽度是否随 N 线性增大\n");
    printf("  3. 包络是否有 32KB 的第二层周期\n");

    float* d_buf;
    CHECK(cudaMalloc(&d_buf, BUF_BYTES));
    CHECK(cudaMemset(d_buf, 0, BUF_BYTES));

    float* d_sink;
    CHECK(cudaMalloc(&d_sink, sizeof(float)));

    expE_small_stride_scan  (d_buf, d_sink);
    expF_fine_scan_multiples(d_buf, d_sink);
    expG_envelope           (d_buf, d_sink);

    CHECK(cudaFree(d_buf));
    CHECK(cudaFree(d_sink));
    printf("\n[完成]\n");
    return 0;
}
