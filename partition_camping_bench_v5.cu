/*
 * page_size_verify_v4.cu
 *
 * 用固定 active_n 测量 bank_period
 *
 * 核心改动：所有 stride 使用相同的 active_n
 * 这样半宽度只由 bank_period 决定，排除 active_n 的干扰
 *
 * 实验设计：
 *   固定 active_n = 2^20 = 1048576（足够大，让所有SM忙碌）
 *   在多个 N×8KB 处做字节级扫描
 *   半宽度 W 满足：active_n × W / 4 = bank_period
 *   → bank_period = active_n × W / 4
 *
 * 编译: nvcc -O3 -arch=sm_86 -o page_size_verify_v4 page_size_verify_v4.cu
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
static const long   BUF_BYTES  = 2048L * 1024 * 1024;
static const int    WARMUP     = 5;
static const int    ITERS      = 30;
static const int    THREADS    = 256;
static const int    BLOCKS     = 1024;
static const double PEAK_BW    = 912.0;

// 固定 active_n：所有实验使用同一个值
// 选择依据：
//   足够大 → 所有 SM 都忙碌（BLOCKS×THREADS×64 = 1024×256×64 ≈ 16M，取 2^20 = 1M 偏小）
//   但不能太大 → 避免地址折回导致重复访问污染结果
//   2^23 = 8388608 ≈ 8M 是一个合理的折中
static const long FIXED_ACTIVE_N = 1L << 23;  // 8388608

/* ─── 核函数：固定 active_n 版本 ───────────────────────── */
__global__ void stride_fixed_n(const float* __restrict__ buf,
                                long stride_bytes,
                                long active_n,     // 外部传入，固定值
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

/* ─── 计时函数：固定 active_n ───────────────────────────── */
static double measure_bw_fixed(const float* d_buf,
                                long stride_bytes,
                                long active_n,
                                float* d_sink)
{
    for (int i = 0; i < WARMUP; i++)
        stride_fixed_n<<<BLOCKS, THREADS>>>(d_buf, stride_bytes,
                                             active_n, d_sink);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK(cudaEventCreate(&t0));
    CHECK(cudaEventCreate(&t1));
    CHECK(cudaEventRecord(t0));
    for (int i = 0; i < ITERS; i++)
        stride_fixed_n<<<BLOCKS, THREADS>>>(d_buf, stride_bytes,
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

/* ─── 辅助：找半宽度 ────────────────────────────────────── */
// 给定一组 delta→BW 的测量值，找到带宽跌回底线（<60 GB/s）的最小 |delta|
// 返回半宽度（字节）
static void print_halfwidth_estimate(long center, double peak_bw)
{
    printf("  峰值: %.1f GB/s @ stride=%ld\n", peak_bw, center);
    printf("  半宽度估算: 带宽跌到峰值50%%(%-.0f GB/s)所需的delta\n",
           peak_bw * 0.5);
}

/* ─── EXP H: 固定 active_n，多个中心点的字节级扫描 ─────── */
//
// 对 N=1,2,3,4,8 的 N×8KB 各做 ±32B 字节级扫描
// 所有扫描用同一个 FIXED_ACTIVE_N
// 观察：半宽度是否变成单调递增（排除 active_n 干扰后）
//
static void expH_fixed_n_scan(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP H: 固定 active_n=%ld 的字节级扫描\n", FIXED_ACTIVE_N);
    printf("目的: 排除 active_n 干扰，得到真实半宽度\n");
    printf("理论: bank_period = FIXED_ACTIVE_N × W / 4\n");
    printf("      (W=半宽度字节数，即带宽跌回底线的最小delta)\n");
    printf("────────────────────────────────────────────────────────\n");

    long centers[] = {8192, 16384, 24576, 32768, 65536};
    const char* labels[] = {
        "N=1: 1×8KB=8192",
        "N=2: 2×8KB=16384",
        "N=3: 3×8KB=24576",
        "N=4: 4×8KB=32768",
        "N=8: 8×8KB=65536",
    };

    for (int ci = 0; ci < 5; ci++) {
        long center = centers[ci];
        printf("\n--- %s ---\n", labels[ci]);
        printf("%-18s %-12s %-14s %-10s\n",
               "stride_bytes", "partitions", "BW(GB/s)", "delta");
        printf("────────────────────────────────────────\n");

        double peak = 0.0;
        for (long delta = -32; delta <= 32; delta++) {
            long   sb    = center + delta;
            if (sb <= 0) continue;
            int    parts = partitions_touched(sb);
            double bw    = measure_bw_fixed(d_buf, sb, FIXED_ACTIVE_N, d_sink);
            if (bw > peak) peak = bw;
            const char* tag = (bw > 60.0) ? " <--" : "";
            printf("%-18ld %-12d %-14.1f %+ld%s\n", sb, parts, bw, delta, tag);
        }
        print_halfwidth_estimate(center, peak);
    }
}

/* ─── EXP I: 不同 active_n 下同一峰的半宽度变化 ────────── */
//
// 固定中心点 = 65536（已知峰值465 GB/s）
// 用不同的 active_n 做字节级扫描
// 验证：半宽度是否与 1/active_n 成正比
// 如果是，就能用任意一个 active_n 精确反推 bank_period
//
static void expI_active_n_sweep(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP I: 不同 active_n 下 65536 峰的半宽度\n");
    printf("目的: 验证半宽度 ∝ 1/active_n，精确反推 bank_period\n");
    printf("────────────────────────────────────────────────────────\n");

    // 测试不同的 active_n（都是2的幂次，方便分析）
    long active_ns[] = {
        1L << 20,  //  1M
        1L << 21,  //  2M
        1L << 22,  //  4M
        1L << 23,  //  8M
        1L << 24,  // 16M
        1L << 25,  // 32M
    };

    long center = 65536;

    for (int ai = 0; ai < 6; ai++) {
        long an = active_ns[ai];
        printf("\n--- active_n = %ld (2^%d) ---\n", an, 20 + ai);
        printf("%-18s %-14s %-10s\n", "stride_bytes", "BW(GB/s)", "delta");
        printf("────────────────────────────────────────\n");

        // 只扫 ±16B，节省时间
        for (long delta = -16; delta <= 16; delta++) {
            long   sb = center + delta;
            double bw = measure_bw_fixed(d_buf, sb, an, d_sink);
            const char* tag = (bw > 60.0) ? " <--" : "";
            printf("%-18ld %-14.1f %+ld%s\n", sb, bw, delta, tag);
        }

        // 计算这个 active_n 下观测到的半宽度，并推算 bank_period
        printf("  若半宽度=W字节，则 bank_period = active_n × W / 4 = %ld × W / 4\n", an);
    }
}

/* ─── EXP J: 对照，验证固定 active_n 不影响峰值位置 ────── */
//
// 用 v3 的可变 active_n 和 v4 的固定 active_n 对比峰值带宽
// 确认两种方式测到的峰值位置相同（只是半宽度不同）
//
static void expJ_sanity(const float* d_buf, float* d_sink)
{
    printf("\n════════════════════════════════════════════════════════\n");
    printf("EXP J: 固定 vs 可变 active_n 对照\n");
    printf("%-18s %-16s %-16s\n",
           "stride_bytes", "固定active_n", "可变active_n");
    printf("────────────────────────────────────────────────────────\n");

    // 可变 active_n 计算（与 v2/v3 相同）
    auto var_active_n = [](long sb) -> long {
        long an = BUF_BYTES / sb;
        long min_an = (long)BLOCKS * THREADS * 64;
        if (an < min_an) an = min_an;
        return an;
    };

    long test_strides[] = {8192, 16384, 32768, 65536, 131072, 196608};
    for (int i = 0; i < 6; i++) {
        long sb  = test_strides[i];
        double bw_fixed = measure_bw_fixed(d_buf, sb, FIXED_ACTIVE_N, d_sink);
        double bw_var   = measure_bw_fixed(d_buf, sb, var_active_n(sb), d_sink);
        printf("%-18ld %-16.1f %-16.1f\n", sb, bw_fixed, bw_var);
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
    printf("\nFIXED_ACTIVE_N = %ld = 2^23\n", FIXED_ACTIVE_N);
    printf("理论半宽度公式: W = bank_period × 4 / FIXED_ACTIVE_N\n");
    printf("若 bank_period=128KB=131072B: W = 131072×4/%ld = %.2fB\n",
           FIXED_ACTIVE_N, 131072.0 * 4 / FIXED_ACTIVE_N);

    float* d_buf;
    CHECK(cudaMalloc(&d_buf, BUF_BYTES));
    CHECK(cudaMemset(d_buf, 0, BUF_BYTES));

    float* d_sink;
    CHECK(cudaMalloc(&d_sink, sizeof(float)));

    expJ_sanity          (d_buf, d_sink);  // 先验证固定active_n不影响峰值
    expH_fixed_n_scan    (d_buf, d_sink);  // 核心：固定active_n的半宽度
    expI_active_n_sweep  (d_buf, d_sink);  // 验证半宽度∝1/active_n

    CHECK(cudaFree(d_buf));
    CHECK(cudaFree(d_sink));
    printf("\n[完成]\n");
    return 0;
}
