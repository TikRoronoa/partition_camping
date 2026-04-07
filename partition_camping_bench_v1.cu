/**
 * partition_camping_bench.cu
 * ─────────────────────────────────────────────────────────────────────────────
 * 验证 RTX 3080 12GB (384-bit, 12 Memory Partitions) 上的 Partition Camping 现象。
 *
 * 测量策略：
 *   对给定 LDA，让每个 warp 顺序读取矩阵的一整行（= LDA 个 FP32 元素）。
 *   当所有行的首地址都映射到同一个显存分区时，访存队列会在该分区堆积，
 *   导致有效带宽大幅下降。
 *
 * 编译（需要 CUDA 12+，Compute Capability 8.6）：
 *   nvcc -O3 -arch=sm_86 -o bench partition_camping_bench.cu -lnvToolsExt
 *
 * 运行：
 *   ./bench
 *
 * 可选：配合 Nsight Systems 捕获：
 *   nsys profile --trace=cuda,nvtx ./bench
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

// ── 编译期常量 ──────────────────────────────────────────────────────────────
static constexpr int   NROWS        = 1024;      // 矩阵行数（越大越稳定）
static constexpr int   WARMUP_ITERS = 5;
static constexpr int   BENCH_ITERS  = 20;
static constexpr int   N_PARTITIONS = 12;
static constexpr int   GRANULE_B    = 16;        // 每分区粒度 (bytes)
static constexpr int   STRIPE_B     = N_PARTITIONS * GRANULE_B; // 192 bytes

// ── 错误检查宏 ──────────────────────────────────────────────────────────────
#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ── Kernel：每个 block 读取矩阵的一行，累加到 sink 防止编译器优化掉 ─────────
__global__ void row_read_kernel(const float* __restrict__ mat,
                                float*       __restrict__ sink,
                                int lda, int nrows)
{
    int row = blockIdx.x;
    if (row >= nrows) return;

    const float* row_ptr = mat + (long long)row * lda;

    float acc = 0.f;
    // 每个线程步进 blockDim.x，覆盖整行
    for (int col = threadIdx.x; col < lda; col += blockDim.x)
        acc += row_ptr[col];

    // warp reduce
    for (int mask = 16; mask > 0; mask >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, mask);

    if (threadIdx.x == 0)
        atomicAdd(sink, acc);
}

// ── 测量单个 LDA 的有效带宽 (GB/s) ─────────────────────────────────────────
struct BenchResult {
    int   lda;
    float bw_gbs;       // 有效带宽
    int   camping_part; // 行首落入的分区（-1 = 分散）
    bool  is_conflict;
};

BenchResult measure_lda(int lda)
{
    // 分配显存：NROWS × lda 个 float
    size_t bytes = (size_t)NROWS * lda * sizeof(float);
    float *d_mat, *d_sink;
    CUDA_CHECK(cudaMalloc(&d_mat,  bytes));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mat,  0, bytes));
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(float)));

    // 初始化：填 1.0f
    // 用 cudaMemset 只能设 0，改用 kernel 或 cudaMemsetAsync + host memset
    {
        std::vector<float> h(lda, 1.f);
        for (int r = 0; r < NROWS; r++)
            CUDA_CHECK(cudaMemcpy(d_mat + (long long)r * lda,
                                  h.data(), lda * sizeof(float),
                                  cudaMemcpyHostToDevice));
    }

    dim3 grid(NROWS), block(256);
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++)
        row_read_kernel<<<grid, block>>>(d_mat, d_sink, lda, NROWS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    char tag[64];
    snprintf(tag, sizeof(tag), "LDA=%d", lda);
    nvtxRangePushA(tag);

    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < BENCH_ITERS; i++)
        row_read_kernel<<<grid, block>>>(d_mat, d_sink, lda, NROWS);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    nvtxRangePop();

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    ms /= BENCH_ITERS;

    // 有效读取量 = NROWS × lda × 4 bytes
    double read_bytes = (double)NROWS * lda * sizeof(float);
    float  bw_gbs     = (float)(read_bytes / (ms * 1e-3) / 1e9);

    // 判断是否冲突：(lda × 4) mod 192 == 0
    int lda_bytes = lda * (int)sizeof(float);
    bool conflict = (lda_bytes % STRIPE_B) == 0;
    int  part0    = (int)((0LL % STRIPE_B) / GRANULE_B); // row 0 总是 P0
    // row 1 的分区：若 conflict，仍然 P0
    long long row1_byte = (long long)lda_bytes % STRIPE_B;
    int part1 = (int)(row1_byte / GRANULE_B);

    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    BenchResult r;
    r.lda         = lda;
    r.bw_gbs      = bw_gbs;
    r.camping_part = conflict ? 0 : part1;
    r.is_conflict  = conflict;
    return r;
}

// ── 主函数 ──────────────────────────────────────────────────────────────────
int main()
{
    // 打印 GPU 信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  GPU: %s  |  显存: %.0f GB  |  位宽: %d-bit\n",
           prop.name,
           prop.totalGlobalMem / 1e9,
           prop.memoryBusWidth);
    printf("  理论峰值带宽: %.1f GB/s\n",
           2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9);
    printf("  N_PARTITIONS=%d  GRANULE=%d B  STRIPE=%d B\n",
           N_PARTITIONS, GRANULE_B, STRIPE_B);
    printf("  矩阵行数 NROWS=%d  迭代次数=%d\n", NROWS, BENCH_ITERS);
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // ── 第一部分：系统扫描 LDA 4096~8192，步长 64 ──────────────────────────
    printf("【扫描段】LDA 4096 → 8192  (步长 64)\n");
    printf("%-8s  %-12s  %-8s  %s\n", "LDA", "带宽(GB/s)", "冲突?", "行首分区");
    printf("────────  ────────────  ────────  ────────\n");

    std::vector<BenchResult> results;
    for (int lda = 4096; lda <= 8192; lda += 64) {
        BenchResult r = measure_lda(lda);
        results.push_back(r);
        printf("%-8d  %-12.2f  %-8s  P%d\n",
               r.lda, r.bw_gbs,
               r.is_conflict ? "★ 冲突" : "  OK",
               r.camping_part);
        fflush(stdout);
    }

    // ── 第二部分：精细对比 —— 典型冲突 vs 加 padding 修复 ─────────────────
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("【精细对比】冲突 LDA vs 加 +8 padding 修复\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    struct Pair { int bad; int good; const char* note; };
    Pair pairs[] = {
        { 4608, 4616, "4608 (典型冲突)  vs  4616 (+8 pad)" },
        { 6144, 6152, "6144 (严重冲突)  vs  6152 (+8 pad)" },
        { 4800, 4808, "4800 (冲突)      vs  4808 (+8 pad)" },
        { 7680, 7688, "7680 (冲突)      vs  7688 (+8 pad)" },
    };

    for (auto& p : pairs) {
        BenchResult rbad  = measure_lda(p.bad);
        BenchResult rgood = measure_lda(p.good);
        float delta = rgood.bw_gbs - rbad.bw_gbs;
        float pct   = delta / rbad.bw_gbs * 100.f;
        printf("\n  %s\n", p.note);
        printf("    冲突  LDA=%-6d  带宽= %6.2f GB/s\n", p.bad,  rbad.bw_gbs);
        printf("    修复  LDA=%-6d  带宽= %6.2f GB/s  (+%.1f%%)\n",
               p.good, rgood.bw_gbs, pct);
    }

    // ── 第三部分：统计摘要 ────────────────────────────────────────────────
    printf("\n═══════════════════════════════════════════════════════════════\n");
    printf("【摘要统计】\n");

    float sum_conflict = 0, sum_ok = 0;
    int   cnt_conflict = 0, cnt_ok = 0;
    float min_conflict = 1e9, max_ok = 0;

    for (auto& r : results) {
        if (r.is_conflict) { sum_conflict += r.bw_gbs; cnt_conflict++; min_conflict = std::min(min_conflict, r.bw_gbs); }
        else               { sum_ok       += r.bw_gbs; cnt_ok++;       max_ok       = std::max(max_ok,       r.bw_gbs); }
    }

    if (cnt_conflict > 0 && cnt_ok > 0) {
        float avg_c = sum_conflict / cnt_conflict;
        float avg_o = sum_ok       / cnt_ok;
        printf("  冲突 LDA 数量: %d  平均带宽: %.2f GB/s  最低: %.2f GB/s\n",
               cnt_conflict, avg_c, min_conflict);
        printf("  正常 LDA 数量: %d  平均带宽: %.2f GB/s  最高: %.2f GB/s\n",
               cnt_ok, avg_o, max_ok);
        printf("  性能差距（正常/冲突）: %.2fx\n", avg_o / avg_c);
    }

    printf("═══════════════════════════════════════════════════════════════\n");
    printf("完成。\n");
    return 0;
}
