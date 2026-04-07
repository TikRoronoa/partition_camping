/**
 * partition_camping_bench_v2.cu
 * ─────────────────────────────────────────────────────────────────────────────
 * RTX 3080 12GB (384-bit / 12 Partitions) Partition Camping 验证 —— 修复版
 *
 * v1 的问题：行读取是顺序流式访存，每行内部天然均匀分布到 12 个分区，
 *            即使行首都在 P0，也不会造成分区热点。
 *
 * v2 的修复：改用"列读取"（stride = LDA），让每次访存都跳 LDA 个元素，
 *            这时不同行同列的地址会反复命中同一批分区，才真正触发 Camping。
 *
 * 三个 kernel：
 *   1. col_read_kernel   —— 每个 block 读一列（stride=LDA），核心场景
 *   2. col_write_kernel  —— 每个 block 写一列，压力更大
 *   3. transpose_kernel  —— 经典转置：读列写行，cuBLAS 最常遇到的场景
 *
 * 编译：
 *   nvcc -O3 -arch=sm_86 -o bench_v2 partition_camping_bench_v2.cu -lnvToolsExt
 *
 * 运行：
 *   ./bench_v2
 *
 * Nsight Systems 捕获：
 *   nsys profile --trace=cuda,nvtx --output=camping_v2 ./bench_v2
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <nvToolsExt.h>

// ── 参数 ────────────────────────────────────────────────────────────────────
static constexpr int   NROWS        = 4096;   // 行数（列读时这是"步幅跳的次数"）
static constexpr int   NCOLS_BASE   = 512;    // 列数（被读的列的总数）
static constexpr int   WARMUP       = 5;
static constexpr int   ITERS        = 30;
static constexpr int   N_PART       = 12;
static constexpr int   GRANULE_B    = 16;
static constexpr int   STRIPE_B     = N_PART * GRANULE_B; // 192 bytes

#define CUDA_CHECK(expr) \
    do { cudaError_t _e=(expr); if(_e!=cudaSuccess){ \
        fprintf(stderr,"[CUDA] %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); \
        exit(1); } } while(0)

// ────────────────────────────────────────────────────────────────────────────
// Kernel 1：列读取
//   每个 block 负责读矩阵的第 blockIdx.x 列，访问地址步长 = lda。
//   当 lda 是 48 的倍数时，(row × lda × 4) mod 192 = 0 → 全部打到 P0。
// ────────────────────────────────────────────────────────────────────────────
__global__ void col_read_kernel(const float* __restrict__ mat,
                                float*       __restrict__ sink,
                                int lda, int nrows, int ncols)
{
    int col = blockIdx.x;
    if (col >= ncols) return;

    float acc = 0.f;
    // 每个线程处理多行，步进 blockDim.x
    for (int row = threadIdx.x; row < nrows; row += blockDim.x)
        acc += mat[(long long)row * lda + col];

    for (int mask = 16; mask > 0; mask >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, mask);

    if (threadIdx.x == 0)
        atomicAdd(sink, acc);
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel 2：列写入
// ────────────────────────────────────────────────────────────────────────────
__global__ void col_write_kernel(float*       __restrict__ mat,
                                 const float* __restrict__ src,
                                 int lda, int nrows, int ncols)
{
    int col = blockIdx.x;
    if (col >= ncols) return;

    float val = src[col]; // 从 src 读一个值
    for (int row = threadIdx.x; row < nrows; row += blockDim.x)
        mat[(long long)row * lda + col] = val;
}

// ────────────────────────────────────────────────────────────────────────────
// Kernel 3：Tiled 矩阵转置（标准 shared memory 版）
//   读 B[col][row]（列访问），写 C[row][col]（行访问）
//   shared memory 消除写侧的 bank conflict，但读侧的显存 partition camping 留着
// ────────────────────────────────────────────────────────────────────────────
#define TILE 32
__global__ void transpose_kernel(const float* __restrict__ in,
                                 float*       __restrict__ out,
                                 int rows, int cols, int lda_in, int lda_out)
{
    __shared__ float tile[TILE][TILE + 1]; // +1 避免 shared bank conflict

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[(long long)y * lda_in + x];
    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < rows && y < cols)
        out[(long long)y * lda_out + x] = tile[threadIdx.x][threadIdx.y];
}

// ────────────────────────────────────────────────────────────────────────────
// 通用计时包装
// ────────────────────────────────────────────────────────────────────────────
struct Timer {
    cudaEvent_t e0, e1;
    Timer() { cudaEventCreate(&e0); cudaEventCreate(&e1); }
    ~Timer() { cudaEventDestroy(e0); cudaEventDestroy(e1); }
    void start() { cudaEventRecord(e0); }
    float stop_ms() { cudaEventRecord(e1); cudaEventSynchronize(e1); float ms=0; cudaEventElapsedTime(&ms,e0,e1); return ms; }
};

// ────────────────────────────────────────────────────────────────────────────
// 测量：列读取带宽
// ────────────────────────────────────────────────────────────────────────────
float bench_col_read(const float* d_mat, float* d_sink, int lda, int nrows, int ncols)
{
    dim3 grid(ncols), block(256);
    Timer t;
    for (int i = 0; i < WARMUP; i++)
        col_read_kernel<<<grid, block>>>(d_mat, d_sink, lda, nrows, ncols);
    cudaDeviceSynchronize();

    t.start();
    for (int i = 0; i < ITERS; i++)
        col_read_kernel<<<grid, block>>>(d_mat, d_sink, lda, nrows, ncols);
    float ms = t.stop_ms() / ITERS;

    double bytes = (double)nrows * ncols * sizeof(float);
    return (float)(bytes / (ms * 1e-3) / 1e9);
}

// ────────────────────────────────────────────────────────────────────────────
// 测量：列写入带宽
// ────────────────────────────────────────────────────────────────────────────
float bench_col_write(float* d_mat, const float* d_src, int lda, int nrows, int ncols)
{
    dim3 grid(ncols), block(256);
    Timer t;
    for (int i = 0; i < WARMUP; i++)
        col_write_kernel<<<grid, block>>>(d_mat, d_src, lda, nrows, ncols);
    cudaDeviceSynchronize();

    t.start();
    for (int i = 0; i < ITERS; i++)
        col_write_kernel<<<grid, block>>>(d_mat, d_src, lda, nrows, ncols);
    float ms = t.stop_ms() / ITERS;

    double bytes = (double)nrows * ncols * sizeof(float);
    return (float)(bytes / (ms * 1e-3) / 1e9);
}

// ────────────────────────────────────────────────────────────────────────────
// 测量：转置带宽（以读侧字节数计）
// ────────────────────────────────────────────────────────────────────────────
float bench_transpose(const float* d_in, float* d_out, int nrows, int ncols, int lda_in, int lda_out)
{
    dim3 block(TILE, TILE);
    dim3 grid((ncols + TILE - 1) / TILE, (nrows + TILE - 1) / TILE);
    Timer t;
    for (int i = 0; i < WARMUP; i++)
        transpose_kernel<<<grid, block>>>(d_in, d_out, nrows, ncols, lda_in, lda_out);
    cudaDeviceSynchronize();

    t.start();
    for (int i = 0; i < ITERS; i++)
        transpose_kernel<<<grid, block>>>(d_in, d_out, nrows, ncols, lda_in, lda_out);
    float ms = t.stop_ms() / ITERS;

    double bytes = 2.0 * nrows * ncols * sizeof(float); // 读 + 写
    return (float)(bytes / (ms * 1e-3) / 1e9);
}

// ────────────────────────────────────────────────────────────────────────────
int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("  GPU: %s | 位宽: %d-bit | 峰值带宽: %.1f GB/s\n",
           prop.name, prop.memoryBusWidth, peak_bw);
    printf("  NROWS=%d  NCOLS=%d  ITERS=%d\n", NROWS, NCOLS_BASE, ITERS);
    printf("  冲突条件: LDA × 4 bytes ≡ 0 (mod %d)  即 LDA ≡ 0 (mod 48)\n", STRIPE_B);
    printf("═══════════════════════════════════════════════════════════════════════\n\n");

    // ── 分配最大显存（按最大 LDA）─────────────────────────────────────────
    const int LDA_MAX = 8256; // 8192 + 64 padding
    size_t mat_bytes = (size_t)NROWS * LDA_MAX * sizeof(float);
    float *d_mat, *d_out, *d_sink, *d_src;
    CUDA_CHECK(cudaMalloc(&d_mat,  mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,  mat_bytes));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_src,  NCOLS_BASE * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_mat,  1, mat_bytes));
    CUDA_CHECK(cudaMemset(d_out,  0, mat_bytes));
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_src,  1, NCOLS_BASE * sizeof(float)));

    // ════════════════════════════════════════════════════════════════════════
    // 第一部分：列读取带宽扫描
    // ════════════════════════════════════════════════════════════════════════
    printf("【列读取扫描】kernel=col_read  stride=LDA（真正触发 Partition Camping）\n");
    printf("%-8s  %-14s  %-10s  %-8s  %s\n",
           "LDA", "带宽(GB/s)", "峰值占比", "冲突?", "LDA mod 48");
    printf("────────  ──────────────  ──────────  ────────  ──────────\n");

    struct ScanResult { int lda; float bw; bool conflict; };
    std::vector<ScanResult> scan_results;

    // 扫描范围：4096~8192，步长 48（每个冲突周期）内取有代表性的点
    // 同时每个周期取 [冲突, +16, +32] 三个点做对比
    for (int base = 4096; base <= 8192; base += 48 * 4) { // 每 4 个周期取一组
        for (int off : {0, 8, 24, 40}) { // 0=冲突, 其余=OK
            int lda = base + off;
            if (lda > LDA_MAX - 64) continue;

            char tag[64];
            snprintf(tag, sizeof(tag), "col_read_LDA=%d", lda);
            nvtxRangePushA(tag);
            float bw = bench_col_read(d_mat, d_sink, lda, NROWS, NCOLS_BASE);
            nvtxRangePop();

            bool conflict = ((lda * 4) % STRIPE_B) == 0;
            scan_results.push_back({lda, bw, conflict});
            printf("%-8d  %-14.2f  %-10.1f%%  %-8s  %d\n",
                   lda, bw, bw / peak_bw * 100.0,
                   conflict ? "★ 冲突" : "  OK",
                   (lda * 4) % STRIPE_B);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // 第二部分：精细对比（列读 / 列写 / 转置）
    // ════════════════════════════════════════════════════════════════════════
    printf("\n════════════════════════════════════════════════════════════════════════\n");
    printf("【精细对比】冲突 LDA vs +8 padding — 三种 kernel\n");
    printf("════════════════════════════════════════════════════════════════════════\n");

    struct TestPair { int bad; int good; };
    TestPair pairs[] = { {4608,4616}, {6144,6152}, {7680,7688}, {4800,4808} };

    for (auto& p : pairs) {
        printf("\n  ─── LDA %d (冲突) vs LDA %d (+8 pad) ───\n", p.bad, p.good);

        // 列读
        {
            nvtxRangePushA("col_read_conflict"); float b_bad  = bench_col_read(d_mat, d_sink, p.bad,  NROWS, NCOLS_BASE); nvtxRangePop();
            nvtxRangePushA("col_read_ok");       float b_good = bench_col_read(d_mat, d_sink, p.good, NROWS, NCOLS_BASE); nvtxRangePop();
            float pct = (b_good - b_bad) / b_bad * 100.f;
            printf("  列读取:   冲突 %6.2f GB/s  修复 %6.2f GB/s  差距 %+.1f%%\n",
                   b_bad, b_good, pct);
        }
        // 列写
        {
            nvtxRangePushA("col_write_conflict"); float b_bad  = bench_col_write(d_mat, d_src, p.bad,  NROWS, NCOLS_BASE); nvtxRangePop();
            nvtxRangePushA("col_write_ok");       float b_good = bench_col_write(d_mat, d_src, p.good, NROWS, NCOLS_BASE); nvtxRangePop();
            float pct = (b_good - b_bad) / b_bad * 100.f;
            printf("  列写入:   冲突 %6.2f GB/s  修复 %6.2f GB/s  差距 %+.1f%%\n",
                   b_bad, b_good, pct);
        }
        // 转置
        {
            int ncols_t = NCOLS_BASE, nrows_t = NROWS;
            nvtxRangePushA("transpose_conflict"); float b_bad  = bench_transpose(d_mat, d_out, nrows_t, ncols_t, p.bad,  ncols_t); nvtxRangePop();
            nvtxRangePushA("transpose_ok");       float b_good = bench_transpose(d_mat, d_out, nrows_t, ncols_t, p.good, ncols_t); nvtxRangePop();
            float pct = (b_good - b_bad) / b_bad * 100.f;
            printf("  矩阵转置: 冲突 %6.2f GB/s  修复 %6.2f GB/s  差距 %+.1f%%\n",
                   b_bad, b_good, pct);
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // 第三部分：统计摘要
    // ════════════════════════════════════════════════════════════════════════
    printf("\n════════════════════════════════════════════════════════════════════════\n");
    printf("【摘要统计】列读取 kernel\n");

    float sum_c=0, sum_o=0;
    int   cnt_c=0, cnt_o=0;
    float min_c=1e9, max_o=0;
    for (auto& r : scan_results) {
        if (r.conflict) { sum_c+=r.bw; cnt_c++; min_c=std::min(min_c,r.bw); }
        else            { sum_o+=r.bw; cnt_o++; max_o=std::max(max_o,r.bw); }
    }
    if (cnt_c && cnt_o) {
        float avg_c = sum_c/cnt_c, avg_o = sum_o/cnt_o;
        printf("  冲突 LDA: %d 个  平均 %.2f GB/s  最低 %.2f GB/s\n", cnt_c, avg_c, min_c);
        printf("  正常 LDA: %d 个  平均 %.2f GB/s  最高 %.2f GB/s\n", cnt_o, avg_o, max_o);
        printf("  性能差距（正常/冲突）: %.3fx  (= %.1f%% 带宽损失)\n",
               avg_o/avg_c, (avg_o-avg_c)/avg_o*100.f);
    }
    printf("════════════════════════════════════════════════════════════════════════\n\n");

    // ════════════════════════════════════════════════════════════════════════
    // 附：打印各 LDA 的分区映射（前 8 行）
    // ════════════════════════════════════════════════════════════════════════
    printf("【附：行首地址分区映射】（前 8 行，bad=4608 vs good=4616）\n");
    printf("行号  bad LDA=4608 → 分区  good LDA=4616 → 分区\n");
    for (int row = 0; row < 8; row++) {
        long long addr_bad  = (long long)row * 4608 * 4;
        long long addr_good = (long long)row * 4616 * 4;
        int p_bad  = (int)((addr_bad  % STRIPE_B) / GRANULE_B);
        int p_good = (int)((addr_good % STRIPE_B) / GRANULE_B);
        printf("  行%d  addr=%8lldB → P%-2d       addr=%8lldB → P%-2d\n",
               row, addr_bad, p_bad, addr_good, p_good);
    }

    CUDA_CHECK(cudaFree(d_mat));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_src));
    printf("\n完成。\n");
    return 0;
}
