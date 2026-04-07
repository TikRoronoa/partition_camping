#include <cstdio>
#include <cuda_runtime.h>
#include <cassert>

// ============================================================
// 工具宏
// ============================================================
#define CHECK(call)                                                   \
  do {                                                                \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
              __FILE__, __LINE__, cudaGetErrorString(err));           \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

// ============================================================
// 参数区（你只需要改 STRIDE）
// ============================================================

// 192 * 4B = 768B → 会强制所有请求落到同一 memory partition
// 改成 193 可以立刻“解 camping”
#define STRIDE_FLOATS 193

// 每个 thread 连续发 8 个 outstanding load
#define LOADS_PER_ITER 8

// 产生足够多 in-flight miss
#define ITERS 64

// ============================================================
// Kernel：强制 partition camping
// ============================================================
__global__ void partition_camping_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int total_elems)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float acc = 0.f;

    // 每个 iteration 会制造 LOADS_PER_ITER 个未完成 load
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
        int base =
            tid * STRIDE_FLOATS +
            it * STRIDE_FLOATS * gridDim.x * blockDim.x;

        // 防止越界（但几乎不会触发）
        if (base + (LOADS_PER_ITER - 1) * STRIDE_FLOATS < total_elems) {
#pragma unroll
            for (int j = 0; j < LOADS_PER_ITER; ++j) {
                acc += in[base + j * STRIDE_FLOATS];
            }
        }
    }

    // 防止编译器消除
    if (tid < total_elems)
        out[tid] = acc;
}

// ============================================================
// 计时工具
// ============================================================
float run_kernel(
    const float* d_in,
    float* d_out,
    int total_elems,
    int blocks,
    int threads)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // warmup
    partition_camping_kernel<<<blocks, threads>>>(
        d_in, d_out, total_elems);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaEventRecord(start));
    partition_camping_kernel<<<blocks, threads>>>(
        d_in, d_out, total_elems);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms;
}

// ============================================================
// main
// ============================================================
int main()
{
    // --------------------------------------------------------
    // 设备信息
    // --------------------------------------------------------
    int dev;
    cudaDeviceProp prop;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("Device           : %s\n", prop.name);
    printf("SMs              : %d\n", prop.multiProcessorCount);
    printf("Memory clock     : %.1f MHz\n", prop.memoryClockRate / 1000.f);
    printf("Memory bus width : %d bits\n", prop.memoryBusWidth);
    printf("Stride (floats)  : %d  (bytes = %d)\n",
           STRIDE_FLOATS, STRIDE_FLOATS * 4);
    printf("--------------------------------------------------\n");

    // --------------------------------------------------------
    // launch 参数
    // --------------------------------------------------------
    int threads = 256;                       // 8 warps / block
    int blocks  = prop.multiProcessorCount;  // 1 block / SM

    int total_threads = threads * blocks;

    // 数据规模：足够大，确保 DRAM miss
    size_t total_elems =
        (size_t)total_threads *
        STRIDE_FLOATS *
        ITERS *
        LOADS_PER_ITER;

    size_t bytes = total_elems * sizeof(float);

    printf("Total elements   : %zu\n", total_elems);
    printf("Total bytes      : %.2f MB\n", bytes / 1e6);
    printf("--------------------------------------------------\n");

    // --------------------------------------------------------
    // 分配内存
    // --------------------------------------------------------
    float* d_in  = nullptr;
    float* d_out = nullptr;

    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, total_threads * sizeof(float)));

    CHECK(cudaMemset(d_in, 0, bytes));
    CHECK(cudaMemset(d_out, 0, total_threads * sizeof(float)));

    // --------------------------------------------------------
    // 运行
    // --------------------------------------------------------
    float ms = run_kernel(
        d_in, d_out,
        (int)total_elems,
        blocks, threads);

    // --------------------------------------------------------
    // 带宽计算
    // --------------------------------------------------------
    // 每个 thread 访问：
    // ITERS × LOADS_PER_ITER × 4 bytes
    double bytes_accessed =
        (double)total_threads *
        ITERS *
        LOADS_PER_ITER *
        sizeof(float);

    double bandwidth =
        bytes_accessed / (ms * 1e-3) / 1e9;

    printf("Kernel time      : %.3f ms\n", ms);
    printf("Effective BW     : %.2f GB/s\n", bandwidth);

    // --------------------------------------------------------
    // 清理
    // --------------------------------------------------------
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));

    return 0;
}