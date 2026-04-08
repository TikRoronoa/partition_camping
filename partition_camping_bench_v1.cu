#include <cuda_runtime.h>
#include <stdio.h>

#include <numeric>  // 用于 std::iota
#include <vector>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
            exit(1);                                                      \
        }                                                                 \
    } while (0)

// 禁用L1/L2的线性访问
__global__ void linear_nocache(float* out, const float* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += total) {
        float val;
        asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(val) : "l"(&in[i]));
        float res = val * 2.0f + 1.0f;
        asm volatile("st.global.cg.f32 [%0], %1;" ::"l"(&out[i]), "f"(res));
    }
}

// 禁用L1/L2的partition camping测试
__global__ void camping_nocache(float* out, const float* in, size_t n, size_t partition_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += total) {
        size_t pos = i % partition_size;
        float val;
        asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(val) : "l"(&in[pos]));
        float res = val * 2.0f + 1.0f;
        asm volatile("st.global.cg.f32 [%0], %1;" ::"l"(&out[pos]), "f"(res));
    }
}

// 带缓存版本
__global__ void linear_cached(float* out, const float* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    // #pragma unroll
    for (size_t i = idx; i < n; i += total) {
        out[i] = in[i] * 2.0f + 1.0f;
    }
}

float measure_linear(float* out, float* in, size_t n, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    linear_nocache<<<1024, 256>>>(out, in, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) linear_nocache<<<1024, 256>>>(out, in, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 线性访问：实际传输 n * 2（读+写）
    return (float)n * sizeof(float) * 2.0f * iters / (ms * 1e6);
}

float measure_camping(float* out, float* in, size_t n, size_t psize, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    camping_nocache<<<1024, 256>>>(out, in, n, psize);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) camping_nocache<<<1024, 256>>>(out, in, n, psize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 关键修正：camping时实际只访问了partition_size范围
    // 虽然循环n次，但取模后实际DRAM传输是 psize * 2 * iters
    // 但由于L2缓存，小psize会被缓存，所以实际DRAM流量更低
    // 这里按实际DRAM传输计算（保守估计）
    return (float)psize * sizeof(float) * 2.0f * iters / (ms * 1e6);
}

// 带缓存的 Partition Camping 测试内核
__global__ void camping_cached_kernel(float* out, const float* in, size_t n, size_t partition_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)gridDim.x * blockDim.x;
    for (size_t i = idx; i < n; i += total) {
        size_t pos = i % partition_size;
        // 使用标准 C++ 赋值，允许编译器生成带缓存的 ld.global (L1/L2)
        out[pos] = in[pos] * 2.0f + 1.0f;
    }
}

// 对应的测量函数
float measure_camping_cached(float* out, float* in, size_t n, size_t psize, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热：这一步非常重要，因为我们要把数据填满 L2
    camping_cached_kernel<<<1024, 256>>>(out, in, n, psize);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) camping_cached_kernel<<<1024, 256>>>(out, in, n, psize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (float)psize * sizeof(float) * 2.0f * iters / (ms * 1e6);
}

int main() {
    size_t n = 8 * 1024 * 1024;  // 32MB
    size_t bytes = n * sizeof(float);

    printf("RTX 3080 Partition Camping (No L1/L2 Cache)\n");
    printf("Data: %zu MB\n\n", bytes / 1024 / 1024);

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_in, 0x3F, bytes);

    // 基准：禁用缓存的线性访问
    float base = measure_linear(d_out, d_in, n, 50);
    printf("Linear (no cache): %.2f GB/s\n\n", base);

    // Camping测试（禁用缓存）
    printf("Partition(MB) | Bandwidth(GB/s) | vs Linear | Status\n");
    printf("--------------|-----------------|-----------|--------\n");

    std::vector<float> sizes;
    for (float s = 0.25f; s <= 64.0f; s += 0.25f) {
        sizes.push_back(s);
    }

    for (int i = 0; i < sizes.size(); i++) {
        size_t psize = static_cast<size_t>(sizes[i] * 1024.0f * 1024.0f / sizeof(float));
        if (psize > n) psize = n;
        
#ifndef CACHE
        float bw = measure_camping(d_out, d_in, n, psize, 50);
#else
        float bw = measure_camping_cached(d_out, d_in, n, psize, 50);        
#endif
        float ratio = bw / base;

        const char* status = (ratio < 0.3f) ? "*** CAMPING ***" : (ratio < 0.6f) ? "** SLOW **" : "OK";

        printf("%-13.2f | %-15.2f | %-9.2f | %s\n", sizes[i], bw, ratio, status);
    }

    printf("\nNote: Bandwidth calculated by actual DRAM bytes transferred\n");
    printf("      (partition_size * 2 for read+write)\n");

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}