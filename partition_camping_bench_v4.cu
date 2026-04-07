#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

// CUDA Kernel: 模拟跨步访问
// stride: 访问跨度（单位：sizeof(int)）
__global__ void camping_kernel(int* data, int n, int stride, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程按照特定的 Stride 进行多次访问
    // 计算索引，确保不同线程落在不同的位置，但跨度固定
    size_t index = (size_t)tid * stride;
    
    int val = 0;
    for (int i = 0; i < iterations; ++i) {
        // 使用取模模拟循环读取，防止越界，同时产生持续压力
        val += data[(index + i) % n];
    }
    
    // 防止编译器优化掉读取操作
    if (val == 0x7FFFFFFF) data[0] = val;
}

void run_test(int stride_bytes) {
    const int n = 1 << 26; // 约 256MB 的测试空间
    const int iterations = 100;
    size_t size = n * sizeof(int);
    
    int *d_data;
    cudaMalloc(&d_data, size);
    cudaMemset(d_data, 1, size);

    int threads = 256;
    int blocks = 1024; // 启动足够的线程块以填满 SM
    int stride_elements = stride_bytes / sizeof(int);

    // 预热
    camping_kernel<<<blocks, threads>>>(d_data, n, stride_elements, 1);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    camping_kernel<<<blocks, threads>>>(d_data, n, stride_elements, iterations);
    cudaEventRecord(stop);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 计算有效吞吐量 (GB/s)
    // 每个线程读取了 iterations 次 int
    double total_bytes = (double)blocks * threads * iterations * sizeof(int);
    double bandwidth = (total_bytes / (milliseconds / 1000.0)) / 1e9;

    std::cout << std::setw(12) << stride_bytes << " B | "
              << std::setw(12) << milliseconds << " ms | "
              << std::setw(12) << bandwidth << " GB/s" << std::endl;

    cudaFree(d_data);
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) return 1;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << " (384-bit / 12 Controllers expected)\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "  Stride Size  |   Time (ms)  |   Throughput  \n";
    std::cout << "--------------------------------------------------------\n";

    // 测试从 256B 到 64KB 的步长
    // 观察 1KB, 2KB, 4KB, 8KB 等 2 的幂次方位置
    for (int i = 8; i <= 16; ++i) {
        run_test(1 << i);
    }

    return 0;
}