#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // 1. 显存基本参数
        int busWidth = prop.memoryBusWidth;
        float memClockMHz = prop.memoryClockRate / 1000.0f;
        
        // 2. 计算理论带宽 (GB/s)
        double bandwidth = (2.0 * memClockMHz * (busWidth / 8.0)) / 1000.0;

        // 3. 获取 L2 Cache Size (单位: Bytes -> MB)
        float l2CacheMB = (float)prop.l2CacheSize / (1024.0f * 1024.0f);

        std::cout << "--- GPU [" << i << "]: " << prop.name << " ---" << std::endl;
        std::cout << "显存容量:   " << std::fixed << std::setprecision(2) 
                  << (double)prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB" << std::endl;
        std::cout << "显存位宽:   " << busWidth << " bit" << std::endl;
        
        // 核心输出：L2 Size
        std::cout << "L2 缓存:    " << l2CacheMB << " MB" << std::endl;
        
        std::cout << "理论带宽:   " << bandwidth << " GB/s" << std::endl;

        // 针对你的架构分析
        int partitions = busWidth / 32;
        float l2PerPartition = l2CacheMB / partitions;
        
        std::cout << "\n[架构深度解析]" << std::endl;
        std::cout << "物理分区数:  " << partitions << " 个" << std::endl;
        std::cout << "每分区 L2:   " << l2PerPartition << " MB" << std::endl;
        std::cout << "Camping 周期: " << partitions * 256 << " 字节 (关键对齐点)" << std::endl;
        std::cout << "-----------------------------------" << std::endl;
    }

    return 0;
}