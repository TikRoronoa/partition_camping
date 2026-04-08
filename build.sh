#! /usr/bin/zsh

# 1. 编译无缓存版本
nvcc -O3 -arch=sm_86 -lineinfo partition_camping_bench_v1.cu -o partition_camping_no_cached
if [ $? -eq 0 ]; then
    echo "[成功] 编译no cached版本完成: partition_camping_no_cached (L1/L2 Thrashing)"
else
    echo "[错误] 编译no cached版本失败" && exit 1
fi

# 2. 编译开启缓存版本 (-DCACHE)
nvcc -DCACHE -O3 -arch=sm_86 -lineinfo partition_camping_bench_v1.cu -o partition_camping_cached
if [ $? -eq 0 ]; then
    echo "[成功] 编译cached版本完成: partition_camping_cached (L1/L2 Thrashing)"
else
    echo "[错误] 编译cached版本失败" && exit 1
fi

echo "ready to run no cached version......"
./partition_camping_no_cached > ./log_no_cached.txt
if [ $? -eq 0 ]; then
    echo "[pass] from no cached versoin"
else
    echo "[fail] from no cached versoin" && exit 1
fi

echo "ready to run cached version......"
./partition_camping_cached > ./log_cached.txt
if [ $? -eq 0 ]; then
    echo "[pass] from cached versoin"
else
    echo "[fail] from cached versoin" && exit 1
fi