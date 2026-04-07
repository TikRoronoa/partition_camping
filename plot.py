import matplotlib.pyplot as plt
import subprocess
import pandas as pd
import io

def run_cuda_bench():
    print("正在运行 CUDA Benchmark...")
    # 假设你的编译产物叫 ./camping_visualizer
    result = subprocess.run(['./partition_camping'], capture_output=True, text=True)
    # 过滤掉表头，只保留数据行
    data = []
    for line in result.stdout.split('\n'):
        if ',' in line:
            data.append(line)
    return "\n".join(data)

# 1. 获取数据 (这里可以用你刚才跑出的数据，或者直接运行)
csv_data = run_cuda_bench()
df = pd.read_csv(io.StringIO(csv_data), names=['LDA', 'BW'])

# 2. 绘图配置
plt.figure(figsize=(15, 7))
plt.plot(df['LDA'], df['BW'], marker='o', markersize=3, linestyle='-', color='#1f77b4', label='Measured Bandwidth')

# 3. 核心：标注 Partition Camping 理论冲突点 (384 floats = 1536 Bytes)
# 对于 384-bit 显存，每 12 个 Partition 循环一次
for i in range(1, 15):
    conflict_lda = 384 * i
    if conflict_lda <= df['LDA'].max():
        plt.axvline(x=conflict_lda, color='red', linestyle='--', alpha=0.5)
        if i == 1:
            plt.text(conflict_lda, plt.ylim()[1]*0.9, 'Partition Conflict (1536B)', color='red', rotation=90)

# 4. 标注 L2 Cache 亲和点 (如 1024, 2048, 4096)
for p2 in [512, 1024, 2048, 4096]:
    if p2 <= df['LDA'].max():
        plt.axvline(x=p2, color='green', linestyle=':', alpha=0.6)
        plt.text(p2, plt.ylim()[1]*0.1, f'L2 Align ({p2})', color='green', rotation=90)

# 5. 图表修饰
plt.title('RTX 3080 Partition Camping Analysis\nStride Read Performance vs LDA', fontsize=14)
plt.xlabel('LDA (Number of Floats)', fontsize=12)
plt.ylabel('Bandwidth (GB/s)', fontsize=12)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.savefig('partition_camping_analysis.png')
plt.show()