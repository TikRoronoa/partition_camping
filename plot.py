import matplotlib.pyplot as plt
import pandas as pd
import io

# 1. 完整数据输入 (由于数据量大，这里截取关键段落，建议运行时替换为你剪贴板的完整内容)
raw_data = """MB,BW
0.25,15.43
0.50,30.92
0.75,46.46
1.00,61.80
1.25,77.45
1.50,92.93
1.75,108.43
2.00,123.90
2.25,139.47
2.50,155.06
2.75,169.84
3.00,184.50
3.25,168.68
3.50,193.21
3.75,160.74
4.00,225.55
4.25,156.55
4.50,168.11
4.75,175.22
5.00,262.71
5.25,164.86
5.50,157.71
5.75,185.58
6.00,281.42
8.00,301.31
16.00,391.13
24.00,561.35
32.00,746.79
64.00,746.59"""

# 读取数据
df = pd.read_csv(io.StringIO(raw_data))

# 2. 绘图设置
plt.figure(figsize=(16, 9), dpi=150) # 设置高分辨率
plt.style.use('dark_background') # 使用深色主题，更有科技感

# 3. 绘制主曲线
plt.plot(df['MB'], df['BW'], color='#00ffcc', linewidth=2, label='Measured Bandwidth')
# 填充曲线下方，增加视觉效果
plt.fill_between(df['MB'], df['BW'], color='#00ffcc', alpha=0.1)

# 4. 绘制参考线
theoretical_max = 760.50
plt.axhline(y=theoretical_max, color='red', linestyle='--', alpha=0.6, label=f'Linear Peak ({theoretical_max} GB/s)')

# 5. 标注关键区域
# 标注 Camping 严重的低效区
plt.axvspan(3.25, 8.5, color='orange', alpha=0.2, label='High Conflict Zone')

# 标注 32MB 稳定点
plt.axvline(x=32, color='#ffffff', linestyle=':', alpha=0.8)
plt.annotate('Full Hash Distribution (32MB)', xy=(32, 740), xytext=(38, 600),
             arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
             fontsize=12, color='white')

# 6. 图表细节美化
plt.title('RTX 3080 Memory Partition Camping Analysis', fontsize=20, pad=20, color='#00ffcc')
plt.xlabel('Partition Size (MB)', fontsize=14)
plt.ylabel('Effective Bandwidth (GB/s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(loc='lower right', fontsize=12)

# 限制坐标轴
plt.xlim(0, 64)
plt.ylim(0, 850)

# 7. 保存为 PNG
plt.tight_layout()
plt.savefig('rtx3080_camping_analysis.png', bbox_inches='tight')
print("图片已保存为: rtx3080_camping_analysis.png")

plt.show()