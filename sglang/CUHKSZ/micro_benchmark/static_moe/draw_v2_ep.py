import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import numpy as np

# 数据
batch_size = [
    2, 4, 6, 8, 10, 12, 14, 16,
    18, 20, 22, 24, 26, 28, 30, 32,
    64, 128, 256, 512, 1024, 2048,
    3072, 4096, 5120, 6144, 8192, 16384
]
epmoe = [
    0.326672, 0.328192, 0.331744, 0.325280, 0.375888, 0.410784, 0.452160, 0.505344,
    0.536800, 0.574720, 0.624800, 0.684128, 0.733440, 0.761440, 0.805648, 0.843520,
    0.860000, 0.852192, 0.837552, 0.821552, 0.836160, 0.839280,
    1.475040, 1.521936, 2.484112, 2.543808, 3.744208, 8.992640
]

# 子集数据（只到1024）
batch_size_subset = batch_size[:21]
epmoe_subset = epmoe[:21]

# 字体设置
plt.rcParams.update({
    "font.size": 20,        # 全局字体大小
    "axes.titlesize": 24,   # 子图标题字体大小
    "axes.labelsize": 22,   # 坐标轴标签字体大小
    "xtick.labelsize": 20,  # x轴刻度字体大小
    "ytick.labelsize": 20,  # y轴刻度字体大小
    "legend.fontsize": 18   # 图例字体大小
})

# 创建左右子图
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# 子图1：子集
axes[0].plot(batch_size_subset, epmoe_subset, marker="o", label="EPMoE (subset)")
axes[0].set_xscale("log", base=2)
axes[0].xaxis.set_major_locator(LogLocator(base=2))
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x)}" if x > 0 and np.isclose(x, 2 ** int(np.log2(x))) else ""))
axes[0].set_xlim(2, 1024)
axes[0].set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
axes[0].set_xticklabels(["2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"]) 
axes[0].margins(x=0.02)
axes[0].set_title("EPMoE Performance (Batch Size ≤ 1024)")
axes[0].set_xlabel("Batch Size")
axes[0].set_ylabel("Latency (ms)")
axes[0].grid(True)
# axes[0].legend()

# 子图2：全量
axes[1].plot(batch_size[:-2], epmoe[:-2], marker="o", color="orange", label="EPMoE (all, w/o last)")
# axes[1].plot(batch_size[:], epmoe[:], marker="o", color="orange", label="EPMoE (all, w/o last)")
axes[1].set_title("EPMoE Performance (All Batch Sizes)")
axes[1].set_xlabel("Batch Size")
axes[1].grid(True)
# axes[1].legend()

plt.tight_layout()
plt.show()

plt.savefig("epmoe.png")
