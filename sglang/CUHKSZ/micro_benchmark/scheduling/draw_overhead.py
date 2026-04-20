import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})

# Data
batches = [16, 64, 256, 1024]
eps = [8, 12, 16]

baseline = np.array([
    [0.068, 0.068, 0.069],
    [0.068, 0.069, 0.067],
    [0.070, 0.073, 0.077],
    [0.070, 0.071, 0.072],
])

our = np.array([
    [0.069, 0.083, 0.073],
    [0.087, 0.98, 0.094],
    [0.089, 0.095, 0.095],
    [0.088, 0.102, 0.096],
])

# 差值
diff = our - baseline

# 只保留 ep=8 和 ep=16
selected_eps = [8, 16]
selected_idx = [eps.index(e) for e in selected_eps]

fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

x = np.arange(len(batches))  # x位置
width = 0.35

for plot_idx, j in enumerate(selected_idx):
    ep = eps[j]
    bars1 = axes[plot_idx].bar(x - width/2, baseline[:, j], width, label="Baseline")
    bars2 = axes[plot_idx].bar(x + width/2, our[:, j], width, label="DistScale")
    
    # 在our柱子上方标注绝对差值
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        d = diff[i, j]
        axes[plot_idx].text(bar.get_x() + bar.get_width()/2, height, f"+{d:.3f}", 
                            ha='center', va='bottom', fontsize=12, rotation=0)
    
    axes[plot_idx].set_title(f"ep = {ep}", fontsize=18)
    axes[plot_idx].set_xticks(x)
    axes[plot_idx].set_xticklabels(batches)
    axes[plot_idx].set_xlabel("Batch Size", fontsize=16)
    if plot_idx == 0:
        axes[plot_idx].set_ylabel("Latency (ms)", fontsize=16)
    axes[plot_idx].legend(fontsize=12)

plt.suptitle("Overhead Comparison (Baseline vs DistScale)", fontsize=18)
plt.tight_layout()
plt.subplots_adjust(wspace=0.15)
plt.show()
plt.savefig("overhead.png")