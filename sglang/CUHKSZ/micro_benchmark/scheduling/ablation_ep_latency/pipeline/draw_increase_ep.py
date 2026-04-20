import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

# -------- Layer 1 数据 --------
data_layer1 = [
    [0.9055, 0.9485, 0.9175, 0.9145, 0.925, 0.904],
    [0.72, 0.769, 0.766, 0.766, 0.7745, 0.783, 0.802, 0.794],
    [0.6115, 0.6725, 0.742, 0.6935, 0.7255, 0.689, 0.665, 0.6545, 0.7465, 0.762],
    [0.491, 0.5425, 0.606, 0.6095, 0.6105, 0.623, 0.633, 0.7155, 0.7055, 0.6795, 0.69, 0.6815],
    [0.47, 0.5115, 0.4875, 0.468, 0.554, 0.6125, 0.654, 0.676, 0.608, 0.628, 0.626, 0.721, 0.6245, 0.6655],
    [0.471, 0.494, 0.4835, 0.527, 0.524, 0.5435, 0.611, 0.677, 0.582, 0.5215, 0.5955, 0.5215, 0.5925, 0.558, 0.5985, 0.7155]
]

mean_layer1 = [np.mean(row) for row in data_layer1]
max_layer1 = [max(row) for row in data_layer1]

# -------- Layer 59 数据 --------
mean_layer59 = [0.9287, 0.7877, 0.6914, 0.6357, 0.5867, 0.5528]
max_layer59 = [0.9655, 0.827, 0.7755, 0.721, 0.693, 0.693]

# -------- 横坐标 --------
x_values = [6, 8, 10, 12, 14, 16]

# -------- 绘制两个子图 --------
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Layer 1
axes[0].plot(x_values[:len(mean_layer1)], mean_layer1, marker='o', label="Average latency")
axes[0].plot(x_values[:len(max_layer1)], max_layer1, marker='s', label="Max latency")
axes[0].set_title("Layer 1", fontsize=18, fontweight='bold')
axes[0].set_xlabel("Expert parallelism degree", fontsize=16)
axes[0].set_ylabel("Latency", fontsize=16)
axes[0].legend(fontsize=12)
axes[0].grid(True)

# Layer 59
axes[1].plot(x_values, mean_layer59, marker='o', label="Average latency")
axes[1].plot(x_values, max_layer59, marker='s', label="Max latency")
axes[1].set_title("Layer 59", fontsize=18, fontweight='bold')
axes[1].set_xlabel("Expert parallelism degree", fontsize=16)
axes[1].legend(fontsize=12)
axes[1].grid(True)

plt.tight_layout()
plt.subplots_adjust(wspace=0.1)
plt.show()
plt.savefig("/home/zhexiangz/prototype/janus/CUHKSZ/micro_benchmark/scheduling/ablation_ep_latency/pipeline/increase_ep.png", dpi=300, bbox_inches='tight')