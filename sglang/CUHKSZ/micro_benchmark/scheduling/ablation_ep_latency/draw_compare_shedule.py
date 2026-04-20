import matplotlib.pyplot as plt
import numpy as np

# Set global font size
plt.rcParams.update({'font.size': 14})

# ===== 数据 =====
# Layer 1
layer1_latency_dist = [0.591598, 0.688374, 0.640099, 0.732256, 0.683608,
                       0.640811, 0.663648, 0.686214, 0.69452, 0.680307]
layer1_expert_dist = [11, 14, 13, 14, 14, 14, 14, 14, 16, 15]

layer1_latency_rand = [0.836914, 0.772573, 0.822717, 0.90773, 0.806907,
                       0.758315, 0.762413, 0.753254, 0.714834, 0.712554]
layer1_expert_rand = [22, 19, 21, 25, 20, 18, 19, 18, 16, 17]

# Layer 59
layer59_latency_dist = [0.610765, 0.640309, 0.633998, 0.624205, 0.66543,
                        0.638141, 0.625902, 0.639314, 0.720755, 0.704619]
layer59_expert_dist = [12, 12, 13, 13, 14, 14, 14, 14, 18, 17]

layer59_latency_rand = [0.842203, 0.799594, 0.680242, 0.69568, 0.693906,
                        0.853189, 0.82279, 0.871819, 0.892288, 0.874058]
layer59_expert_rand = [22, 20, 15, 16, 16, 21, 21, 22, 23, 24]


# ===== 绘图 =====
x = np.arange(10)  # GPU ID

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# ---------------- Layer 1 ----------------
ax = axes[0]
ax2 = ax.twinx()

# bar
ax.bar(x - 0.2, layer1_expert_dist, width=0.4, color="goldenrod",
       alpha=0.8, label="#Experts (DistScale)")
ax.bar(x + 0.2, layer1_expert_rand, width=0.4, color="skyblue",
       alpha=0.8, label="#Experts (Baseline)")

# line - changed to bright red for better contrast
ax2.plot(x, layer1_latency_dist, "o-", color="crimson", linewidth=2, markersize=6, label="Latency (DistScale)")
ax2.plot(x, layer1_latency_rand, "o-", color="blue", linewidth=2, markersize=6, label="Latency (Baseline)")

ax.set_title("Layer 1", fontsize=18, fontweight='bold')
ax.set_xlabel("GPU ID", fontsize=16)
ax.set_ylabel("Expert Count", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(10)])
ax2.set_ylabel("Latency (ms)", fontsize=16, labelpad=10)
ax2.set_ylim(bottom=0, top=1.1)  # 设置Latency轴从0开始

ax.legend(loc="upper left", fontsize=12)
ax2.legend(loc="upper right", fontsize=12)

# ---------------- Layer 59 ----------------
ax = axes[1]
ax2 = ax.twinx()

# bar
ax.bar(x - 0.2, layer59_expert_dist, width=0.4, color="goldenrod",
       alpha=0.8, label="#Experts (DistScale)")
ax.bar(x + 0.2, layer59_expert_rand, width=0.4, color="skyblue",
       alpha=0.8, label="#Experts (Baseline)")

# line - changed to bright red for better contrast
ax2.plot(x, layer59_latency_dist, "o-", color="crimson", linewidth=2, markersize=6, label="Latency (DistScale)")
ax2.plot(x, layer59_latency_rand, "o-", color="blue", linewidth=2, markersize=6, label="Latency (Baseline)")

ax.set_title("Layer 59", fontsize=18, fontweight='bold')
ax.set_xlabel("GPU ID", fontsize=16)
ax.set_ylabel("Expert Count", fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels([str(i) for i in range(10)])
ax2.set_ylabel("Latency (ms)", fontsize=16, labelpad=10)
ax2.set_ylim(bottom=0, top=1.1)  # 设置Latency轴从0开始

ax.legend(loc="upper left", fontsize=12)
ax2.legend(loc="upper right", fontsize=12)

plt.tight_layout()
plt.subplots_adjust(wspace=0.4)  # 增加子图间距
plt.show()
plt.savefig("ablation_ep_latency.png", dpi=300, bbox_inches='tight')