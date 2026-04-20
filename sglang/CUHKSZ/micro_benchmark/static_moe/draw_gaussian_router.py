import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.gridspec import GridSpec

# 字体设置（与 draw.py 保持一致）
plt.rcParams.update({
    "font.size": 20,        # 全局字体大小
    "axes.titlesize": 24,   # 子图标题字体大小
    "axes.labelsize": 22,   # 坐标轴标签字体大小
    "xtick.labelsize": 20,  # x轴刻度字体大小
    "ytick.labelsize": 20,  # y轴刻度字体大小
    "legend.fontsize": 18   # 图例字体大小
})

# 真实延迟数据（从 result.txt 固化到代码中）
STD_LATENCY_DATA = {
    0.25: (
        [32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 3072.0, 4096.0, 5120.0, 6144.0, 8192.0, 16384.0],
        [0.844096, 0.869280, 0.878400, 0.890608, 0.865568, 0.950976, 1.236864, 1.650112, 2.058048, 2.629936, 3.113280, 4.137600, 8.912896],
    ),
    0.5: (
        [32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 3072.0, 4096.0, 5120.0, 6144.0, 8192.0, 16384.0],
        [0.842976, 0.846624, 0.835520, 0.899456, 0.835520, 0.896560, 1.175184, 1.545920, 1.993024, 2.616608, 3.090112, 4.107904, 8.973568],
    ),
    0.75: (
        [32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0, 3072.0, 4096.0, 5120.0, 6144.0, 8192.0, 16384.0],
        [0.827520, 0.865952, 0.865600, 0.864160, 0.849984, 0.915936, 1.180544, 1.504304, 1.981728, 2.668960, 3.074112, 4.056704, 8.980864],
    ),
}

def generate_gaussian_routing(batch_size, num_experts, topk, gaussian_std=1.0):
    """
    生成基于标准化正态分布的expert routing，通过调整标准差改变分布宽度
    
    Args:
        batch_size: batch大小
        num_experts: expert总数
        topk: top-k值
        gaussian_std: 标准化正态分布的标准差，控制分布宽度
    
    Returns:
        topk_ids: expert分配结果
        expert_usage: 每个expert的使用次数
    """
    if batch_size > num_experts:
        # 首先确保每个expert至少有一个token
        topk_ids = torch.zeros(batch_size, topk, dtype=torch.int32)
        
        # 前num_experts个token分别分配给不同的expert
        for i in range(num_experts):
            topk_ids[i, 0] = i
        
        # 剩余的token按照标准化正态分布分配
        remaining_tokens = batch_size - num_experts
        if remaining_tokens > 0:
            # 使用标准化正态分布 (mean=0, std=1)，然后乘以gaussian_std调整宽度
            mean = 0.0
            base_std = 1.0
            

            
            # 为剩余token生成标准化正态分布的expert分配
            for i in range(remaining_tokens):
                token_idx = num_experts + i
                # 生成标准化正态分布的随机数，然后乘以gaussian_std
                z_score = torch.normal(mean=mean, std=base_std, size=(1,)) * gaussian_std
                
                # 使用CDF（Φ函数）将z-score映射到[0, 1]区间
                # Φ(z) = 0.5 * (1 + erf(z/sqrt(2)))
                cdf_value = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0))))
                
                # 缩放到expert ID范围
                expert_id = (cdf_value * (num_experts - 1)).to(torch.int32)
                topk_ids[token_idx, 0] = expert_id
    else:
        # 如果batch <= expert总数，直接按顺序分配
        topk_ids = torch.arange(batch_size).unsqueeze(1).expand(-1, topk).to(torch.int32)
    
    # 统计每个expert被调用的次数
    expert_usage = torch.zeros(num_experts, dtype=torch.int32)
    for i in range(batch_size):
        for j in range(topk):
            expert_id = topk_ids[i, j]
            expert_usage[expert_id] += 1
    
    return topk_ids, expert_usage

def analyze_routing_distribution(expert_usage, num_experts, batch_size, topk):
    """
    分析routing分布情况
    
    Args:
        expert_usage: 每个expert的使用次数
        num_experts: expert总数
        batch_size: batch大小
        topk: top-k值
    
    Returns:
        dict: 包含各种统计指标的字典
    """
    min_usage = expert_usage.min().item()
    max_usage = expert_usage.max().item()
    mean_usage = expert_usage.float().mean().item()
    std_usage = expert_usage.float().std().item()
    
    # 计算负载均衡指标
    max_min_ratio = max_usage / min_usage if min_usage > 0 else float('inf')
    
    # 计算覆盖率 (有多少expert被使用)
    coverage = (expert_usage > 0).sum().item() / num_experts
    
    # 计算分布的不均匀性 (Gini系数)
    sorted_usage = torch.sort(expert_usage, descending=True)[0].float()
    cumsum = torch.cumsum(sorted_usage, dim=0)
    n = num_experts
    gini = (n + 1 - 2 * torch.sum(cumsum) / torch.sum(sorted_usage)) / n
    
    return {
        'min_usage': min_usage,
        'max_usage': max_usage,
        'mean_usage': mean_usage,
        'std_usage': std_usage,
        'max_min_ratio': max_min_ratio,
        'coverage': coverage,
        'gini_coefficient': gini.item()
    }

def plot_routing_results(expert_usage, num_experts, batch_size, topk, gaussian_std, save_path=None):
    """
    绘制routing结果图表 - 只显示expert usage distribution
    
    Args:
        expert_usage: 每个expert的使用次数
        num_experts: expert总数
        batch_size: batch大小
        topk: top-k值
        gaussian_std: 高斯分布标准差
        save_path: 保存路径
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle(f'Expert Usage Distribution (batch={batch_size}, experts={num_experts}, topk={topk}, std={gaussian_std})', 
                 fontsize=16)
    
    # Expert Usage Bar Chart
    expert_ids = list(range(num_experts))
    bars = ax.bar(expert_ids, expert_usage.numpy(), alpha=0.7, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Expert ID')
    ax.set_ylabel('Number of Tokens')
    ax.set_title('Expert Usage Distribution')
    ax.grid(True, alpha=0.3)
    
    # 添加平均值线
    mean_usage = expert_usage.float().mean().item()
    ax.axhline(y=mean_usage, color='red', linestyle='--', label=f'Mean: {mean_usage:.2f}')
    ax.legend()
    
    # 添加统计信息
    stats = analyze_routing_distribution(expert_usage, num_experts, batch_size, topk)
    stats_text = f'Max/Min: {stats["max_min_ratio"]:.1f}\nCoverage: {stats["coverage"]:.1%}\nGini: {stats["gini_coefficient"]:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def load_latency_from_result(result_path):
    """
    解析 result.txt，提取三种 std 下的 (x, y) 延迟曲线。
    期望格式类似：
        std = 0.25
        epmoe-performance:
        15        32.0  0.844096
        ...
    返回: dict，例如 {0.25: (xs, ys), 0.5: (xs, ys), 0.75: (xs, ys)}
    """
    std_to_xy = {}
    if not os.path.exists(result_path):
        return std_to_xy

    with open(result_path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    current_std = None
    collecting = False
    for line in lines:
        if line.startswith('std ='):
            try:
                current_std = float(line.split('=')[1].strip())
                std_to_xy[current_std] = ([], [])
            except Exception:
                current_std = None
            # 直接开始收集该 std 下的行（有些段落可能缺少 "epmoe-performance:" 标头）
            collecting = True
            continue
        if line.startswith('epmoe-performance'):
            # 标头可选，遇到则继续
            collecting = True
            continue
        if collecting and line:
            # 允许三列: idx, x, y  或 两列: x, y
            parts = line.split()
            # 如果遇到非数字行则跳过
            try:
                if len(parts) >= 3:
                    # 忽略第一列索引
                    x_val = float(parts[-2])
                    y_val = float(parts[-1])
                elif len(parts) == 2:
                    x_val = float(parts[0])
                    y_val = float(parts[1])
                else:
                    continue
                if current_std is not None:
                    std_to_xy[current_std][0].append(x_val)
                    std_to_xy[current_std][1].append(y_val)
            except ValueError:
                # 非数据行或下一段的干扰行，将其忽略
                continue
    return std_to_xy


def plot_three_stds_comparison(batch_size, num_experts, topk, save_path=None):
    """
    在一个大图中显示3个不同标准差的标准化正态分布expert usage distribution
    
    Args:
        batch_size: batch大小
        num_experts: expert总数
        topk: top-k值
        save_path: 保存路径
    """
    # 选择3个不同的标准差值
    std_values = [0.25, 0.5, 0.75]  # 小、中、大标准差
    
    # 预先计算三个标准差对应的usage，以统一坐标轴
    usages = []
    stats_list = []
    for std in std_values:
        _, expert_usage = generate_gaussian_routing(batch_size, num_experts, topk, gaussian_std=std)
        usages.append(expert_usage)
        stats_list.append(analyze_routing_distribution(expert_usage, num_experts, batch_size, topk))
    
    global_max = max(int(u.max().item()) for u in usages)
    # 向上取整到更美观的范围
    if global_max <= 10:
        y_max = 10
    else:
        # 四舍五入到最近的10%
        magnitude = 10 ** (len(str(global_max)) - 1)
        y_max = ((global_max + magnitude - 1) // magnitude) * magnitude
    
    # 使用GridSpec：上面三张子图共享y轴，下面一张大图展示三种分布下的“真实延迟曲线”
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, height_ratios=[3, 1.6], hspace=0.55, wspace=0.25)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_latency = fig.add_subplot(gs[1, :])
    
    fig.suptitle(f'Expert activation distribution under different std\n(batch={batch_size}, experts={num_experts})', fontsize=28)
    # 增大总标题与子图之间的距离
    fig.subplots_adjust(top=0.84)
    
    for idx, std in enumerate(std_values):
        expert_usage = usages[idx]
        stats = stats_list[idx]
        
        ax = axes[idx]
        expert_ids = list(range(num_experts))
        bars = ax.bar(expert_ids, expert_usage.numpy(), alpha=0.7, color=f'C{idx}', edgecolor='navy')
        ax.set_xlabel('Expert ID')
        if idx == 0:
            ax.set_ylabel('Number of Tokens')
        ax.set_title(f'std={std}')
        ax.grid(True, alpha=0.3)
        
        # 统一坐标轴
        ax.set_xlim([-0.5, num_experts - 0.5])
        ax.set_ylim([0, y_max])
    
    # 使用内置 STD_LATENCY_DATA 绘制底部真实 latency 曲线
    for idx, std in enumerate(std_values):
        if std in STD_LATENCY_DATA:
            xs, ys = STD_LATENCY_DATA[std]
            xs_np = np.array(xs)
            ys_np = np.array(ys)
            mask = (xs_np >= 32) & (xs_np <= 2048)
            ax_latency.plot(xs_np[mask], ys_np[mask], marker='o', linewidth=2, label=f'std={std}')
    ax_latency.set_xscale('log', base=2)
    ax_latency.set_xlim(32, 2048)
    ax_latency.set_xticks([32, 64, 128, 256, 512, 1024, 2048])
    ax_latency.set_xticklabels(["32", "64", "128", "256", "512", "1024", "2048"]) 
    ax_latency.set_xlabel('Batch Size')
    ax_latency.set_ylabel('Latency (ms)')
    ax_latency.set_title('Latency under different stds')
    ax_latency.grid(True, alpha=0.3)
    ax_latency.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Standardized Normal Distribution Router Analysis Tool')
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--num-experts", type=int, default=32, help="Number of experts")
    parser.add_argument("--topk", type=int, default=1, help="Top-k value")
    parser.add_argument("--gaussian-std", type=float, default=None)
    parser.add_argument("--single-plot", action="store_true")
    parser.add_argument("--save-path", type=str, default="/home/zhexiangz/prototype/janus/CUHKSZ/benchmark/moe")
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)
    
    if args.single_plot and args.gaussian_std is not None:
        # 分析单个标准差
        topk_ids, expert_usage = generate_gaussian_routing(args.batch_size, args.num_experts, args.topk, args.gaussian_std)
        
        # 打印统计信息

        
        stats = analyze_routing_distribution(expert_usage, args.num_experts, args.batch_size, args.topk)
        
        # 绘制图表
        plot_save_path = os.path.join(args.save_path, 
                                    f"standardized_normal_analysis_b{args.batch_size}_e{args.num_experts}_std{args.gaussian_std}.png")
        plot_routing_results(expert_usage, args.num_experts, args.batch_size, args.topk, args.gaussian_std, plot_save_path)
    else:
        # 默认显示三个标准差的比较

        
        # 绘制三个子图的比较
        plot_save_path = os.path.join(args.save_path, 
                                    f"standardized_normal_comparison_b{args.batch_size}_e{args.num_experts}.png")
        plot_three_stds_comparison(args.batch_size, args.num_experts, args.topk, plot_save_path)

if __name__ == "__main__":
    main() 