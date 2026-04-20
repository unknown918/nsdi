import argparse
import torch
import triton
# /home/zhexiangz/prototype/janus/CUHKSZ/disaggregate/afd_moe
# from afd_moe.ep_moe_kernel import EPMoE
import sys
import os
import json
##sys.path.insert(0, "/data/250010042/two_stages/janus")
from moeLayers.ep_moe_kernel import EPMoE
#sys.path.insert(0, "/home/zhexiangz/prototype/janus")
# PYTHONPATH=$PYTHONPATH:/home/zhexiangz/prototype/janus/CUHKSZ/disaggregate python3 /home/zhexiangz/prototype/janus/CUHKSZ/micro_benchmark/moe/bench_ep.py

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
                64, 128, 256, 512, 1024, 2048 ,3072, 4096, 5120, 6144, 8192, 16384],
        line_arg="provider",
        line_vals=["epmoe"],
        line_names=["EPMoE"],
        styles=[("red", "-")],
        ylabel="Time (ms)",
        plot_name="epmoe-performance",
        args={},
    )
)
def benchmark(batch_size, provider, num_experts, hidden_size,
              intermediate_size, topk, dtype, router_mode, gaussian_std=None):
    print(f"benchmark {provider} with batch_size={batch_size}, router_mode={router_mode}")
    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)

    # 输入
    hidden_states = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")

    # ===============================
    # Router: 生成 topk_ids
    # ===============================
    if router_mode == "random":
        # 随机均匀选择 expert
        topk_ids = torch.randint(0, num_experts, (batch_size, topk),
                                 device="cuda", dtype=torch.int32)
    elif router_mode == "roundrobin":
        # 轮询均匀分配 expert
        indices = torch.arange(batch_size * topk, device="cuda") % num_experts
        topk_ids = indices.view(batch_size, topk).to(torch.int32)
    elif router_mode == "gaussian":
        # 如果batch > expert总数，先保证每个expert分配一个token，然后剩余token按标准化正态分布分配
        if batch_size > num_experts:
            # 首先确保每个expert至少有一个token
            topk_ids = torch.zeros(batch_size, topk, device="cuda", dtype=torch.int32)
            
            # 前num_experts个token分别分配给不同的expert
            for i in range(num_experts):
                topk_ids[i, 0] = i
            
            # 剩余的token按照标准化正态分布分配
            remaining_tokens = batch_size - num_experts
            if remaining_tokens > 0:
                # 使用标准化正态分布 (mean=0, std=1)，然后乘以gaussian_std调整宽度
                mean = 0.0
                base_std = 1.0
                if gaussian_std is None:
                    std = 1.0  # 默认使用标准正态分布
                else:
                    std = gaussian_std
                
                # 为剩余token生成标准化正态分布的expert分配
                for i in range(remaining_tokens):
                    token_idx = num_experts + i
                    # 生成标准化正态分布的随机数，然后乘以gaussian_std
                    z_score = torch.normal(mean=mean, std=base_std, size=(1,), device="cuda") * std
                    
                    # 使用CDF（Φ函数）将z-score映射到[0, 1]区间
                    # Φ(z) = 0.5 * (1 + erf(z/sqrt(2)))
                    cdf_value = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0, device="cuda"))))
                    
                    # 缩放到expert ID范围
                    expert_id = (cdf_value * (num_experts - 1)).to(torch.int32)
                    topk_ids[token_idx, 0] = expert_id
        else:
            # 如果batch <= expert总数，直接按顺序分配
            topk_ids = torch.arange(batch_size, device="cuda").unsqueeze(1).expand(-1, topk).to(torch.int32)
    else:
        raise ValueError(f"Unsupported router_mode={router_mode}")

    # 统计每个expert被调用的次数
    expert_usage = torch.zeros(num_experts, device="cuda", dtype=torch.int32)
    for i in range(batch_size):
        for j in range(topk):
            expert_id = topk_ids[i, j]
            expert_usage[expert_id] += 1
    
    # 打印routing统计信息
    print(f"\n=== Router Mode: {router_mode} ===")
    print(f"Batch size: {batch_size}, Top-k: {topk}, Num experts: {num_experts}")
    print("Expert usage statistics:")
    for i in range(num_experts):
        print(f"  Expert {i:2d}: {expert_usage[i].item():3d} tokens")
    
    # 计算负载均衡指标
    min_usage = expert_usage.min().item()
    max_usage = expert_usage.max().item()
    mean_usage = expert_usage.float().mean().item()
    std_usage = expert_usage.float().std().item()
    
    print(f"\nLoad balancing metrics:")
    print(f"  Min usage: {min_usage}")
    print(f"  Max usage: {max_usage}")
    print(f"  Mean usage: {mean_usage:.2f}")
    print(f"  Std usage: {std_usage:.2f}")
    print(f"  Max/Min ratio: {max_usage/min_usage:.2f}" if min_usage > 0 else "  Max/Min ratio: N/A")
    print("=" * 50)

    # 权重随便给（模型 forward 需要）
    topk_weights = torch.ones(batch_size, topk, device="cuda", dtype=torch.bfloat16)

    # 构建EPMoE
    model = EPMoE(
        num_experts=num_experts,
        top_k=topk,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=dtype,
        start_expert_id=0,
        end_expert_id=num_experts - 1,
    ).cuda()

    # warmup
    for _ in range(10):
        _ = model(hidden_states, topk_ids, topk_weights)
    torch.cuda.synchronize()

    # benchmark
    bench_lambda = lambda: model(hidden_states, topk_ids, topk_weights)
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(bench_lambda, quantiles=quantiles)
    return ms, min_ms, max_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--save-path", type=str, default="/home/zhexiangz/ep_kernel_bench")
    parser.add_argument("--router-mode", type=str, choices=["random", "roundrobin", "gaussian"], default="roundrobin")
    parser.add_argument("--gaussian-std", type=float, default=0.75)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "float16" else torch.bfloat16

    benchmark.run(
        show_plots=True,
        print_data=True,
        save_path=args.save_path,
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        topk=args.topk,
        dtype=dtype,
        router_mode=args.router_mode,
        gaussian_std=args.gaussian_std,
    )


if __name__ == "__main__":
    main()
