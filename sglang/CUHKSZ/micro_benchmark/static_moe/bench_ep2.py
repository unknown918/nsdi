import argparse
import torch
import triton
from moeLayers.ep_moe_kernel import EPMoE
import sys

# 添加 janus 路径 （如需）
sys.path.insert(0, "/home/zhexiangz/prototype/janus")

# ===============================================================
# perf_report：激活 1 到 32 个 expert
# ===============================================================
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["active_experts"],
        x_vals=list(range(1, 33)),    # 激活 1~32 个专家
        line_arg="provider",
        line_vals=["epmoe"],
        line_names=["EPMoE"],
        styles=[("red", "-")],
        ylabel="Time (ms)",
        plot_name="epmoe-performance",
        args={},
    )
)
def benchmark(active_experts, provider, num_experts, hidden_size,
              intermediate_size, topk, dtype):

    torch.set_default_device("cuda")
    batch_size = 64

    print(f"\n========== Benchmark: active_experts = {active_experts} ==========")

    # 输入 hidden states
    hidden_states = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")

    # 限制激活数
    active_experts = min(active_experts, num_experts)

    # 随机选 expert
    chosen_experts = torch.randperm(num_experts, device="cuda")[:active_experts]

    # token 映射
    expert_index = torch.randint(0, active_experts, (batch_size, topk), device="cuda")
    topk_ids = chosen_experts[expert_index].to(torch.int32)

    # ===============================================================
    # 打印每个 expert 收到的 token 数量
    # ===============================================================
    expert_usage = torch.zeros(num_experts, device="cuda", dtype=torch.int32)
    expert_usage.index_add_(0, topk_ids.view(-1), torch.ones_like(topk_ids.view(-1)))

    print(f"Chosen experts: {chosen_experts.tolist()}")
    print("Expert token usage:", expert_usage.tolist())

    # 权重
    topk_weights = torch.ones(batch_size, topk, device="cuda", dtype=torch.bfloat16)

    # ===============================================================
    # 构建 EPMoE 模型
    # ===============================================================
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
    for _ in range(20):
        _ = model(hidden_states, topk_ids, topk_weights)
    torch.cuda.synchronize()

    # benchmark
    bench_lambda = lambda: model(hidden_states, topk_ids, topk_weights)
    ms, min_ms, max_ms = triton.testing.do_bench(
        bench_lambda, quantiles=[0.5, 0.2, 0.8]
    )

    print(f"Latency = {ms:.4f} ms\n")
    return ms, min_ms, max_ms


# ===============================================================
# main()
# ===============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--save-path", type=str, default="/home/zhexiangz/ep_kernel_bench")
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
    )


if __name__ == "__main__":
    main()
