import argparse
import torch
import triton
from moeLayers.ep_moe_kernel import EPMoE
import sys
sys.path.insert(0, "/home/zhexiangz/prototype/janus")

from config.ep_placement import *

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dummy"],               # x轴占位
        x_vals=[0],                      # 固定一个点
        line_arg="expert_range",         # 每条线是一个区间
        line_vals=EXPERT_RANGES,
        line_names=EXPERT_RANGE_NAMES,
        ylabel="Time (ms)",
        plot_name="epmoe-latency",                # 不画图
        args={},
    )
)
def benchmark(dummy, expert_range, num_experts, hidden_size,
              intermediate_size, dtype, topk_id, rep=20):

    start_expert_id, end_expert_id = expert_range
    print(f"\nBenchmarking experts {start_expert_id}-{end_expert_id}")

    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)

    batch = topk_id.shape[0]
    hidden_states = torch.randn(batch, hidden_size, dtype=dtype, device="cuda")
    topk_weights = torch.ones(batch, topk_id.shape[1], device="cuda", dtype=dtype)

    model = EPMoE(
        num_experts=num_experts,
        top_k=topk_id.shape[1],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=dtype,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
    ).cuda()

    # warmup
    for _ in range(5):
        _ = model(hidden_states, topk_id, topk_weights)
    torch.cuda.synchronize()

    # timing
    iters_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(rep):
        start_event.record()
        _ = model(hidden_states, topk_id, topk_weights)
        end_event.record()
        torch.cuda.synchronize()
        iters_ms.append(start_event.elapsed_time(end_event))

    return sum(iters_ms) / len(iters_ms), min(iters_ms), max(iters_ms)


def run_simulator(topk_id, num_experts=290, hidden_size=5120, intermediate_size=1536,
                  dtype=torch.bfloat16, rep=20, print_data=True, show_plots=False):
    return benchmark.run(
        print_data=print_data,
        show_plots=show_plots,
        save_path=None,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        topk_id=topk_id,
        rep=rep,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=290) # 29 * gpu 个数
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    args = parser.parse_args()

    dtype = torch.bfloat16
    greedy_topk_id = torch.randint(0, args.num_experts, (65, 6), device="cuda", dtype=torch.int32)

    run_simulator(
        topk_id=greedy_topk_id,
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        dtype=dtype,
        rep=20,
        print_data=True,
        show_plots=False,
    )


if __name__ == "__main__":
    main()
