# export PYTHONPATH=/wangye/1025

import sys
sys.path.insert(0, "/wangye/1025/janus")
import argparse
import torch
import triton
# /home/zhexiangz/prototype/janus/CUHKSZ/disaggregate/afd_moe
# /wangye/1025/janus/CUHKSZ/disaggregate/afd_moe/ep_moe_kernel.py
from CUHKSZ.disaggregate.afd_moe.kernels.ep_moe_kernel import EPMoE


# PYTHONPATH=$PYTHONPATH:/home/zhexiangz/prototype/janus/CUHKSZ/disaggregate python3 /home/zhexiangz/prototype/janus/CUHKSZ/micro_benchmark/moe/bench_ep.py
# /wangye/1025/janus/run_req.py
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
    # 随机均匀选择 expert
    topk_ids = torch.randint(0, 380, (batch_size, topk), device="cuda", dtype=torch.int32)
    
 
    
    print(f"\nLoad balancing metrics:")
    print(topk_ids)
    print("=" * 50)

    # 权重随便给（模型 forward 需要）
    topk_weights = torch.ones(batch_size, topk, device="cuda", dtype=torch.bfloat16)

    # 构建EPMoE
    model = EPMoE(
        num_experts=160,
        top_k=topk,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=dtype,
        # start_expert_id=203,
        # end_expert_id=231,
        start_expert_id=145,
        end_expert_id=173,
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
