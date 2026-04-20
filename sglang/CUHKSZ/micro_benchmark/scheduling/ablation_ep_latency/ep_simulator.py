import argparse
import torch
import triton
from moeLayers.ep_moe_kernel import EPMoE
import sys
sys.path.insert(0, "/home/zhexiangz/prototype/janus")

# PYTHONPATH=$PYTHONPATH:/home/zhexiangz/prototype/janus/CUHKSZ/disaggregate python3 /home/zhexiangz/prototype/janus/CUHKSZ/micro_benchmark/scheduling/ablation_ep_latency/test.py
EXPERT_RANGES = [
    (0, 28),
    (29, 57),
    (58, 86),
    (87, 115),
    (116, 144),
    (145, 173),
    (174, 202),
    (203, 231),
    (232, 260),
    (261, 289),
]


EXPERT_RANGE_NAMES = [f"{s}-{e}" for s, e in EXPERT_RANGES]


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-experts", type=int, default=290)
    parser.add_argument("--hidden-size", type=int, default=5120)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    args = parser.parse_args()

    dtype = torch.bfloat16
    # 示例：这里用随机的 topk_id，你可以换成自己的 greedy_topk_id
    # greedy_topk_id = torch.randint(0, args.num_experts, (65, 6), device="cuda", dtype=torch.int32)
    greedy_topk_id = torch.tensor([[  9,  11, 151, 174, 181, 180],
        [149, 175, 172, 206, 268, 154],
        [287,  73,  88, 181, 183, 206],
        [ 24,  38,  26,  93, 149,  70],
        [ 87,  90,  95, 212, 238, 274],
        [  7,  26,  44, 149, 158,   1],
        [ 68,  71, 126, 245, 266,  84],
        [ 13, 218, 237, 262, 275,  30],
        [  9,  11,  89, 213, 234, 239],
        [ 66,  91,  98, 119, 187, 131],
        [ 63, 101, 116, 215, 236,  34],
        [279,   5,  14, 156, 218,  13],
        [  7, 289,  36, 188, 208, 281],
        [  3,  13, 182, 172, 186,  93],
        [  1, 279,  36,  40, 118,  29],
        [ 66,  84, 123, 246, 247, 260],
        [279,   5,  14, 217, 218,  63],
        [283, 178, 203, 245, 260, 256],
        [285,   9, 289, 218, 272, 213],
        [116, 128, 180, 181, 243, 234],
        [100, 102, 124, 128, 174, 241],
        [ 26, 175, 178, 182,  38, 150],
        [  5,  13,  14,  33, 275,  63],
        [277,  24,  38,  44, 144,  29],
        [ 89, 145, 212, 234, 236, 243],
        [ 42,  60,  65,  56, 182,  35],
        [ 24,  26, 123, 152, 160,  58],
        [ 13,  58,  65, 172, 187, 189],
        [ 24,  62, 129, 145, 147, 116],
        [ 73,  90, 212, 238, 241, 217],
        [ 35,  44,  62,  84, 270, 263],
        [277, 279,  13,  40, 156, 287],
        [  5,  13,  14,  84, 156, 279],
        [ 89, 151, 213, 234, 236, 145],
        [ 93, 186, 206, 254, 275,  84],
        [283,  30, 118, 266, 268, 264],
        [283, 203, 268, 256, 258, 182],
        [127, 155, 182, 203, 205, 189],
        [154, 159, 207, 217, 233, 242],
        [ 13,  40, 156, 159, 160, 155],
        [181, 172, 237, 231, 258, 243],
        [  7,  15, 123, 208,   3, 122],
        [ 89, 174, 212, 234, 236, 213],
        [281,   9,  11,  70, 272,  68],
        [ 89, 234, 236, 243, 256, 217],
        [ 33,  35, 180, 182, 211, 186],
        [285, 127, 268, 272, 260, 116],
        [ 68, 145, 174, 181, 204, 147],
        [ 58,  65, 122, 131, 270, 127],
        [176, 217, 239, 265, 270, 152],
        [ 71, 148, 177, 263, 265, 275],
        [  5,  14,  32, 148, 156,  33],
        [ 64, 125, 215, 217, 232, 131],
        [279,  13,  15,  44, 218, 215],
        [149, 186, 208, 231, 258, 264],
        [145, 147, 174, 181, 204,  56],
        [285,  13,  84, 218, 232,   7],
        [279,  13,  40, 211, 237, 218],
        [ 24, 212, 217, 240, 241, 160],
        [277, 101, 179, 187, 210, 189],
        [ 29,  73,  91, 119, 131, 117],
        [ 11, 145, 147, 181, 204, 175],
        [ 40,  28,  94, 186, 207, 183],
        [ 38,  40, 156, 159, 268, 264],
        [ 67,  92,  84, 179, 266,  88]], device='cuda:0', dtype=torch.int32)
    
    
    benchmark.run(
        print_data=True,
        show_plots=False,   
        save_path=None,     
        num_experts=args.num_experts,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        dtype=dtype,
        topk_id=greedy_topk_id,
        rep=20,
    )


if __name__ == "__main__":
    main()
