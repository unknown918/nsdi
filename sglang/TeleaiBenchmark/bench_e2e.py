"""    
python3 sglang_latency_test.py \
    --model-path /ai_sds_yumc/models/Qwen1.5-MoE-A2.7B \
    --batch-size 1 8 16 \
    --input-len 128 512 \
    --output-len 16 32

# 使用数据并行
python3 sglang_latency_test.py \
    --model-path /ai_sds_yumc/models/Qwen1.5-MoE-A2.7B \
    --dp-size 2 \
    --batch-size 1 8 16

# 使用张量并行
python3 sglang_latency_test.py \
    --model-path /ai_sds_yumc/models/Qwen1.5-MoE-A2.7B \
    --tp-size 4 \
    --batch-size 1 8 16

# 使用专家并行
python3 sglang_latency_test.py \
    --model-path /ai_sds_yumc/models/Qwen1.5-MoE-A2.7B \
    --ep-size 4 \
    --enable-ep-moe \
    --batch-size 1 8 16

# 使用 DP Attention
python3 sglang_latency_test.py \
    --model-path /ai_sds_yumc/models/DeepSeek-V2 \
    --tp-size 4 \
    --enable-dp-attention \
    --batch-size 1 8 16
"""

import sys

import os
import time
import json
import argparse
import itertools
import dataclasses
from typing import List, Tuple

import torch
import sglang as sgl
# import 

# sys.path.append("/zhangzhexiang")  

# 设置环境变量
os.environ["NUM_HIDDEN_LAYERS"] = "2"
os.environ["NUM_EXPERTS"] = "64"
# os.environ["NUM_EXPERTS_PER_TOK"] = "6"
# os.environ["N_GROUP"] = "1" # Only for DeepSeek-V3
# os.environ["TOPK_GROUP"] = "1" # Only for DeepSeek-V3

@dataclasses.dataclass
class BenchArgs:
    """基准测试参数配置"""
    model_path: str = "/ai_sds_yumc/models/Qwen1.5-MoE-A2.7B"
    run_name: str = "default"
    batch_size: Tuple[int] = (1,)
    input_len: Tuple[int] = (512,)
    output_len: Tuple[int] = (16,)
    result_filename: str = "result.jsonl"
    dp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1
    enable_ep_moe: bool = False
    enable_dp_attention: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--model-path", type=str, default=BenchArgs.model_path)
        parser.add_argument("--run-name", type=str, default=BenchArgs.run_name)
        parser.add_argument("--batch-size", type=int, nargs="+", default=BenchArgs.batch_size)
        parser.add_argument("--input-len", type=int, nargs="+", default=BenchArgs.input_len)
        parser.add_argument("--output-len", type=int, nargs="+", default=BenchArgs.output_len)
        parser.add_argument("--result-filename", type=str, default=BenchArgs.result_filename)
        parser.add_argument("--dp-size", type=int, default=BenchArgs.dp_size)
        parser.add_argument("--tp-size", type=int, default=BenchArgs.tp_size)
        parser.add_argument("--ep-size", type=int, default=BenchArgs.ep_size)
        parser.add_argument("--enable-ep-moe", action="store_true")
        parser.add_argument("--enable-dp-attention", action="store_true")

    @classmethod
    def from_cli_args(cls, args):
        dp_size = args.dp_size
        # 如果启用了dp_attention，强制dp_size等于tp_size
        if args.enable_dp_attention:
            dp_size = args.tp_size
            print(f"DP Attention enabled, set dp_size to tp_size: {dp_size}")
            
        return cls(
            model_path=args.model_path,
            run_name=args.run_name,
            batch_size=args.batch_size,
            input_len=args.input_len,
            output_len=args.output_len,
            result_filename=args.result_filename,
            dp_size=dp_size,
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            enable_ep_moe=args.enable_ep_moe,
            enable_dp_attention=args.enable_dp_attention,
        )


def prepare_synthetic_inputs(batch_size: int, input_len: int) -> List[str]:
    """Create synthetic input data"""
    return ["A" * input_len] * batch_size


def run_latency_test_once(
    engine: sgl.Engine,
    batch_size: int,
    input_len: int,
    output_len: int,
    model_path: str,
    run_name: str,
):
    """measure the latency of the generation"""
    
    # prepare synthetic input
    prompts = prepare_synthetic_inputs(batch_size, input_len)
    
    # initialize the result dictionary
    measurement_results = {
        "run_name": run_name,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "model": model_path,
    }
    
    # 1. measure the prefill latency
    torch.cuda.synchronize()
    prefill_start = time.time()
        
    _ = engine.generate(prompts, {"max_new_tokens": 0})
    
    torch.cuda.synchronize()
    prefill_latency = time.time() - prefill_start
    prefill_throughput = input_len * batch_size / prefill_latency
    
    print(f"Prefill latency: {prefill_latency:6.5f} s, throughput: {prefill_throughput:9.2f} token/s")
    
    measurement_results["prefill_latency"] = prefill_latency
    measurement_results["prefill_throughput"] = prefill_throughput
    
    # 2. measure the total latency 
    torch.cuda.synchronize()
    total_start = time.time()
    
    outputs = engine.generate(prompts, {"max_new_tokens": output_len})
    
    torch.cuda.synchronize()
    total_latency = time.time() - total_start
    total_tokens = (input_len + output_len) * batch_size
    throughput = total_tokens / total_latency
    
    print(f"Total latency: {total_latency:6.3f} s, throughput: {throughput:9.2f} token/s")
    
    measurement_results["total_latency"] = total_latency
    measurement_results["overall_throughput"] = throughput
    
    # 3. calculate the latency and throughput of the decode stage
    decode_latency = total_latency - prefill_latency
    # the decode latency per token
    decode_per_token_latency = decode_latency / output_len
    # the throughput of the decode stage
    decode_throughput = (output_len * batch_size) / decode_latency
    
    print(f"Decode latency: {decode_latency:6.5f} s, per token: {decode_per_token_latency:6.5f} s")
    print(f"Decode throughput: {decode_throughput:9.2f} token/s")
    
    # measurement_results["decode_latency"] = decode_latency
    measurement_results["decode_per_token_latency"] = decode_per_token_latency
    measurement_results["decode_throughput"] = decode_throughput
    
    return measurement_results


def main():
    parser = argparse.ArgumentParser(description="SgLang latency test tool")
    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    bench_args = BenchArgs.from_cli_args(args)
    
    print("=========================== initialize the engine ============================")
    print(f"parallel config: DP={bench_args.dp_size}, TP={bench_args.tp_size}, EP={bench_args.ep_size}")
        
    llm = sgl.Engine(
        model_path=bench_args.model_path,
        disable_cuda_graph=True,
        dp_size=bench_args.dp_size,
        tp_size=bench_args.tp_size,
        ep_size=bench_args.ep_size,
        enable_p2p_check=True,
        disable_radix_cache=True,
        enable_ep_moe=bench_args.enable_ep_moe,
        enable_dp_attention=bench_args.enable_dp_attention
    )
        
    print("=========================== warmup ============================")
    warmup_batch_size = bench_args.batch_size[0]
    warmup_input_len = bench_args.input_len[0]
    warmup_prompts = prepare_synthetic_inputs(warmup_batch_size, warmup_input_len)
    llm.generate(warmup_prompts, {"max_new_tokens": 8})
        
    print("=========================== benchmark ============================")
    result_list = []
    
    # 遍历所有参数组合
    for bs, il, ol in itertools.product(
        bench_args.batch_size, bench_args.input_len, bench_args.output_len
    ):
        print(f"\n测试配置: batch_size={bs}, input_len={il}, output_len={ol}")
        try:
            result = run_latency_test_once(
                engine=llm,
                batch_size=bs,
                input_len=il,
                output_len=ol,
                model_path=bench_args.model_path,
                run_name=bench_args.run_name
            )
            result_list.append(result)
        except Exception as e:
            print(f"Error: {e}")
    
    # 保存结果
    if bench_args.result_filename:
        with open(bench_args.result_filename, "a") as fout:
            for result in result_list:
                fout.write(json.dumps(result) + "\n")
    
    print("=========================== 测试完成 ============================")

if __name__ == "__main__":
    main()
