import sys
import re
import os
import time
import json
import torch
import argparse
import dataclasses
from typing import List, Tuple

import sglang as sgl

# sys.path.append("/zhangzhexiang")  

# 设置环境变量
# os.environ["NUM_HIDDEN_LAYERS"] = "2"
# os.environ["NUM_EXPERTS"] = "64"

@dataclasses.dataclass
class BenchArgs:    
    model_path: str = "/ai_sds_yumc/models/Qwen1.5-MoE-A2.7B"
    run_name: str = "default"
    batch_size: int = 1
    input_len: int = 256
    output_len: int = 16
    result_filename: str = "result.jsonl"
    dp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1
    enable_ep_moe: bool = False
    enable_dp_attention: bool = False
    profile: bool = False
    profile_filename_prefix: str = "profile"
    dataset_path: str = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument("--model-path", type=str, default=BenchArgs.model_path)
        parser.add_argument("--batch-size", type=int, default=BenchArgs.batch_size)
        parser.add_argument("--input-len", type=int, default=BenchArgs.input_len)
        parser.add_argument("--output-len", type=int, default=BenchArgs.output_len)
        parser.add_argument("--result-filename", type=str, default=BenchArgs.result_filename)
        parser.add_argument("--dataset-path", type=str, default=None)
        parser.add_argument("--dp-size", type=int, default=BenchArgs.dp_size)
        parser.add_argument("--tp-size", type=int, default=BenchArgs.tp_size)
        parser.add_argument("--ep-size", type=int, default=BenchArgs.ep_size)
        parser.add_argument("--enable-ep-moe", action="store_true")
        parser.add_argument("--enable-dp-attention", action="store_true")
        parser.add_argument("--profile", action="store_true")

    @classmethod
    def from_cli_args(cls, args):
        dp_size = args.dp_size
        # 如果启用了dp_attention，强制dp_size等于tp_size
        if args.enable_dp_attention:
            dp_size = args.tp_size
            args.dp_size = dp_size
            print(f"DP Attention enabled, set dp_size to tp_size: {dp_size}")
            
        return cls(
            model_path=args.model_path,
            batch_size=args.batch_size,
            input_len=args.input_len,
            output_len=args.output_len,
            result_filename=args.result_filename,
            dp_size=dp_size,
            tp_size=args.tp_size,
            ep_size=args.ep_size,
            enable_ep_moe=args.enable_ep_moe,
            enable_dp_attention=args.enable_dp_attention,
            profile=args.profile,
            dataset_path=args.dataset_path,
        )


def prepare_synthetic_inputs(batch_size: int, input_len: int) -> List[str]:
    """Create synthetic input data"""
    # TypeError: can't multiply sequence by non-int of type 'list'
    # return ["A" * input_len] * batch_size
    return ["A" * input_len for _ in range(batch_size)]

def monitor_trace_file():       
    PROFIL_DONE_FILE = os.getenv("PROFILE_DONE_FILE")
    # 打开文件,读取,直到读取到1
    with open(PROFIL_DONE_FILE, "r") as f:
        while f.read() != "1":
            print("waiting for profile done")
            time.sleep(2)
    print("monitor_trace_file done")
    time.sleep(10) # 防止还有其他文件

def run_latency_test_once(
    engine: sgl.Engine,
    batch_size: int,
    input_len: int,
    output_len: int,
    profile: bool = False,
    tp_size: int = 1,
    ep_size: int = 1,
    dp_size: int = 1,
    enable_dp_attention: bool = False,
    dataset_path: str = None,
):
    """measure the latency of the generation"""
    
    if dataset_path:        
        all_prompts = load_prompts_from_dataset(dataset_path, num_samples=batch_size)       
        if len(all_prompts) < batch_size:
            all_prompts = all_prompts * (batch_size // len(all_prompts) + 1)
        prompts = all_prompts[:batch_size]
        print(f"Use real dataset: {dataset_path}, loaded {len(all_prompts)} samples")
    else:
        # 使用合成数据
        prompts = prepare_synthetic_inputs(batch_size, input_len)
        print(f"Use synthetic data")
    
    if enable_dp_attention:
        dp_size = tp_size

    # initialize the result dictionary
    measurement_results = {
        "tp_size": tp_size,
        "ep_size": ep_size,
        "dp_size": dp_size,
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "use_real_data": dataset_path is not None, 
    }
    
    # 使用 SGLang 内置的 profiler 功能
    if profile:        
        if not os.getenv("SGLANG_TORCH_PROFILER_DIR"):
            raise ValueError("SGLANG_TORCH_PROFILER_DIR is not set, profiler may not work")                    
        try:
            engine.start_profile()
            print("Successfully start SGLang profiler")
        except Exception as e:
            raise ValueError(f"Start profiler error: {e}")    
    
    # 1. measure the total latency 
    torch.cuda.synchronize()    
    total_start = time.time()
    
    outputs = engine.generate(prompts, {"max_new_tokens": output_len, "min_new_tokens": output_len})
    
    torch.cuda.synchronize()
    total_latency = time.time() - total_start
    total_tokens = (input_len + output_len) * batch_size
    throughput = total_tokens / total_latency
    
    total_latency = round(total_latency, 3)
    throughput = round(throughput, 2)
    
    print(f"Total latency: {total_latency} s, throughput: {throughput} token/s")
    
    measurement_results["latency"] = total_latency
    measurement_results["throughput"] = throughput
           
    # 停止 profiler 并监控 trace 文件
    if profile:
        try:
            engine.stop_profile()            
            profiler_dir = os.getenv("SGLANG_TORCH_PROFILER_DIR")
            if profiler_dir:
                time.sleep(1)                
                monitor_trace_file()                
            else:
                print("SGLANG_TORCH_PROFILER_DIR is not set, cannot monitor trace file")
        except Exception as e:
            print(f"Stop profiler error: {e}")
    
    return measurement_results

def load_prompts_from_dataset(dataset_path, num_samples=10):
    if not dataset_path:
        return []
        
    prompts = []
    try:        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
                   
        for value in list(data.values())[:num_samples]:
            if isinstance(value, str) and value.startswith("User:"):
                # 删除"User:"前缀
                prompt = re.sub(r'^User:\s*', '', value)
                prompts.append(prompt)
                
        print(f"Successfully loaded {len(prompts)} prompts from dataset")
            
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
    except Exception as e:
        print(f"Error when loading dataset: {e}")
        
    return prompts

def main():
    parser = argparse.ArgumentParser(description="SgLang latency test tool")
    BenchArgs.add_cli_args(parser)
        
    args = parser.parse_args()
    bench_args = BenchArgs.from_cli_args(args)

    newfile_name = f"tp{bench_args.tp_size}_ep{bench_args.ep_size}_dp{bench_args.dp_size}_batch{bench_args.batch_size}_I{bench_args.input_len}_O{bench_args.output_len}"
    os.environ["PROFILE_FILE_NAME"] = newfile_name

    print("=========================== initialize the engine ============================")
    print(f"parallel config: DP={bench_args.dp_size}, TP={bench_args.tp_size}, EP={bench_args.ep_size}")
    
    # 清空文件内容
    PROFIL_DONE_FILE = os.getenv("PROFILE_DONE_FILE")    
    with open(PROFIL_DONE_FILE, "w") as f:
        f.truncate(0)

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
    warmup_batch_size = bench_args.batch_size // 2
    if warmup_batch_size == 0:
        warmup_batch_size = 1
    warmup_input_len = bench_args.input_len
    warmup_prompts = prepare_synthetic_inputs(warmup_batch_size, warmup_input_len)
    warmup_output_len = 2
    llm.generate(warmup_prompts, {"max_new_tokens": warmup_output_len, "min_new_tokens": warmup_output_len})
    
    llm.release_memory_occupation()
    llm.resume_memory_occupation()

    torch.cuda.synchronize()    
    time.sleep(5)

    print("=========================== benchmark ============================")
    result_list = []
        
    print(f"\n测试配置: batch_size={bench_args.batch_size}, input_len={bench_args.input_len}, output_len={bench_args.output_len}")
    try:
        result = run_latency_test_once(
            engine=llm,
            batch_size=bench_args.batch_size,
            input_len=bench_args.input_len,
            output_len=bench_args.output_len,
            profile=bench_args.profile,
            tp_size=bench_args.tp_size,
            ep_size=bench_args.ep_size,
            dp_size=bench_args.dp_size,
            enable_dp_attention=bench_args.enable_dp_attention,
            dataset_path=args.dataset_path,  # 传入数据集路径
        )
        result_list.append(result)  
    except Exception as e:
        print(f"Error: {e}")
        
    if bench_args.profile:
        print("Profiler is enabled, result is not saved")
    else:
        if bench_args.result_filename:
            with open(bench_args.result_filename, "a") as fout:
                for result in result_list:
                    fout.write(json.dumps(result) + "\n")
            print(f"performance result is saved to {bench_args.result_filename}")
    
    print("=========================== 测试完成 ============================")

if __name__ == "__main__":
    main()