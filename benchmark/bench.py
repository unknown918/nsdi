import subprocess
from tqdm import tqdm
import os


def run_bench_serving(
        model_path: str,
        data_path: str,
        bsz: int,
        prompt_per_batch: int = 3,
        input_len: int = 64,
        output_len: int = 16,
        warm_up: int = 2,
        host: str = "127.0.0.1",
        port: int = 30010,
        output_dir: str = "./qwen3/megascale/a1e7-21"
):
    os.makedirs(output_dir, exist_ok=True)
    num_prompts = bsz * prompt_per_batch
    output_jsonl = os.path.join(output_dir, f"bench_batch_size_{bsz}.jsonl")
    stdout_log = os.path.join(output_dir, f"bench_batch_size_{bsz}.log")

    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--num-prompts", f"{num_prompts}",
        "--random-input-len", f"{input_len}",
        "--random-output-len", f"{output_len}",
        "--random-range-ratio", "1.0",
        "--dataset-name", "random",
        "--dataset-path", f"{data_path}",
        "--max-concurrency", f"{bsz}",
        "--seed", "42",
        "--host", f"{host}",
        "--port", f"{port}",
        "--output-file", f"{output_jsonl}",
        "--model", f"{model_path}"
    ]

    with open(os.devnull, "w") as fnull:
        for _ in range(warm_up):
            try:
                subprocess.run(cmd, stdout=fnull, stderr=fnull, check=True)
            except subprocess.CalledProcessError as e:
                raise e

    with open(stdout_log, "w") as log_file:
        try:
            subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise e


if __name__ == "__main__":
    model_path = "../Qwen3-235B"
    data_path = "./ShareGPT_V3.json"
    input_len = 16
    output_len = 64
    prompt_per_batch = 1

    batch_sizes = [4, 16, 64, 256, 512]

    for bsz in tqdm(batch_sizes):
        run_bench_serving(
            model_path=model_path,
            data_path=data_path,
            bsz=bsz,
            prompt_per_batch=prompt_per_batch,
            input_len=input_len,
            output_len=output_len,
            warm_up=2,
            port=30010
        )
