import subprocess
import os
# Read SGLANG_FOLDER from ./config.env (no fallback)
def load_sglang_folder(env_path: str) -> str:
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("SGLANG_FOLDER="):
                val = line.split("=", 1)[1].strip()
                if val.startswith('"') and val.endswith('"'):
                    val = val[1:-1]
                return val
    raise KeyError("SGLANG_FOLDER not found in config.env")

script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, "config.env")
SGLANG_FOLDER = load_sglang_folder(env_path)

# --- 固定参数 ---
base_command = (
    f"python {SGLANG_FOLDER}/bench_serving.py "
    "--backend janus "
    "--seed 42 "
    "--host 127.0.0.1 "
    "--port 30010 "
    "--model /wangye/models/DeepSeek-V2 "
    "--dataset-path /wangye/models/ShareGPT_Vicuna_unfiltered/ShareGPT_V3.json "
    "--dataset-name random "
    "--random-output-len 64 "
    "--random-range-ratio 1.0 "
    "--random-input-len 512 "
    "--num-prompts 2 "      # batch = 1
    "--max-concurrency 2"   # 单并发
)

print("执行命令:")
print(base_command)

try:
    result = subprocess.run(
        base_command,
        shell=True,
        check=True,
        text=True,
        capture_output=False  # 不捕获，直接打印 stdout/stderr
    )
except subprocess.CalledProcessError as e:
    print(f"!!! 命令执行出错，退出码: {e.returncode}")