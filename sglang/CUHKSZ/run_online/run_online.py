import sys
import os
from dotenv import load_dotenv

# in online mode， environment variables are set in /home/moe/amdissagCore/janus/srt/entrypoints/http_server.py

current_dir = os.path.dirname(__file__)

env_path = os.path.join(current_dir, "../userConfig/.env")
att_env_path = os.path.join(current_dir, "../userConfig/att.env")
load_dotenv(env_path)  
load_dotenv(att_env_path)  

REPO_PATH = os.getenv("REPO_PATH")
sys.path.insert(0, REPO_PATH)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["UCX_NET_DEVICES"] = "all"
os.environ["UCX_TLS"] = "all"
 
from sglang.CUHKSZ.disaggregate.dist_patch import apply_world_size_patch
apply_world_size_patch()
from sglang.utils import (
    execute_shell_command,
    wait_for_server,    
)

# python3 -m janus.launch_server --model-path /home/moe/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9 --port=30010 --disable-cuda-graph --disable-overlap-schedule --tp-size=1 --dp-size=1 --enable-p2p-check

model_used = os.getenv("MODEL_USED")

if model_used == "Qwen":
    model_path =  os.getenv("MODEL_PATH_QWEN")
elif model_used == "DS2":
    model_path =  os.getenv("MODEL_PATH_DS2")
elif model_used == "DS3":
    model_path =  os.getenv("MODEL_PATH_DS3")
else:
    raise ValueError(f"Invalid model used: {model_used}")

dp_size = int(os.getenv("ATTN_WORKERS_COUNT"))  

server_process = execute_shell_command(
    f"""
    python {os.path.join(REPO_PATH, "janus/launch_server.py")} \
    --model-path {model_path} \
    --port=30010 \
    --disable-cuda-graph \    
    --tp-size={dp_size} \
    --dp-size={dp_size} \
    --enable-p2p-check \   
    --trust-remote-code \
    --disable-radix-cache \
    --enable-dp-attention
"""
)

# --log-level info \
#     --enable-metrics \
#     --show-time-cost \
# --disable-overlap-schedule \
    #     
    #     --chunked-prefill-size=-1 \
            #     --schedule-policy fcfs \   
        #     --disable-overlap-schedule \
                    
                # --prefill-only-one-req=true \  
wait_for_server("http://localhost:30010")
