import sys
import os
from dotenv import load_dotenv

# in online mode， environment variables are set in /home/moe/amdissagCore/janus/srt/entrypoints/http_server.py

current_dir = os.path.dirname(__file__)

env_path = os.path.join(current_dir, "../userConfig/.env")
att_env_path = os.path.join(current_dir, "../userConfig/att.env")
# load_dotenv(env_path)  
# load_dotenv(att_env_path)  

REPO_PATH = os.getenv("REPO_PATH")
sys.path.insert(0, REPO_PATH)
import torch.distributed as dist
# from janus.CUHKSZ.disaggregate.dist_patch import apply_world_size_patch
# apply_world_size_patch()
import torch
from sglang.srt.distributed import init_distributed_environment 
# python3 -m janus.launch_server --model-path /home/moe/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9 --port=30010 --disable-cuda-graph --disable-overlap-schedule --tp-size=1 --dp-size=1 --enable-p2p-check


# dp_size = int(os.getenv("ATTN_WORKERS_COUNT"))  
# moe_workers_count = int(os.getenv("MOE_WORKERS_COUNT"))
# nccl_master_ip = os.getenv("NCCL_MASTER_IP")
# nccl_master_port = os.getenv("NCCL_MASTER_PORT")
# print(f"init_distributed_environment begin, world_size={dp_size + moe_workers_count}, rank=0, local_rank=0")

# init_distributed_environment(
#     backend="nccl",
#     world_size=dp_size + moe_workers_count,
#     rank=0,
#     local_rank=0,
#     distributed_init_method=f"tcp://{nccl_master_ip}:{nccl_master_port}",
# )

RANK = os.getenv("RANK")
WORLD_SIZE = os.getenv("WORLD_SIZE")
MASTER_ADDR = os.getenv("MASTER_ADDR")
MASTER_PORT = os.getenv("MASTER_PORT")
LOCAL_RANK = os.getenv("LOCAL_RANK")
print(f"init_distributed_environment begin, world_size={WORLD_SIZE}, rank={RANK}, local_rank={LOCAL_RANK}")

torch.distributed.init_process_group(
        backend="nccl",
        init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
        world_size=WORLD_SIZE,           
        rank=RANK,
    )  
print(f"init_distributed_environment success, world_size={WORLD_SIZE}, rank={RANK}, local_rank={LOCAL_RANK}")

data = torch.tensor([0], dtype=torch.float32).cuda()
# 存放所有 rank 的结果
gather_list = [torch.zeros_like(data) for _ in range(3)]
# 执行 all_gather
dist.all_gather(gather_list, data)
# reduce

# dist.reduce(data, dst=0)
print(f"data={data}")

# 启动
# torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29501 run_online.py