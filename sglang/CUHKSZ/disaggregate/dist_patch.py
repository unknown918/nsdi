import os
import torch.distributed as dist
from dotenv import load_dotenv

current_dir = os.path.dirname(__file__)
env_path = os.path.join(current_dir, "../userConfig/.env")
load_dotenv(env_path)  

original_get_world_size = dist.get_world_size

# replace the original get_world_size function
def patched_get_world_size(group=None):

    dp_size = int(os.getenv("ATTN_WORKERS_COUNT"))
    ep_size = int(os.getenv("MOE_WORKERS_COUNT"))
    if group is None:
        return dp_size
    original_world_size = original_get_world_size(group)
    # 获取当前group中的rank列表
    ranks = dist.get_process_group_ranks(group)
    # 最大rank
    max_rank = max(ranks)
    if max_rank + 1 >= dp_size + ep_size:
        return dp_size
    else:
        return original_world_size
    # dp_size = int(os.getenv("ATTN_WORKERS_COUNT"))
    # ep_size = int(os.getenv("MOE_WORKERS_COUNT"))
    # if dp_size + ep_size == original_world_size:
    #     return original_world_size - ep_size
    # if os.getenv("MOE_WORKERS_COUNT") is not None:
    #     moe_workers = int(os.getenv("MOE_WORKERS_COUNT"))
    #     return original_world_size - moe_workers
    
    # return original_world_size

# apply the patch
def apply_world_size_patch():
    dist.get_world_size = patched_get_world_size
    print("applied the patch of torch.distributed.get_world_size")

# restore the original function
def restore_world_size_original():
    dist.get_world_size = original_get_world_size
    print("restore the original torch.distributed.get_world_size function") 