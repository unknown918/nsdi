import numpy as np
import time
import torch
import glob
from typing import Dict, List

from sglang.srt.configs.logger_config import configure_logger
logger = configure_logger(__name__)


def random_expert_scheduling(topk_ids, logical_to_all_physical_map, logical_to_all_physical_map_num_valid):
    """
    使用EPLB随机选择冗余expert的方法，将逻辑专家ID转换为物理专家ID
    
    Args:
        topk_ids: 逻辑专家ID，形状 (B, K)
        logical_to_all_physical_map: 逻辑到物理专家映射，形状 (num_logical_experts, max_replicas)
        logical_to_all_physical_map_num_valid: 每个逻辑专家的有效副本数，形状 (num_logical_experts)
    
    Returns:
        torch.Tensor: 转换后的物理专家ID，形状与topk_ids相同
    """
    # 保存原始形状
    topk_ids_original_shape = topk_ids.shape
    device = topk_ids.device
    
    # 展平topk_ids以便处理
    topk_ids = topk_ids.flatten()
    
    # 使用和SGLang相同的方法：随机选择物理专家副本
    # 1. 生成随机索引
    chosen_dispatch_index = (
        torch.randint(0, 65536, topk_ids.shape, dtype=torch.int32, device=device)
        % logical_to_all_physical_map_num_valid[topk_ids]
    )
    
    # 2. 根据随机索引选择物理专家ID
    # 从logical_to_all_physical_map中获取对应的物理专家ID
    topk_ids = logical_to_all_physical_map[topk_ids, chosen_dispatch_index]
    
    # 恢复原始形状
    topk_ids = topk_ids.view(topk_ids_original_shape)
    
    return topk_ids

def greedy_expert_scheduling_split(topk_ids, expert_deployment, f=None, eplb_structures=None):

    B, K = topk_ids.shape
    device = topk_ids.device

    import torch, time
    torch.cuda.synchronize()
    time1 = time.time()
    
    batch_indices = torch.arange(B, device=device).repeat_interleave(K)
    k_indices = torch.arange(K, device=device).repeat(B)
    original_token_positions = torch.stack([batch_indices, k_indices], dim=1)  # shape (B*K, 2)

    flat_topk_ids = topk_ids.reshape(-1)

    # --- Original time_spend 1 to time_spend 2 ---
    # (Sort flat_topk_ids, find unique experts, counts, inverse indices)
    # Modify: also capture sort_indices to reorder token_positions
    sorted_flat_topk_ids, sort_indices = torch.sort(flat_topk_ids)
    unique_experts, inverse_indices, counts = torch.unique_consecutive(
        sorted_flat_topk_ids, return_inverse=True, return_counts=True
    )
    # Reorder original_token_positions based on the sort order
    sorted_token_positions = original_token_positions[sort_indices] # (B*K, 2), now positions are sorted by expert ID

    # Calculate start indices for each unique expert in the sorted lists (on GPU)
    # This tells us where each expert's block starts in sorted_token_positions
    expert_start_indices = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int64),
        counts.cumsum(0)[:-1]
    ]) # (num_unique_experts,)


    # --- Transfer necessary data to CPU *once* and build Python structures efficiently ---

    # Convert GPU tensors needed for building Python dicts to Python lists
    unique_experts_list = unique_experts.tolist()
    counts_list = counts.tolist()
    expert_start_indices_list = expert_start_indices.tolist()
    # Convert the *entire* sorted token positions tensor to a single Python list of lists/tuples
    # This is one large transfer instead of many small ones or per-append item transfers.
    sorted_token_positions_list = sorted_token_positions.tolist() # (B*K, 2) -> list of [b, k] lists

    # Build expert_count_dict (same logic as original time_spend 3)
    expert_count_dict = dict(zip(unique_experts_list, counts_list))

    # Build expert_token_indices efficiently by slicing the pre-generated list
    expert_token_indices = {}
    # unique_experts_list corresponds to counts_list and expert_start_indices_list by index
    for i, expert_id in enumerate(unique_experts_list):
        start_idx = expert_start_indices_list[i]
        count = counts_list[i]
        # Slice the single large list to get the positions for this expert
        positions_list_of_lists = sorted_token_positions_list[start_idx : start_idx + count]
        # Convert inner lists [b, k] to tuples (b, k) to match original exact format
        expert_token_indices[expert_id] = [tuple(pos) for pos in positions_list_of_lists]


    # expert_token_dict = list(expert_count_dict.items()) # This variable is unused in original flow after this point, can remove

    # Set expert_tokens for the greedy algorithm (uses expert_count_dict)
    expert_tokens = expert_count_dict # Direct assignment is sufficient


    experts_set = set() # Use a different name to avoid clash with unique_experts tensor if full function is used
    for gpu_id, gpu_experts in expert_deployment.items():
        experts_set.update(gpu_experts)

    gpu_tokens = {gpu_id: 0 for gpu_id in expert_deployment}
    gpu_experts = {gpu_id: 0 for gpu_id in expert_deployment}
    allocation = {}

    token_allocation = {gpu_id: {} for gpu_id in expert_deployment}

    # Sort experts by token count (descending) using the expert_tokens dict
    # TODO 排序有意义吗，和expert_set 应该一样吧
    sorted_experts = sorted(list(experts_set), key=lambda x: expert_tokens.get(x, 0), reverse=True)

    # return None
    # torch.cuda.synchronize()        
    # time2 = time.time()
    # print(time2-time1)
    # print("[][]")
    
    # First phase: Assign non-redundant experts
    for expert_id in sorted_experts:
        deployed_gpus = [gpu_id for gpu_id, gpu_experts in expert_deployment.items() if expert_id in gpu_experts]

        if len(deployed_gpus) == 1:
            gpu_id = deployed_gpus[0]
            tokens = expert_tokens.get(expert_id, 0)
            if tokens > 0:
                allocation[(expert_id, gpu_id)] = tokens
                gpu_tokens[gpu_id] += tokens
                # Increment expert count only if this is the first token for this expert on this GPU
                if expert_id not in token_allocation[gpu_id] or not token_allocation[gpu_id][expert_id]:
                     gpu_experts[gpu_id] += 1
                # Assign all token indices from the precomputed list
                # .get(expert_id, []) handles experts with 0 tokens, which might be in experts_set but not expert_token_indices
                token_allocation[gpu_id][expert_id] = expert_token_indices.get(expert_id, [])




    # Second phase: Assign redundant experts
    for expert_id in sorted_experts:
         deployed_gpus = [gpu_id for gpu_id, gpu_experts in expert_deployment.items() if expert_id in gpu_experts]

         if len(deployed_gpus) > 1:
             tokens = expert_tokens.get(expert_id, 0)
             if tokens == 0: continue

             token_indices = expert_token_indices.get(expert_id, []) # Get the full list of indices

             # Calculate all GPU's potential latency if all tokens are assigned here
             current_latencies = {}
             for gpu_id in deployed_gpus:
                 expert_count = gpu_experts[gpu_id]
                 # Count as new expert if it's not already allocated tokens on this GPU
                 new_expert_count = expert_count + (1 if expert_id not in token_allocation[gpu_id] or not token_allocation[gpu_id][expert_id] else 0)
                 current_latencies[gpu_id] = f(gpu_tokens[gpu_id] + tokens, new_expert_count)

             min_gpu_id = min(current_latencies, key=current_latencies.get)
 
             if 1>0: # No significant imbalance or only one deployed GPU, allocate all to min_gpu_id
                 #Update allocation count - add to existing if expert was already partially allocated
                allocation[(expert_id, min_gpu_id)] = allocation.get((expert_id, min_gpu_id), 0) + tokens
                if expert_id not in token_allocation[min_gpu_id]:
                    token_allocation[min_gpu_id][expert_id] = []
                # Extend the list with *all* indices for this expert
                token_allocation[min_gpu_id][expert_id].extend(token_indices) # Extend Python list

                gpu_tokens[min_gpu_id] += tokens
                # Update GPU's expert count *if* this allocation made it non-zero for this expert on this GPU
                if allocation.get((expert_id, min_gpu_id), 0) == tokens: # If this is the first allocation batch
                    gpu_experts[min_gpu_id] += 1

    # torch.cuda.synchronize()        
    # time3 = time.time()
    # print(time3-time2)
    # print("[3][2]")
    
    
    # 创建新的topk_ids，使用物理专家ID
    B, K = topk_ids.shape
    device = topk_ids.device
    new_topk_ids = torch.full((B, K), -1, dtype=torch.int64, device=device)
    
    # 使用预计算的映射
    for gpu_id, expert_dict in token_allocation.items():
        for expert_id, token_indices in expert_dict.items():
            if not token_indices:  # Skip if empty
                continue
                
            physical_id = eplb_structures.logical_to_physical_by_gpu_map.get((expert_id, gpu_id))
            if physical_id is not None:
                # 将token_indices转换为张量索引
                indices = torch.tensor(token_indices, device=device)
                # 批量更新所有位置
                new_topk_ids[indices[:, 0], indices[:, 1]] = physical_id
                    
    # print(token_allocation)
    # print(new_topk_ids)
    
    # torch.cuda.synchronize()        
    # time4 = time.time()
    # print(time4-time3)
    # print("[4][3]")
    return new_topk_ids

# 0.03494548797607422 秒