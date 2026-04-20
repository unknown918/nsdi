import numpy as np
import time
import torch
from typing import Dict, List, Tuple, Set
import os
# 设置cuda visable 为gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def greedy_expert_scheduling_optimized_preproc(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_deployment: Dict[int, set],
    f=None,
    hidden_states: torch.Tensor = None
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, List[int]]]:

    B, K = topk_ids.shape
    device = topk_ids.device

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

    # --- Optimized Replacement for original time_spend 3 to time_spend 4 ---
    # Goal: Build expert_count_dict and expert_token_indices efficiently on CPU
    # by leveraging the sorted_token_positions and unique/counts on GPU.
    # Avoid the slow Python loop with append in original time_spend 4.

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
             min_latency = current_latencies[min_gpu_id]

             # Calculate latency on other GPUs (those *not* considered for this expert's full allocation)
             other_latencies_values = [f(gpu_tokens[gid], gpu_experts[gid])
                                       for gid in expert_deployment if gid not in deployed_gpus]
             if not other_latencies_values: # If the expert is deployed on ALL GPUs
                 other_latencies_values = [f(gpu_tokens[gid], gpu_experts[gid]) for gid in expert_deployment]

             second_min_latency = min(other_latencies_values) if other_latencies_values else float('inf')


             # If full allocation creates significant imbalance, split the load
             imbalance_threshold = 1.2 # Configurable threshold
             if min_latency > second_min_latency * imbalance_threshold and len(deployed_gpus) > 1:
                 # Split tokens based on current load
                 remaining_tokens = tokens
                 remaining_indices = token_indices.copy() # Make a copy of the Python list
                 gpu_latencies_current = [(gpu_id, f(gpu_tokens[gpu_id], gpu_experts[gpu_id]))
                                  for gpu_id in deployed_gpus]

                 # Sort GPUs by current latency
                 gpu_latencies_current.sort(key=lambda x: x[1])

                 for i, (gpu_id, _) in enumerate(gpu_latencies_current):
                     # Process up to the second to last GPU
                     if i == len(gpu_latencies_current) - 1 or remaining_tokens == 0:
                         break

                     # Bring this GPU's latency closer to the next GPU's latency in the sorted list
                     next_gpu_id, next_latency = gpu_latencies_current[i+1]

                     # Binary search to find optimal token allocation for this GPU
                     low, high = 0, remaining_tokens
                     best_allocation = 0

                     while low <= high:
                         mid = (low + high) // 2
                         # Potential new expert count if 'mid' tokens are allocated
                         new_expert_count_if_allocated = gpu_experts[gpu_id] + (1 if expert_id not in token_allocation[gpu_id] or not token_allocation[gpu_id][expert_id] else 0)
                         new_latency = f(gpu_tokens[gpu_id] + mid, new_expert_count_if_allocated)

                         if new_latency < next_latency:
                             best_allocation = mid
                             low = mid + 1
                         else:
                             high = mid - 1

                     # Allocate tokens to this GPU based on best_allocation
                     if best_allocation > 0:
                         # Update allocation count - add to existing
                         allocation[(expert_id, gpu_id)] = allocation.get((expert_id, gpu_id), 0) + best_allocation
                         # Assign concrete token indices from the remaining list
                         selected_indices = remaining_indices[:best_allocation]
                         if expert_id not in token_allocation[gpu_id]:
                             token_allocation[gpu_id][expert_id] = []
                         token_allocation[gpu_id][expert_id].extend(selected_indices) # Extend Python list

                         # Update remaining tokens and indices list
                         remaining_indices = remaining_indices[best_allocation:]
                         remaining_tokens -= best_allocation

                         # Update GPU's total token count
                         gpu_tokens[gpu_id] += best_allocation
                         # Update GPU's expert count *if* this allocation made it non-zero for this expert on this GPU
                         if allocation.get((expert_id, gpu_id), 0) == best_allocation: # If this is the first allocation batch
                              gpu_experts[gpu_id] += 1


                 # Allocate any remaining tokens to the last GPU in the sorted list
                 if remaining_tokens > 0:
                     last_gpu_id = gpu_latencies_current[-1][0]
                     # Add to existing allocation
                     allocation[(expert_id, last_gpu_id)] = allocation.get((expert_id, last_gpu_id), 0) + remaining_tokens
                     if expert_id not in token_allocation[last_gpu_id]:
                          token_allocation[last_gpu_id][expert_id] = []
                     token_allocation[last_gpu_id][expert_id].extend(remaining_indices) # Extend Python list

                     # Update GPU's expert count *if* this allocation made it non-zero for this expert on this GPU
                     if allocation.get((expert_id, last_gpu_id), 0) == remaining_tokens: # If this is the first allocation batch
                          gpu_experts[last_gpu_id] += 1
                     gpu_tokens[last_gpu_id] += remaining_tokens

             else: # No significant imbalance or only one deployed GPU, allocate all to min_gpu_id
                 # Update allocation count - add to existing if expert was already partially allocated
                 allocation[(expert_id, min_gpu_id)] = allocation.get((expert_id, min_gpu_id), 0) + tokens
                 if expert_id not in token_allocation[min_gpu_id]:
                     token_allocation[min_gpu_id][expert_id] = []
                 # Extend the list with *all* indices for this expert
                 token_allocation[min_gpu_id][expert_id].extend(token_indices) # Extend Python list

                 gpu_tokens[min_gpu_id] += tokens
                 # Update GPU's expert count *if* this allocation made it non-zero for this expert on this GPU
                 if allocation.get((expert_id, min_gpu_id), 0) == tokens: # If this is the first allocation batch
                      gpu_experts[min_gpu_id] += 1


    # 1. 每个gpu上有哪些token
    all_gpu_b_pairs_set = set()  # 
    for gpu_id, expert_allocations in token_allocation.items():
        if not expert_allocations or all(not token_list for token_list in expert_allocations.values()):
             continue # Skip GPUs with no tokens allocated

        for expert_id, token_list in expert_allocations.items():
            # token_list is a list of (b, k) tuples
            for b, k in token_list:
                 all_gpu_b_pairs_set.add((gpu_id, b)) # Collect each unique (gpu_id, batch_idx) pair
    
    # Convert set to sorted list
    all_gpu_b_pairs_list = sorted(list(all_gpu_b_pairs_set))

    # 2. Move to GPU
    unique_gpu_b_pairs = torch.tensor(all_gpu_b_pairs_list, device=device, dtype=torch.int64) # Shape (UniqueAllocatedTokens, 2)
    # ----------————————————————————————————————————————————————————————————————————————————————
    # 获取每个gpu的切片

    # unique_gb_gpu_ids
    # tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3], device='cuda:0')
    # unique_gb_b_indices
    # tensor([0, 2, 3, 4, 1, 3, 4, 0, 2, 3, 0, 1], device='cuda:0')
    # gpu_boundaries_indices
    # tensor([ 4,  7, 10], device='cuda:0')
    # gpu_boundaries
    # tensor([ 0,  4,  7, 10, 12], device='cuda:0')

    unique_gb_gpu_ids = unique_gpu_b_pairs[:, 0] # Shape (U,)
    unique_gb_b_indices = unique_gpu_b_pairs[:, 1] # Shape (U,)

    # 5. Find start indices for each unique gpu_id in unique_gpu_b_pairs (on GPU)
    # This identifies blocks in unique_gb_b_indices belonging to each GPU ID
    # We find indices where the gpu_id column changes value.
    gpu_boundaries_indices = (unique_gb_gpu_ids[:-1] != unique_gb_gpu_ids[1:]).nonzero(as_tuple=False)[:, 0] + 1
    gpu_boundaries = torch.cat([
            torch.tensor([0], device=device, dtype=torch.int64),
            gpu_boundaries_indices,
            torch.tensor([unique_gb_gpu_ids.numel()], device=device, dtype=torch.int64) # Add end boundary
        ]) # Shape (num_unique_gpus_with_tokens + 1,)

    # ----------————————————————————————————————————————————————————————————————————————————————
    # 6. Populate gpu_token_indices (slice_mapping) and slice tensors
   
    gpu_token_indices = {}
    slice_mapping = {}
    gpu_hidden_states = {}
    gpu_topk = {}
    gpu_topk_weights = {}

    # Iterate through the unique GPU IDs that received tokens (on CPU, using list from GPU tensor)
    unique_gpus_with_tokens_list = unique_gb_gpu_ids[gpu_boundaries[:-1]].tolist()

    # Pre-allocate some tensors to reduce memory fragmentation
    all_indices = {}
    
    for i, gpu_id in enumerate(unique_gpus_with_tokens_list):
        start_bound = gpu_boundaries[i]
        end_bound = gpu_boundaries[i+1]

        # Keep indices as tensors to avoid unnecessary conversions
        batch_indices_for_gpu_tensor = unique_gb_b_indices[start_bound:end_bound]
        
        # Store the tensor
        all_indices[gpu_id] = batch_indices_for_gpu_tensor
        
        # Convert to list only once for output format requirement
        indices_list = batch_indices_for_gpu_tensor.tolist()
        gpu_token_indices[gpu_id] = indices_list
        slice_mapping[gpu_id] = indices_list
    
    # Process each tensor operation in batch (across all GPUs) instead of one GPU at a time
    for gpu_id, indices_tensor in all_indices.items():
        if indices_tensor.numel() > 0:
            # Use index_select for more efficient indexing (avoids unnecessary memory operations)
            if hidden_states is not None:
                gpu_hidden_states[gpu_id] = hidden_states.index_select(0, indices_tensor)
            
            # Use index_select instead of indexing with list + clone
            gpu_topk[gpu_id] = topk_ids.index_select(0, indices_tensor)
            gpu_topk_weights[gpu_id] = topk_weights.index_select(0, indices_tensor)
        else:
            # Empty case
            if hidden_states is not None:
                gpu_hidden_states[gpu_id] = None
            
            # Create empty tensors with correct dimensions
            gpu_topk[gpu_id] = torch.empty(0, K, dtype=topk_ids.dtype, device=device)
            gpu_topk_weights[gpu_id] = torch.empty(0, K, dtype=topk_weights.dtype, device=device)

    
    # 边缘case： 如果一个gpu没有token，也需要初始化
    for gpu_id in expert_deployment:
        if gpu_id not in gpu_token_indices: # If a GPU was in deployment but got no tokens
                gpu_token_indices[gpu_id] = []
                slice_mapping[gpu_id] = []
                if hidden_states is not None:
                    gpu_hidden_states[gpu_id] = None # Or empty tensor depending on desired output for empty case
                gpu_topk[gpu_id] = torch.empty(0, K, dtype=topk_ids.dtype, device=device)
                gpu_topk_weights[gpu_id] = torch.empty(0, K, dtype=topk_weights.dtype, device=device)

    print(gpu_hidden_states)
    print(gpu_topk)
    print(gpu_topk_weights)
    print(slice_mapping)
    
    return gpu_hidden_states, gpu_topk, gpu_topk_weights, slice_mapping

