# Start combining
import numpy as np
import time
import torch
from typing import Dict, List, Tuple, Set

def greedy_expert_scheduling_user_hybrid(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor, # topk_weights is needed in original postproc
    expert_deployment: Dict[int, set],
    f=None,
    hidden_states: torch.Tensor = None
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, List[int]]]:
    """
    Expert scheduling - User's Hybrid version (Optimized Preproc + Original Algo + Original Postproc)

    Combines the optimized preprocessing for faster token counting and position mapping
    with the original code's greedy algorithm and postprocessing logic.
    Aims to leverage faster preproc while retaining original postproc performance/behavior.
    """

    if f is None:
        raise ValueError("No latency prediction function provided")

    # --- Optimized Preprocessing (Replacing original time_spend 0 to time_spend 4) ---
    # Goal: Efficiently build expert_count_dict (expert_tokens) and expert_token_indices (dict of lists of tuples)

    torch.cuda.synchronize()
    start_time_0 = time.time()

    B, K = topk_ids.shape
    device = topk_ids.device

    # Original time_spend 0 to 1: Create basic position tensors and flatten
    batch_indices = torch.arange(B, device=device).repeat_interleave(K)
    k_indices = torch.arange(K, device=device).repeat(B)
    original_token_positions = torch.stack([batch_indices, k_indices], dim=1) # (B*K, 2)
    flat_topk_ids = topk_ids.reshape(-1)

    # torch.cuda.synchronize()
    # start_time_1 = time.time()
    # print(f"time spend 1 (preproc-basic): {(start_time_1 - start_time_0)*1000:.2f} ms")

    # Original time_spend 1 to 2: Sort flat_topk_ids, find unique experts, counts
    # Modified: Capture sort_indices and reorder token_positions
    sorted_flat_topk_ids, sort_indices = torch.sort(flat_topk_ids)
    unique_experts, inverse_indices, counts = torch.unique_consecutive(
        sorted_flat_topk_ids, return_inverse=True, return_counts=True
    )
    # Reorder original_token_positions based on the sort order
    sorted_token_positions = original_token_positions[sort_indices] # (B*K, 2), positions sorted by expert ID

    # torch.cuda.synchronize()
    # start_time_2 = time.time()
    # print(f"time spend 2 (preproc-sort/unique): {(start_time_2 - start_time_1)*1000:.2f} ms")

    # Optimized: Calculate GPU start indices and transfer to CPU lists
    expert_start_indices = torch.cat([
        torch.tensor([0], device=device, dtype=torch.int64),
        counts.cumsum(0)[:-1]
    ]) # (num_unique_experts,)

    unique_experts_list = unique_experts.tolist()
    counts_list = counts.tolist()
    expert_start_indices_list = expert_start_indices.tolist()
    sorted_token_positions_list = sorted_token_positions.tolist() # One large transfer

    # torch.cuda.synchronize()
    # start_time_3_opt = time.time()
    # print(f"time spend 3_opt (preproc-gpu_ops+transfer): {(start_time_3_opt - start_time_2)*1000:.2f} ms")

    # Optimized: Build Python dicts efficiently on CPU
    expert_count_dict = dict(zip(unique_experts_list, counts_list))

    expert_token_indices = {}
    for i, expert_id in enumerate(unique_experts_list):
        start_idx = expert_start_indices_list[i]
        count = counts_list[i]
        positions_list_of_lists = sorted_token_positions_list[start_idx : start_idx + count]
        expert_token_indices[expert_id] = [tuple(pos) for pos in positions_list_of_lists] # Ensure tuples

    expert_tokens = expert_count_dict # Set expert_tokens for the greedy algorithm

    torch.cuda.synchronize()
    start_time_4_opt = time.time()
    print(f"time spend 4_opt (preproc-build_cpu_dicts): {(start_time_4_opt - start_time_0)*1000:.2f} ms")
    # print(f"time spend 4_opt (preproc-build_cpu_dicts): {(start_time_4_opt - start_time_3_opt)*1000:.2f} ms")
    # End of Optimized Preprocessing


    # --- Original/Optimized Greedy Algorithm Core (start_time_5 to start_time_6) ---
    # Uses expert_tokens and expert_token_indices from the optimized preproc.
    # Logic is the same as original code.
    torch.cuda.synchronize()
    start_time_5 = time.time()

    experts_set = set() # Set of all deployed experts
    for gpu_id, gpu_experts in expert_deployment.items():
        experts_set.update(gpu_experts)

    gpu_tokens = {gpu_id: 0 for gpu_id in expert_deployment}
    gpu_experts = {gpu_id: 0 for gpu_id in expert_deployment}
    allocation = {} # (expert_id, gpu_id) -> count

    token_allocation = {gpu_id: {} for gpu_id in expert_deployment} # {gpu_id: {expert_id: [(b, k), ...]}}

    sorted_experts = sorted(list(experts_set), key=lambda x: expert_tokens.get(x, 0), reverse=True) # Sort by token count

    # Phase 1: Assign non-redundant experts
    for expert_id in sorted_experts:
        deployed_gpus = [gpu_id for gpu_id, gpu_experts in expert_deployment.items() if expert_id in gpu_experts]
        if len(deployed_gpus) == 1:
            gpu_id = deployed_gpus[0]
            tokens = expert_tokens.get(expert_id, 0)
            if tokens > 0:
                allocation[(expert_id, gpu_id)] = tokens
                gpu_tokens[gpu_id] += tokens
                # Check if this expert is new to this GPU's allocation
                # Original check: if (expert_id, gpu_id) not in allocation
                # Using the more robust check based on token_allocation content:
                if expert_id not in token_allocation[gpu_id] or not token_allocation[gpu_id][expert_id]:
                    gpu_experts[gpu_id] += 1
                token_allocation[gpu_id][expert_id] = expert_token_indices.get(expert_id, []) # Assign full list

    # Phase 2: Assign redundant experts
    for expert_id in sorted_experts:
         deployed_gpus = [gpu_id for gpu_id, gpu_experts in expert_deployment.items() if expert_id in gpu_experts]

         if len(deployed_gpus) > 1:
             tokens = expert_tokens.get(expert_id, 0)
             if tokens == 0: continue

             token_indices = expert_token_indices.get(expert_id, []) # Get the full list of indices

             current_latencies = {} # Potential latency if all tokens are assigned to a deployed GPU
             for gpu_id in deployed_gpus:
                 expert_count = gpu_experts[gpu_id]
                 new_expert_count = expert_count + (1 if expert_id not in token_allocation[gpu_id] or not token_allocation[gpu_id][expert_id] else 0) # More robust check
                 current_latencies[gpu_id] = f(gpu_tokens[gpu_id] + tokens, new_expert_count)

             min_gpu_id = min(current_latencies, key=current_latencies.get)
             min_latency = current_latencies[min_gpu_id]

             # Calculate latency on other GPUs (those NOT in deployed_gpus, or all if deployed_gpus covers all)
             other_latencies_values = [f(gpu_tokens[gid], gpu_experts[gid])
                                       for gid in expert_deployment if gid not in deployed_gpus]
             if not other_latencies_values:
                 other_latencies_values = [f(gpu_tokens[gid], gpu_experts[gid]) for gid in expert_deployment]

             second_min_latency = min(other_latencies_values) if other_latencies_values else float('inf')

             imbalance_threshold = 1.2 # Configurable threshold
             if min_latency > second_min_latency * imbalance_threshold and len(deployed_gpus) > 1:
                 # Split tokens
                 remaining_tokens = tokens
                 remaining_indices = token_indices.copy() # Copy the list

                 gpu_latencies_current = [(gpu_id, f(gpu_tokens[gpu_id], gpu_experts[gpu_id])) for gpu_id in deployed_gpus]
                 gpu_latencies_current.sort(key=lambda x: x[1]) # Sort GPUs by current latency

                 for i, (gpu_id, _) in enumerate(gpu_latencies_current):
                     if i == len(gpu_latencies_current) - 1 or remaining_tokens == 0: break # Stop at last GPU or if done

                     next_gpu_id, next_latency = gpu_latencies_current[i+1] # Next GPU in sorted list

                     low, high = 0, remaining_tokens
                     best_allocation = 0 # Tokens to allocate to current gpu_id

                     while low <= high:
                         mid = (low + high) // 2
                         new_expert_count_if_allocated = gpu_experts[gpu_id] + (1 if expert_id not in token_allocation[gpu_id] or not token_allocation[gpu_id][expert_id] else 0)
                         new_latency = f(gpu_tokens[gpu_id] + mid, new_expert_count_if_allocated)

                         if new_latency < next_latency:
                             best_allocation = mid
                             low = mid + 1
                         else:
                             high = mid - 1

                     if best_allocation > 0:
                         allocation[(expert_id, gpu_id)] = allocation.get((expert_id, gpu_id), 0) + best_allocation
                         selected_indices = remaining_indices[:best_allocation]
                         if expert_id not in token_allocation[gpu_id]:
                             token_allocation[gpu_id][expert_id] = []
                         token_allocation[gpu_id][expert_id].extend(selected_indices) # Extend Python list

                         remaining_indices = remaining_indices[best_allocation:]
                         remaining_tokens -= best_allocation
                         gpu_tokens[gpu_id] += best_allocation
                         if allocation.get((expert_id, gpu_id), 0) == best_allocation: # First allocation batch
                              gpu_experts[gpu_id] += 1

                 # Allocate remaining tokens to the last GPU
                 if remaining_tokens > 0:
                     last_gpu_id = gpu_latencies_current[-1][0]
                     allocation[(expert_id, last_gpu_id)] = allocation.get((expert_id, last_gpu_id), 0) + remaining_tokens
                     if expert_id not in token_allocation[last_gpu_id]:
                          token_allocation[last_gpu_id][expert_id] = []
                     token_allocation[last_gpu_id][expert_id].extend(remaining_indices) # Extend Python list
                     if allocation.get((expert_id, last_gpu_id), 0) == remaining_tokens: # First allocation batch
                          gpu_experts[last_gpu_id] += 1
                     gpu_tokens[last_gpu_id] += remaining_tokens

             else: # No significant imbalance, allocate all to min_gpu_id
                 allocation[(expert_id, min_gpu_id)] = allocation.get((expert_id, min_gpu_id), 0) + tokens
                 if expert_id not in token_allocation[min_gpu_id]:
                     token_allocation[min_gpu_id][expert_id] = []
                 token_allocation[min_gpu_id][expert_id].extend(token_indices) # Extend Python list

                 gpu_tokens[min_gpu_id] += tokens
                 if allocation.get((expert_id, min_gpu_id), 0) == tokens: # First allocation batch
                      gpu_experts[min_gpu_id] += 1


    torch.cuda.synchronize()
    start_time_6 = time.time()
    print(f"time spend Algo: {(start_time_6 - start_time_5)*1000:.2f} ms")
    # End of Greedy Algorithm Core (Outputs token_allocation dict)


    # --- Original Postprocessing (start_time_7 onwards) ---
    # Using the token_allocation dict produced by the algorithm.
    # Reverting to the exact original code's logic and timing structure.

    torch.cuda.synchronize()
    start_time_7 = time.time() # Aligned with original time_spend 7


    # Clean token_allocation (Original time_spend 7 - start)
    cleaned_token_allocation = {}
    for gpu_id, experts_in_alloc in token_allocation.items():
        cleaned_experts = {expert_id: tokens_list for expert_id, tokens_list in experts_in_alloc.items() if tokens_list} # Check if list is non-empty
        if cleaned_experts:
            cleaned_token_allocation[gpu_id] = cleaned_experts

    # Initialize output dicts
    gpu_token_indices = {}
    gpu_hidden_states = {}
    gpu_topk = {}
    gpu_topk_weights = {}
    slice_mapping = {} # Identical to gpu_token_indices

    # Rebuild unused expert_token_map (Original time_spend 8 - start)
    expert_token_map = {} # (expert_id, token_idx) -> gpu_id map
    for gpu_id, experts_in_cleaned_alloc in cleaned_token_allocation.items():
        for expert_id, token_list in experts_in_cleaned_alloc.items():
             for idx in token_list: # idx is a (b, k) tuple
                 expert_token_map[(expert_id, idx[0])] = gpu_id # idx[0] is batch_idx

    # torch.cuda.synchronize()
    # start_time_8 = time.time() # Aligned with original time_spend 9

    # Collect unique GPU token indices and slice tensors (Original time_spend 9 - start)
    # Loop through all deployed GPUs
    for gpu_id in expert_deployment:
        # Only process if this GPU received tokens in the cleaned allocation
        if gpu_id not in cleaned_token_allocation:
            # Ensure empty entries for GPUs with no tokens, matching original output structure
            gpu_token_indices[gpu_id] = []
            slice_mapping[gpu_id] = []
            if hidden_states is not None:
                gpu_hidden_states[gpu_id] = None # Or empty tensor
            gpu_topk[gpu_id] = torch.empty(0, K, dtype=topk_ids.dtype, device=device)
            gpu_topk_weights[gpu_id] = torch.empty(0, K, dtype=topk_weights.dtype, device=device)
            continue # Skip processing for this empty GPU

        # Collect unique original batch indices (b) using Python set (Original time_spend 9 - start)
        token_idx_set = {idx[0] for expert_list in cleaned_token_allocation[gpu_id].values() for idx in expert_list}

        # Convert set to list and sort it (Original time_spend 10 - start)
        indices_list = sorted(list(token_idx_set))

        # Store the sorted unique batch indices
        gpu_token_indices[gpu_id] = indices_list
        slice_mapping[gpu_id] = indices_list # slice_mapping is identical

        # Process hidden states and topk by slicing using the Python list (Original time_spend 10 - cont.)
        if hidden_states is not None:
            # Check if list is not empty before slicing (already covered by cleaned_token_allocation check above)
            gpu_hidden_states[gpu_id] = hidden_states[indices_list]

        gpu_topk[gpu_id] = topk_ids[indices_list].clone()
        gpu_topk_weights[gpu_id] = topk_weights[indices_list].clone()

        # Original commented-out block for filtering topk/weights is skipped


    # Original postprocessing finishes the loop and exits
    torch.cuda.synchronize()
    start_time_9 = time.time() # Aligned with original time_spend 10
    print(f"time spend 9 (postproc-build_indices_slice): {(start_time_9 - start_time_7)*1000:.2f} ms")

    # 把延迟加起来
    total_time = (start_time_9 - start_time_7) * 1000 + (start_time_4_opt - start_time_0) * 1000 + (start_time_6 - start_time_5) * 1000
    print(f"total time: {total_time:.2f} ms")



    return gpu_hidden_states, gpu_topk, gpu_topk_weights, slice_mapping

# Example __main__ block (same as before)
if __name__ == "__main__":
    # topk_ids = torch.tensor([[58, 15,  7,  6],
    # [18, 58, 41, 59],
    # [ 0,  0,  0,  0],
    # [ 0,  0,  0,  0],
    # [ 0,  0,  0,  0],
    # [ 0,  0,  0,  0],
    # [ 0,  1,  2,  3],
    # [ 0,  1,  2,  3],
    # [ 0,  0,  0,  0],
    # [ 0,  0,  0,  0],
    # [ 0, 14, 19, 59],
    # [ 0,  0,  0,  0],
    # [ 0, 19, 59, 42],
    # [ 0,  0,  0,  0],
    # [ 0, 19, 52,  7],
    # [ 7,  0, 19,  3],
    # [ 0, 19,  5, 54]], device='cuda:0', dtype=torch.int32)

    # 生成topk_ids，大小 32*4，每个元素在0-59之间
    topk_ids = torch.randint(0, 60, (512, 4), device='cuda:0', dtype=torch.int32)

    topk_weights = torch.randn_like(topk_ids, dtype=torch.float32) # Use float for weights

    expert_deployment = {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25},
                            1: {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40},
                            2: {30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55},
                            3: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}}

    shape = topk_ids.shape
    hidden_states = torch.randn(shape[0], shape[1], 1024).cuda()

    def latency_function(total_tokens, num_experts):
        return 0.305597 + 0.011089 * num_experts + 0.001311 * total_tokens + 0.000087 * num_experts**2 - 0.000004 * total_tokens**2 + 0.000065 * num_experts * total_tokens

    print("--- Running User's Hybrid Version (Optimized Preproc + Original Algo + Original Postproc) ---")
    for i in range(10):
        print(f"== i: {i}")
        gpu_hidden_states, gpu_topk, gpu_topk_weights, slice_mapping = greedy_expert_scheduling_user_hybrid(
            topk_ids, topk_weights, expert_deployment, f=latency_function, hidden_states=hidden_states
        )