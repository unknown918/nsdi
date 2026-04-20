import numpy as np
import time
import torch
from typing import Dict, List
def greedy_expert_scheduling(topk_ids, topk_weights, expert_deployment, f=None, hidden_states=None):


    # 将topk_ids转换为expert_tokens格式
    torch.cuda.synchronize()
    start_time_0 = time.time()
    
    # 完全在GPU上高效构建专家计数和索引
    B, K = topk_ids.shape
    device = topk_ids.device
    
    # 构造token的位置信息：[[b0, k0], [b0, k1], ..., [bN, kK]]
    batch_indices = torch.arange(B, device=device).repeat_interleave(K)
    k_indices = torch.arange(K, device=device).repeat(B)
    token_positions = torch.stack([batch_indices, k_indices], dim=1)  # shape (B*K, 2)
    
    # 拉平topk_ids，形状：(B*K,)
    flat_topk_ids = topk_ids.reshape(-1)
    
    # cuda sync
    torch.cuda.synchronize()
    start_time_1 = time.time()

    # 假设 flat_topk_ids 是已经排序的
    sorted_flat_topk_ids, _ = torch.sort(flat_topk_ids)
    unique_experts, inverse_indices, counts = torch.unique_consecutive(sorted_flat_topk_ids, return_inverse=True, return_counts=True)
    torch.cuda.synchronize()
    start_time_2 = time.time()

    # 构建expert_count_dict：专家ID -> token数量
    expert_count_dict = dict(zip(unique_experts.tolist(), counts.tolist()))
    
    torch.cuda.synchronize()
    start_time_3 = time.time()

    # 构建expert_token_indices：专家ID -> [(b, k), ...]
    expert_token_indices = {int(expert.item()): [] for expert in unique_experts}
    for idx, expert_idx in enumerate(inverse_indices):
        expert_id = int(unique_experts[expert_idx].item())
        pos = token_positions[idx]
        expert_token_indices[expert_id].append((int(pos[0].item()), int(pos[1].item())))
    
    expert_token_dict = list(expert_count_dict.items())
    
    torch.cuda.synchronize()
    start_time_4 = time.time()
    print(f"time spend 1: {(start_time_1 - start_time_0)*1000:.2f} ms")
    print(f"time spend 2: {(start_time_2 - start_time_1)*1000:.2f} ms")
    print(f"time spend 3: {(start_time_3 - start_time_2)*1000:.2f} ms")
    print(f"time spend 4: {(start_time_4 - start_time_3)*1000:.2f} ms")

    expert_tokens = {expert_id: count for expert_id, count in expert_token_dict}
    

    # 计时    
    torch.cuda.synchronize()
    start_time_5 = time.time()

    # 预先计算所有GPU上的专家集合，避免重复计算    
    experts = set()
    gpu_expert_lists = {}  # 缓存已计算的专家部署信息
    
    for gpu_id, gpu_experts in expert_deployment.items():
        experts.update(gpu_experts)
        gpu_expert_lists[gpu_id] = list(gpu_experts)
    
    # Initialize GPU loads and expert counts
    gpu_tokens = {gpu_id: 0 for gpu_id in expert_deployment}
    gpu_experts = {gpu_id: 0 for gpu_id in expert_deployment}
    allocation = {}
    
    # 初始化新的token_allocation结构
    token_allocation = {gpu_id: {} for gpu_id in expert_deployment}
    
    # Sort experts by token count (descending)
    sorted_experts = sorted(experts, key=lambda x: expert_tokens.get(x, 0), reverse=True)
    
    # First phase: Assign non-redundant experts
    for expert_id in sorted_experts:
        # 获取专家部署的GPU列表
        deployed_gpus = [gpu_id for gpu_id, gpu_experts in expert_deployment.items() if expert_id in gpu_experts]
        
        if len(deployed_gpus) == 1:
            # Only one option - must assign all tokens here
            gpu_id = deployed_gpus[0]
            tokens = expert_tokens.get(expert_id, 0)
            allocation[(expert_id, gpu_id)] = tokens
            gpu_tokens[gpu_id] += tokens
            gpu_experts[gpu_id] += 1
            # 分配所有token索引
            token_allocation[gpu_id][expert_id] = expert_token_indices.get(expert_id, [])
    
    # Second phase: Assign redundant experts
    for expert_id in sorted_experts:
        deployed_gpus = [gpu_id for gpu_id, gpu_experts in expert_deployment.items() if expert_id in gpu_experts]
        
        if len(deployed_gpus) > 1:
            tokens = expert_tokens.get(expert_id, 0)
            token_indices = expert_token_indices.get(expert_id, [])
            
            # 直接计算所有GPU的当前延迟，避免重复计算
            current_latencies = {}
            for gpu_id in deployed_gpus:
                expert_count = gpu_experts[gpu_id]
                new_expert_count = expert_count + (1 if (expert_id, gpu_id) not in allocation else 0)
                current_latencies[gpu_id] = f(gpu_tokens[gpu_id] + tokens, new_expert_count)
            
            min_gpu_id = min(current_latencies, key=current_latencies.get)
            min_latency = current_latencies[min_gpu_id]
            
            # 计算其他GPU的延迟
            other_latencies = [f(gpu_tokens[gpu_id], gpu_experts[gpu_id]) 
                              for gpu_id in expert_deployment if gpu_id not in deployed_gpus]
            if not other_latencies:
                other_latencies = [f(gpu_tokens[gpu_id], gpu_experts[gpu_id]) for gpu_id in expert_deployment]
            
            second_min_latency = min(other_latencies) if other_latencies else float('inf')
            
            # If full allocation creates significant imbalance, split the load
            imbalance_threshold = 1.2  # Configurable threshold
            if min_latency > second_min_latency * imbalance_threshold and len(deployed_gpus) > 1:
                # Split tokens based on current load
                remaining_tokens = tokens
                remaining_indices = token_indices.copy()
                gpu_latencies = [(gpu_id, f(gpu_tokens[gpu_id], gpu_experts[gpu_id])) 
                                 for gpu_id in deployed_gpus]
                
                # Sort GPUs by current latency
                gpu_latencies.sort(key=lambda x: x[1])
                
                for gpu_id, _ in gpu_latencies:
                    # Skip last GPU - remaining tokens go there
                    if gpu_id == gpu_latencies[-1][0] or remaining_tokens == 0:
                        continue
                    
                    # Calculate how many tokens to allocate to this GPU
                    # We want to bring its latency closer to the next GPU's latency
                    next_gpu_idx = next((i for i, (g, _) in enumerate(gpu_latencies) if g != gpu_id), 0)
                    next_gpu_id, next_latency = gpu_latencies[next_gpu_idx]
                    
                    # Binary search to find optimal token allocation for this GPU
                    low, high = 0, remaining_tokens
                    best_allocation = 0
                    
                    while low <= high:
                        mid = (low + high) // 2
                        new_expert_count = gpu_experts[gpu_id] + 1 if (expert_id, gpu_id) not in allocation else gpu_experts[gpu_id]
                        x = gpu_tokens[gpu_id] + mid
                        new_latency = f(x, new_expert_count)
                        
                        if new_latency < next_latency:
                            best_allocation = mid
                            low = mid + 1
                        else:
                            high = mid - 1
                    
                    # Allocate tokens to this GPU
                    if best_allocation > 0:
                        allocation[(expert_id, gpu_id)] = best_allocation
                        # 分配具体的token索引
                        selected_indices = remaining_indices[:best_allocation]
                        if expert_id not in token_allocation[gpu_id]:
                            token_allocation[gpu_id][expert_id] = []
                        token_allocation[gpu_id][expert_id].extend(selected_indices)
                        
                        remaining_indices = remaining_indices[best_allocation:]
                        
                        gpu_tokens[gpu_id] += best_allocation
                        if (expert_id, gpu_id) not in allocation or allocation[(expert_id, gpu_id)] == 0:
                            gpu_experts[gpu_id] += 1
                        remaining_tokens -= best_allocation
                
                # Allocate remaining tokens to the last GPU
                if remaining_tokens > 0:
                    last_gpu_id = gpu_latencies[-1][0]
                    if (expert_id, last_gpu_id) not in allocation:
                        allocation[(expert_id, last_gpu_id)] = remaining_tokens
                        if expert_id not in token_allocation[last_gpu_id]:
                            token_allocation[last_gpu_id][expert_id] = []
                        token_allocation[last_gpu_id][expert_id].extend(remaining_indices)
                        gpu_experts[last_gpu_id] += 1
                    else:
                        allocation[(expert_id, last_gpu_id)] += remaining_tokens
                        if expert_id not in token_allocation[last_gpu_id]:
                            token_allocation[last_gpu_id][expert_id] = []
                        token_allocation[last_gpu_id][expert_id].extend(remaining_indices)
                    gpu_tokens[last_gpu_id] += remaining_tokens
            else:
                # Full allocation to the GPU with minimum latency
                allocation[(expert_id, min_gpu_id)] = tokens
                if expert_id not in token_allocation[min_gpu_id]:
                    token_allocation[min_gpu_id][expert_id] = []
                token_allocation[min_gpu_id][expert_id].extend(token_indices)
                
                gpu_tokens[min_gpu_id] += tokens
                if (expert_id, min_gpu_id) not in allocation or allocation[(expert_id, min_gpu_id)] == 0:
                    gpu_experts[min_gpu_id] += 1
    
    torch.cuda.synchronize()
    start_time_6 = time.time()
    print(f"time spend Algo: {(start_time_6 - start_time_5)*1000:.2f} ms")

    # Calculate final latencies
    gpu_latencies = [f(gpu_tokens[j], gpu_experts[j]) for j in expert_deployment]    
    
    # Print the allocation and per-GPU latency
    # print("\nExpert allocation and per-GPU latency:")
    for gpu_id in expert_deployment:
        # print(f"\nGPU {gpu_id+1} (deployed experts: {sorted(expert_deployment[gpu_id])}):")
        tokens_on_gpu = 0
        experts_with_tokens = 0
        
        for expert_id in sorted(experts):
            if expert_id in expert_deployment[gpu_id] and (expert_id, gpu_id) in allocation:
                tokens = allocation[(expert_id, gpu_id)]
                if tokens > 0:
                    tokens_on_gpu += tokens
                    experts_with_tokens += 1                                    
    
    # print("-"*100)
    
    # 清理token_allocation，移除没有分配token的专家
    cleaned_token_allocation = {}
    for gpu_id, experts in token_allocation.items():
        cleaned_experts = {expert_id: tokens for expert_id, tokens in experts.items() if tokens}
        if cleaned_experts:
            cleaned_token_allocation[gpu_id] = cleaned_experts
        
    gpu_token_indices = {}
    gpu_hidden_states = {}
    gpu_topk = {}
    gpu_topk_weights = {}
    slice_mapping = {}

    # torch.cuda.synchronize()
    # start_time_8 = time.time()
    # 预计算专家分配映
    expert_token_map = {}  # (expert_id, token_idx) -> gpu_id 映射
    
    for gpu_id, experts in cleaned_token_allocation.items():
        for expert_id, token_indices in experts.items():
            for idx in token_indices:
                expert_token_map[(expert_id, idx[0])] = gpu_id
    
    # torch.cuda.synchronize()
    # start_time_9 = time.time()
    # 一次性收集每个GPU的token索引
    for gpu_id in expert_deployment:
        if gpu_id not in cleaned_token_allocation:
            continue
        
        # torch.cuda.synchronize()
        # start_time_a = time.time()
        token_idx_set = {idx[0] for expert_list in cleaned_token_allocation[gpu_id].values() for idx in expert_list}
        gpu_token_indices[gpu_id] = sorted(list(token_idx_set))
        slice_mapping[gpu_id] = gpu_token_indices[gpu_id]
        
        # torch.cuda.synchronize()
        # start_time_b = time.time()
        # 高效处理hidden states和topk
        if hidden_states is not None:
            gpu_hidden_states[gpu_id] = hidden_states[gpu_token_indices[gpu_id]]
            
        gpu_topk[gpu_id] = topk_ids[gpu_token_indices[gpu_id]].clone()
        gpu_topk_weights[gpu_id] = topk_weights[gpu_token_indices[gpu_id]].clone()


    torch.cuda.synchronize()   
    start_time_10 = time.time()
    print(f"time spend 10: Postprocessing {(start_time_10 - start_time_6)*1000:.2f} ms")


    return gpu_hidden_states, gpu_topk, gpu_topk_weights, slice_mapping

if __name__ == "__main__":
    topk_ids = torch.tensor([[58, 15,  7,  6],
    [18, 58, 41, 59],
    [ 0,  0,  0,  0],
    [ 0,  0,  0,  0],   
    [ 0,  0,  0,  0],
    [ 0,  0,  0,  0],   
    [ 0,  1,  2,  3],
    [ 0,  1,  2,  3],
    [ 0,  0,  0,  0],
    [ 0,  0,  0,  0],
    [ 0, 14, 19, 59],
    [ 0,  0,  0,  0],
    [ 0, 19, 59, 42],   
    [ 0,  0,  0,  0],
    [ 0, 19, 52,  7],
    [ 7,  0, 19,  3],
    [ 0, 19,  5, 54]], device='cuda:0', dtype=torch.int32)

    expert_deployment = {0: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}, 
                            1: {15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}, 
                            2: {30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55}, 
                            3: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59}}

    shape = topk_ids.shape
    hidden_states = torch.randn(shape[0], shape[1], 1024).cuda()

    def latency_function(total_tokens, num_experts):
        # Example latency function
        # return total_tokens * 0.1 + num_experts * 0.5
        return 0.305597 + 0.011089 * num_experts + 0.001311 * total_tokens + 0.000087 * num_experts**2 - 0.000004 * total_tokens**2 + 0.000065 * num_experts * total_tokens
    
    for i in range(10):
        print(f"== i: {i}")
        gpu_hidden_states, gpu_topk, gpu_topk_weights, slice_mapping = greedy_expert_scheduling(topk_ids, topk_ids, expert_deployment, f=latency_function, hidden_states=hidden_states)
