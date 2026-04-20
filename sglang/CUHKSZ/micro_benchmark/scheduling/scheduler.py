import torch
import time
import called_experts

def build_mapping_from_config(gpu_config):
    """
    根据用户提供的 GPU 配置生成 expert 映射关系
    
    Args:
        gpu_config (list[dict]): 每个元素是一个 GPU 的配置，例如：
            [
                {"route_experts": list, "redundant_experts": list},
                {"route_experts": list, "redundant_experts": list},
                ...
            ]
    
    Returns:
        logical_to_physical (dict): logical_id -> [physical_ids]
        physical_to_logical (dict): physical_id -> logical_id
        gpu_local_index (dict): physical_id -> (gpu_id, local_index)
    """
    logical_to_physical = {}
    physical_to_logical = {}
    gpu_local_index = {}

    global_id = 0
    for gpu_id, cfg in enumerate(gpu_config):
        # 拼接 route_experts + redundant_experts
        logical_list = cfg["route_experts"] + cfg["redundant_experts"]
        
        for local_idx, logical_id in enumerate(logical_list):
            logical_to_physical.setdefault(logical_id, []).append(global_id)
            physical_to_logical[global_id] = logical_id
            gpu_local_index[global_id] = (gpu_id, local_idx)
            global_id += 1

    return logical_to_physical, physical_to_logical, gpu_local_index


def prepare_mapping_tensor(logical_to_physical, device="cuda"):
    """
    把 dict 转成 tensor，方便快速采样
    
    Returns:
        map_tensor: (num_logical, max_copies) 填充后的映射表
        copy_count: (num_logical,) 每个逻辑id的副本数量
    """
    max_copies = max(len(v) for v in logical_to_physical.values())
    num_logical = max(logical_to_physical.keys()) + 1
    
    map_tensor = torch.full((num_logical, max_copies), -1, dtype=torch.int32)
    copy_count = torch.zeros(num_logical, dtype=torch.int32)
    
    for lid, pids in logical_to_physical.items():
        map_tensor[lid, :len(pids)] = torch.tensor(pids, dtype=torch.int32)
        copy_count[lid] = len(pids)
    
    return map_tensor.to(device), copy_count.to(device)

import numpy as np

def random_schedule(topk_id, map_tensor, copy_count):
    print("map_tensor:\n", map_tensor)
    print("copy_count:\n", copy_count)
    count_val = copy_count[topk_id].max().item()
    rand_np = np.random.randint(0, 65536, size=topk_id.shape, dtype=np.int64) % copy_count[topk_id].max().item()
    rand_idx = torch.from_numpy(rand_np).to(topk_id.device)
    rand_idx = torch.minimum(rand_idx, copy_count[topk_id] - 1)
    physical_id = map_tensor[topk_id, rand_idx]    
    return physical_id

def benchmark_mapping(topk_id, map_tensor, copy_count, warmup=10, iters=100):
    # warmup (预热，避免首次运行偏差)
    for _ in range(warmup):
        _ = random_schedule(topk_id, map_tensor, copy_count)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iters):
        _ = random_schedule(topk_id, map_tensor, copy_count)
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time_ms = (end - start) / iters * 1000
    print(f"Average mapping time: {avg_time_ms:.3f} ms for shape {topk_id.shape}")

def benchmark_get_call_expert(topk_id, map_tensor, copy_count, warmup=10, iters=100):
    for _ in range(warmup):
        torch.unique(topk_id)

    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(iters):
        _ = torch.unique(topk_id)
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time_ms = (end - start) / iters * 1000
    print(f"Average get_call_expert time: {avg_time_ms:.3f} ms for shape {topk_id.shape}")
    
def benchmark_get_call_expert2(topk_id, num_experts, warmup=10, iters=100):
    """
    性能测试：用布尔 mask 方式统计被调用的 expert
    """
    mask = torch.zeros(num_experts, dtype=torch.bool, device=topk_id.device)
    for _ in range(warmup):
        
        mask[topk_id] = True
        # _ = mask.nonzero(as_tuple=True)[0]

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iters):
        # mask = torch.zeros(num_experts, dtype=torch.bool, device=topk_id.device)
        mask[topk_id] = True
        # _ = mask.nonzero(as_tuple=True)[0]

    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / iters * 1000
    print(f"Average get_call_expert2 (mask) time: {avg_time_ms:.3f} ms for shape {topk_id.shape}")

def benchmark_get_call_expert3(topk_id, num_experts, warmup=10, iters=100):
    """
    性能测试：用自定义 CUDA kernel (called_experts) 统计被调用的 expert
    """
    # warmup
    for _ in range(warmup):
        _ = called_experts.called_experts(topk_id, num_experts)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iters):
        _ = called_experts.called_experts(topk_id, num_experts)

    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / iters * 1000
    print(f"Average get_call_expert3 (kernel) time: {avg_time_ms:.3f} ms for shape {topk_id.shape}")


def build_expert_to_gpus(physical_to_logical, gpu_local_index):
    expert_to_gpus = {}
    for pid, lid in physical_to_logical.items():
        g, _ = gpu_local_index[pid]
        expert_to_gpus.setdefault(lid, set()).add(g)
    return expert_to_gpus


def build_expert_gpu_to_phys(physical_to_logical, gpu_local_index, num_experts):
    """
    构建 (logical_id, gpu_id) -> physical_id 的查表
    """
    expert_gpu_to_phys = {lid: {} for lid in range(num_experts)}
    for pid, lid in physical_to_logical.items():
        g, _ = gpu_local_index[pid]
        expert_gpu_to_phys[lid][g] = pid
    return expert_gpu_to_phys


def greedy_schedule_fast(topk_id, expert_to_gpus, expert_gpu_to_phys, num_experts=160,):
    assign = {}
    # assign = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    # 在调度时直接 assign[e] = gpu_id

    load = {}     
    # mask = called_experts.called_experts(topk_id, num_experts)
    # called_experts_ids = mask.nonzero(as_tuple=True)[0].tolist()
    
    ids_ = called_experts.called_experts(topk_id, num_experts)
    called_experts_ids = ids_.tolist()
    
    # print("called_experts_ids:\n", called_experts_ids)
    # print("111")
    # Step 1: 无冗余专家固定分配
    for e in called_experts_ids:
        gpus = expert_to_gpus[e]
        if len(gpus) == 1:
            g = next(iter(gpus))
            assign[e] = g
            load[g] = load.get(g, 0) + 1

    # Step 2: 有冗余专家贪心分配
    for e in called_experts_ids:
        if e in assign:
            continue
        gpus = expert_to_gpus[e]
        g = min(gpus, key=lambda x: load.get(x, 0))
        assign[e] = g
        load[g] = load.get(g, 0) + 1

    # kernel不能处理dict之类的，我们的assin要不设置成两个list，然后放在kernel里面做？
    print("assign:\n", assign)
    # logical2phys = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    # lids, pids = [], []
    # for lid, g in assign.items():
    #     pids.append(expert_gpu_to_phys[lid][g])
    #     lids.append(lid)
    # logical2phys[torch.tensor(lids, device="cuda")] = torch.tensor(pids, device="cuda")
        
    logical2phys = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    lids, pids = [], []
    for lid, g in assign.items():
        pids.append(expert_gpu_to_phys[lid][g])
        lids.append(lid)

    logical2phys[torch.tensor(lids, dtype=torch.long, device="cuda")] = \
        torch.tensor(pids, dtype=torch.int32, device="cuda")

    mapped_topk = logical2phys[topk_id]

    return assign, load, mapped_topk


def greedy_schedule(topk_id, map_tensor, gpu_local_index, num_experts):
    """
    贪心调度算法（低延迟版本）：
    1. 用 CUDA kernel 快速获取被调用的 expert 集合
    2. 无冗余专家直接分配
    3. 有冗余专家分配到负载最小的 GPU
    
    Args:
        topk_id: (batch, k) 逻辑 expert id (int32)
        map_tensor: (num_logical, max_copies) 逻辑id -> 物理id表
        gpu_local_index: dict, physical_id -> (gpu_id, local_idx)
        num_experts: 总逻辑 expert 数量
    
    Returns:
        assign: dict, logical_id -> gpu_id
        load: dict, gpu_id -> expert_count
    """
    # Step 1: CUDA kernel 获取被调用的 experts mask
    mask = called_experts.called_experts(topk_id, num_experts)
    called_experts_ids = mask.nonzero(as_tuple=True)[0].tolist()

    print("called_experts_ids:\n", called_experts_ids)
    assign = {}
    load = {}

    # Step 2: 先分配无冗余专家
    for e in called_experts_ids:
        phys = map_tensor[e]
        phys = phys[phys != -1].tolist()
        gpus = list(set(gpu_local_index[p][0] for p in phys))
        if len(gpus) == 1:  # 无冗余
            g = gpus[0]
            assign[e] = g
            load[g] = load.get(g, 0) + 1

    # Step 3: 分配有冗余专家（贪心策略）
    for e in called_experts_ids:
        if e in assign:
            continue
        phys = map_tensor[e]
        phys = phys[phys != -1].tolist()
        gpus = list(set(gpu_local_index[p][0] for p in phys))
        # 贪心：选当前负载最小的 GPU
        g = min(gpus, key=lambda x: load.get(x, 0))
        assign[e] = g
        load[g] = load.get(g, 0) + 1

    # mapped_topk = map_tensor[topk_id]
    return assign, load


def calculate_gpu_load(physical_id, gpu_local_index, gpu_num):
    """
    给定已映射到物理 expert 的 `physical_id`，统计每个 GPU 上激活的 expert 数量。

    Args:
        physical_id (Tensor): (batch, k) 或 (N,) 的物理 expert id（int32/int64），可能包含重复
        gpu_local_index (dict): physical_id -> (gpu_id, local_index)
        gpu_num (int): GPU 数量

    Returns:
        tuple:
            - dict: gpu_id -> 激活的 expert 数量（去重后）
            - dict: gpu_id -> 激活的物理 expert 列表（去重后，按升序）
    """
    # 展平并过滤无效 id（如 -1）
    if physical_id.dim() > 1:
        phys_flat = physical_id.reshape(-1)
    else:
        phys_flat = physical_id

    if phys_flat.numel() == 0:
        empty_count = {g: 0 for g in range(gpu_num)}
        empty_list = {g: [] for g in range(gpu_num)}
        return empty_count, empty_list

    phys_flat = phys_flat[phys_flat >= 0]

    # 去重：统计被调用过的物理 expert
    active_phys = torch.unique(phys_flat).tolist()

    # 统计每个 GPU 上的激活 expert
    load = {g: 0 for g in range(gpu_num)}
    gpu_to_experts = {g: set() for g in range(gpu_num)}
    for pid in active_phys:
        gpu_id, _ = gpu_local_index[int(pid)]
        load[gpu_id] += 1
        gpu_to_experts[gpu_id].add(int(pid))

    gpu_to_experts_sorted = {g: sorted(list(s)) for g, s in gpu_to_experts.items()}

    return load, gpu_to_experts_sorted


def calculate_gpu_token_count(physical_id, gpu_local_index, gpu_num):
    """
    统计每个 GPU 上的 token 总量（不去重）。

    Args:
        physical_id (Tensor): (batch, k) 或 (N,) 的物理 expert id（int32/int64），可能包含重复
        gpu_local_index (dict): physical_id -> (gpu_id, local_index)
        gpu_num (int): GPU 数量

    Returns:
        dict: gpu_id -> token 数量
    """
    # 展平并过滤无效 id（如 -1）
    if physical_id.dim() > 1:
        phys_flat = physical_id.reshape(-1)
    else:
        phys_flat = physical_id

    if phys_flat.numel() == 0:
        return {g: 0 for g in range(gpu_num)}

    phys_flat = phys_flat[phys_flat >= 0]

    token_count = {g: 0 for g in range(gpu_num)}
    for pid in phys_flat.tolist():
        gpu_id, _ = gpu_local_index[int(pid)]
        token_count[gpu_id] += 1

    return token_count


def benchmark_greedy_schedule(topk_id, map_tensor, gpu_local_index, num_experts, warmup=10, iters=100):
    # warmup
    for _ in range(warmup):
        _ = greedy_schedule(topk_id, map_tensor, gpu_local_index, num_experts)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iters):
        _ = greedy_schedule(topk_id, map_tensor, gpu_local_index, num_experts)

    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / iters * 1000
    print(f"Average greedy_schedule time: {avg_time_ms:.3f} ms for shape {topk_id.shape}")

def benchmark_greedy_schedule_fast(topk_id, expert_to_gpus, expert_gpu_to_phys, warmup=10, iters=100):
    # warmup
    for _ in range(warmup):
        _ = greedy_schedule_fast(topk_id, expert_to_gpus, expert_gpu_to_phys)

    torch.cuda.synchronize()
    start = time.time()

    for _ in range(iters):
        _ = greedy_schedule_fast(topk_id, expert_to_gpus, expert_gpu_to_phys)

    torch.cuda.synchronize()
    end = time.time()

    avg_time_ms = (end - start) / iters * 1000
    print(f"Average greedy_schedule_fast time: {avg_time_ms:.3f} ms for shape {topk_id.shape}")
# ======================
# 示例用法
# ======================
if __name__ == "__main__":
    
    # 模拟 topk_id
    batch = 3
    topk = 6
    num_experts = 160
    # topk_id = torch.randint(0, num_experts, (batch, topk), device="cuda")
    
    topk_id = torch.tensor([[  2,   5,  14,  41,  91,  58],
        [ 32,  40,  45,  53, 149,  33],
        [ 40,  51, 125, 126, 150,  48],
        [ 26,  31,  92,  93, 153,  85],
        [ 12,  44,  46,  57, 147,  58],
        [ 13,  21,  30, 129, 133, 128],
        [ 20,  27,  52,  59,  62,  23],
        [  3,  13, 108, 119, 152,   0],
        [ 85,  92, 124, 139, 153,  89],
        [ 28,  31,  43,  56, 107, 104],
        [ 35,  36,  68,  73, 153,  25],
        [ 22, 104, 107, 108, 152, 100],
        [  1,   4,  60,  71, 119, 100],
        [  5,  14,  15,  37, 143, 142],
        [ 13,  14,  42,  55, 136,   2],
        [ 41,  51,  52, 121, 134, 136],
        [ 25,  39,  40,  45,  97,  84],
        [ 65,  74,  77, 156, 159,   6],
        [  7,  43,  48,  58, 140,  55],
        [ 45, 110, 112, 117, 133, 116],
        [ 55,  69, 106, 111, 118, 100],
        [  5,  13,  14,  37,  98,   2],
        [ 50,  86, 122, 130, 132, 121],
        [ 17, 131, 138, 152, 159,   1],
        [ 50,  86, 122, 130, 132, 121],
        [100, 104, 111, 126, 151, 138],
        [ 31,  71, 102, 111, 119, 100],
        [ 50,  86, 120, 122, 130, 132],
        [ 48,  51, 112, 118, 157, 159],
        [  5,  13,  14,  66,  91,  80],
        [ 13,  62,  90,  91,  95,   2],
        [ 40,  48, 103, 118, 140, 107],
        [ 36,  80,  82,  96, 113, 103],
        [ 50,  56,  76,  86,  96,  77],
        [  0,  42,  58, 114, 118, 102],
        [ 13,  19,  96, 143, 152,  99],
        [  3,  54,  55,  56,  70,  57],
        [  8,  24,  36,  64,  75,  26],
        [ 64,  75,  80,  82, 153,  89],
        [  1,  24,  25,  26, 146,  22],
        [ 35,  42,  53, 108, 115,  25],
        [ 43,  47,  58, 117, 158,  46],
        [ 31,  84,  89, 102, 114,  98],
        [ 93, 104, 121, 134, 136,  84],
        [ 50,  86, 122, 130, 132, 135],
        [ 71,  97, 102, 104, 119, 111],
        [ 26,  31, 111, 115, 129,  25],
        [ 81, 134, 136, 150, 153,  97],
        [  2,   9,  36, 103, 113,  16],
        [  8,  57, 145, 148, 151,  46],
        [  9,  11,  86,  96, 103, 102],
        [ 84,  97, 107, 115, 151,  89],
        [ 10,  47,  49, 103, 105, 115],
        [ 24,  25,  26,  54,  84,  44],
        [ 48,  51,  56, 121, 134, 157],
        [  7,  26,  31,  84,  93,   1],
        [ 42,  45,  74, 141, 149,  58],
        [ 13, 127, 133, 145, 158,  17],
        [  9,  11,  50, 122, 130, 135],
        [ 40,  52,  59,  67, 109,  79],
        [ 37,  62,  64, 124, 132,  21],
        [  2,   5,  14,  91, 127,  13],
        [  7,  12,  23, 110, 117,   4],
        [  3,  13, 104, 107, 108,  54],
        [  1,   2,  23,  27,  66,  16],
        [ 40,  58,  71, 142, 143, 156],
        [  2,   5,  14, 126, 127,  37],
        [  6, 100, 112, 141, 156, 152],
        [  8,   9,  12, 127, 155, 122],
        [ 64,  76, 102, 103, 139, 130],
        [ 61,  63,  72,  76,  96, 137],
        [ 26,  97, 100, 104,  25,  85],
        [  5,  13,  14,  20, 158,  37],
        [  0,  24,  25,  31,  92,  16],
        [ 50,  80, 121, 130, 132, 139],
        [ 29,  34,  39,  43, 104,  22],
        [ 24,  26,  71,  87,  95,  32],
        [ 13,  32,  39, 107, 109, 111],
        [ 24,  36,  77,  80,  82,  64],
        [ 47,  51, 121, 134, 137, 126],
        [ 22,  31,  36,  58, 153, 146],
        [  0,   2,  13,  27,  91,  10],
        [  5,  13,  14,  58,  91,   2],
        [ 50,  86, 122, 130, 132,  80],
        [ 54, 108, 115, 150, 158,  58],
        [  6,  17,  66, 149, 151, 147],
        [  6, 112, 151, 152, 154, 104],
        [ 75,  90, 104, 112, 114, 111],
        [ 89,  94, 116, 126, 129, 138],
        [ 13,  27,  91,  94,  95,  90],
        [103, 107, 133, 140, 154, 139],
        [  7,  15,  71, 117,   3,  70],
        [ 50,  96, 121, 130, 132, 122],
        [  4,   9,  11,  44, 155,  42],
        [ 50, 130, 132, 139, 152, 126],
        [ 20,  22, 102, 104, 120, 108],
        [  8,  75, 151, 155, 156,  64],
        [ 42,  80,  96, 103, 113,  82],
        [ 32,  39,  70,  79, 153,  75],
        [ 98, 126, 135, 148, 153,  87],
        [ 45,  83,  99, 146, 148, 158],
        [  5,  14,  19,  83,  91,  20],
        [ 38,  73, 124, 126, 128,  79],
        [  2,  13,  15,  31, 127, 124],
        [ 84, 108, 117, 140, 154, 147],
        [ 80,  82,  96, 103, 113,  43],
        [  8,  13,  58, 127, 128,   7],
        [  2,  13,  27, 120, 133, 127],
        [ 24, 121, 126, 136, 137,  95],
        [  0,  62, 101, 109, 119, 111],
        [ 16,  47,  52,  67,  79,  65],
        [ 11,  80,  82, 103, 113,  97],
        [ 27,  28,  55, 108, 116, 105],
        [ 25,  27,  91,  94, 151, 147],
        [ 41,  53,  58, 101, 149,  49]], device='cuda', dtype=torch.int32)
    print("\nTopk logical IDs:\n", topk_id.shape)
    
    
    
    # gpu_config = [
    #     {"route_experts": list(range(0, 20)), "redundant_experts": list(range(20, 30))},  # GPU0
    #     {"route_experts": list(range(20, 40)), "redundant_experts": list(range(40, 50))}, # GPU1
    #     {"route_experts": list(range(40, 60)), "redundant_experts": list(range(60, 70))}, # GPU2
    #     {"route_experts": list(range(60, 80)), "redundant_experts": list(range(0, 10))},  # GPU3
    # ]
    
    # gpu_config = [
    #     {"route_experts": list(range(0, 20)), "redundant_experts": list(range(20, 29))},  # GPU0
    #     {"route_experts": list(range(20, 40)), "redundant_experts": list(range(40, 49))}, # GPU1
    #     {"route_experts": list(range(40, 60)), "redundant_experts": list(range(60, 69))}, # GPU2
    #     {"route_experts": list(range(60, 80)), "redundant_experts": list(range(80, 89))},  # GPU3
    #     {"route_experts": list(range(80, 100)), "redundant_experts": list(range(100, 109))},  # GPU4
    #     {"route_experts": list(range(100, 120)), "redundant_experts": list(range(120, 129))},  # GPU5
    #     {"route_experts": list(range(120, 140)), "redundant_experts": list(range(140, 149))},  # GPU6
    #     {"route_experts": list(range(140, 160)), "redundant_experts": list(range(0, 9))},  # GPU7        
    # ]
    
    gpu_config = [
        {"route_experts": list(range(0, 10)), "redundant_experts": list(range(10, 12))},  # GPU0
        {"route_experts": list(range(10, 20)), "redundant_experts": list(range(20, 22))}, # GPU1
        {"route_experts": list(range(20, 30)), "redundant_experts": list(range(30, 32))}, # GPU2
        {"route_experts": list(range(30, 40)), "redundant_experts": list(range(40, 42))},  # GPU3
        {"route_experts": list(range(40, 50)), "redundant_experts": list(range(50, 52))},  # GPU4
        {"route_experts": list(range(50, 60)), "redundant_experts": list(range(60, 62))},  # GPU5
        {"route_experts": list(range(60, 70)), "redundant_experts": list(range(70, 72))},  # GPU6
        {"route_experts": list(range(70, 80)), "redundant_experts": list(range(80, 82))},  # GPU7        
        {"route_experts": list(range(80, 90)), "redundant_experts": list(range(90, 92))},  # GPU8
        {"route_experts": list(range(90, 100)), "redundant_experts": list(range(100, 102))},  # GPU9
        {"route_experts": list(range(100, 110)), "redundant_experts": list(range(110, 112))},  # GPU10
        {"route_experts": list(range(110, 120)), "redundant_experts": list(range(120, 122))},  # GPU11
        {"route_experts": list(range(120, 130)), "redundant_experts": list(range(130, 132))},  # GPU12
        {"route_experts": list(range(130, 140)), "redundant_experts": list(range(140, 142))},  # GPU13
        {"route_experts": list(range(140, 150)), "redundant_experts": list(range(150, 152))},  # GPU14
        {"route_experts": list(range(150, 160)), "redundant_experts": list(range(0, 2))},  # GPU15
    ]
    # PYTHONPATH=$PYTHONPATH:/home/zhexiangz/prototype/janus/CUHKSZ/disaggregate python3 /home/zhexiangz/prototype/janus/CUHKSZ/micro_benchmark/scheduling/scheduler.py
    # gpu_config = [
    #     {"route_experts": list(range(0, 16)), "redundant_experts": list(range(16, 29))},  # GPU0
    #     {"route_experts": list(range(16, 32)), "redundant_experts": list(range(32, 45))}, # GPU1
    #     {"route_experts": list(range(32, 48)), "redundant_experts": list(range(48, 61))}, # GPU2
    #     {"route_experts": list(range(48, 64)), "redundant_experts": list(range(64, 77))}, # GPU3
    #     {"route_experts": list(range(64, 80)), "redundant_experts": list(range(80, 93))}, # GPU4
    #     {"route_experts": list(range(80, 96)), "redundant_experts": list(range(96, 109))}, # GPU5
    #     {"route_experts": list(range(96, 112)), "redundant_experts": list(range(112, 125))}, # GPU6
    #     {"route_experts": list(range(112, 128)), "redundant_experts": list(range(128, 141))}, # GPU7
    #     {"route_experts": list(range(128, 144)), "redundant_experts": list(range(144, 157))}, # GPU8
    #     {"route_experts": list(range(144, 160)), "redundant_experts": list(range(0, 13))},  # GPU9
    # ]
    # gpu_id: 0, layer: 0, route_experts: [0, 15] (count=16), redundant_experts: [16, 28] (count=13)
    # gpu_id: 1, layer: 0, route_experts: [16, 31] (count=16), redundant_experts: [32, 44] (count=13)
    # gpu_id: 2, layer: 0, route_experts: [32, 47] (count=16), redundant_experts: [48, 60] (count=13)
    # gpu_id: 3, layer: 0, route_experts: [48, 63] (count=16), redundant_experts: [64, 76] (count=13)
    # gpu_id: 4, layer: 0, route_experts: [64, 79] (count=16), redundant_experts: [80, 92] (count=13)
    # gpu_id: 5, layer: 0, route_experts: [80, 95] (count=16), redundant_experts: [96, 108] (count=13)
    # gpu_id: 6, layer: 0, route_experts: [96, 111] (count=16), redundant_experts: [112, 124] (count=13)
    # gpu_id: 7, layer: 0, route_experts: [112, 127] (count=16), redundant_experts: [128, 140] (count=13)
    # gpu_id: 8, layer: 0, route_experts: [128, 143] (count=16), redundant_experts: [144, 156] (count=13)
    # gpu_id: 9, layer: 0, route_experts: [144, 159] (count=16), redundant_experts: [0, 12] (count=13)

    gpu_num = len(gpu_config)    
    
    # 构建映射 logical id - physical id
    l2p, p2l, gidx = build_mapping_from_config(gpu_config)
    # 转换成 tensor
    map_tensor, copy_count = prepare_mapping_tensor(l2p, device="cuda")
        
    expert_gpu_to_phys = build_expert_gpu_to_phys(p2l, gidx, num_experts)
    expert_to_gpus = build_expert_to_gpus(p2l, gidx)        
    
    # print("Logical → Physical:")
    # for k, v in l2p.items():
    #     print(f"Logical {k} → Physical {v}")
        
    # print("Physical → Logical:")
    # for k, v in p2l.items():
    #     print(f"Physical {k} → Logical {v}")            

    # 测试pre-processing 性能
    benchmark_mapping(topk_id, map_tensor, copy_count)
    
    # benchmark_get_call_expert(topk_id, map_tensor, copy_count)
    # benchmark_get_call_expert2(topk_id, num_experts=80)
    topk_id = topk_id.to(torch.int32)   
    # benchmark_get_call_expert3(topk_id, num_experts=80)          # kernel 版
                    
    
    # --------------------------------------------------------------------------------------------------------------------------
    
    # without schedule
    # 直接把topkid转换成physical id        
        
    # random schedule
    physical_id = random_schedule(topk_id, map_tensor, copy_count)
    # print("random schedule mapped physical IDs:\n", physical_id)
    # load, gpu_experts = calculate_gpu_load(physical_id, gidx, gpu_num)
    # print("gpu load summery [random schedule]:\n", load)
    # print("gpu experts [random schedule]:\n", gpu_experts)
    # print("random schedule mapped physical IDs:\n", physical_id)
    # print("random schedule load:\n", load)
    # # 新增：统计 token 总量
    # token_count = calculate_gpu_token_count(physical_id, gidx, gpu_num)
    # sum_token_count = sum(token_count.values())
    # print("random schedule token count per GPU:\n", token_count)
    # print("random schedule sum token count:\n", sum_token_count)
    
    
    
        
    # print("*"*100)    
    # topk_id = topk_id.to(torch.int32)
    # print(f"topk_id={topk_id}")    
    # print("-"*100)    
    # assign, load = greedy_schedule(topk_id, map_tensor, gidx, num_experts=num_experts)
    # print(assign)
    # print()
    # print(load)
    # # benchmark_greedy_schedule(topk_id, map_tensor, gidx, num_experts=80)
    
    # print("-"*100)
    benchmark_greedy_schedule_fast(topk_id, expert_to_gpus, expert_gpu_to_phys)        
    
    
    # assign, load, mapped_topk = greedy_schedule_fast(topk_id, expert_to_gpus, expert_gpu_to_phys)
    # print(assign)
    # print()
    # print(load)
    # print("greedy schedule mapped physical IDs:\n", mapped_topk)
    
    # load, gpu_experts = calculate_gpu_load(mapped_topk, gidx, gpu_num)
    # print("gpu load summery [greedy schedule]:\n", load)
    # print("gpu experts [greedy schedule]:\n", gpu_experts)
    
    
    # physical_id = random_schedule(topk_id, map_tensor, copy_count)
    # print("random schedule mapped physical IDs:\n", physical_id)
    # load, gpu_experts = calculate_gpu_load(physical_id, gidx, gpu_num)
    # print("gpu load summery [random schedule]:\n", load)
    # print("gpu experts [random schedule]:\n", gpu_experts)
    
    
    
    # 新增：统计 token 总量
    # token_count = calculate_gpu_token_count(mapped_topk, gidx, gpu_num)
    # sum_token_count = sum(token_count.values())
    # print("greedy schedule token count per GPU:\n", token_count)
    # print("greedy schedule sum token count:\n", sum_token_count)
    
