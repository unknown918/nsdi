import torch
import time
import called_experts
import os
from config.topk_real_trace_128 import topk_id
from config.ep_placement import *
import triton
from moeLayers.ep_moe_kernel import EPMoE
# from ep_simulator2 import run_simulator
# PYTHONPATH=$PYTHONPATH:/home/zhexiangz/prototype/janus/CUHKSZ/disaggregate python3  /home/zhexiangz/prototype/janus/CUHKSZ/micro_benchmark/scheduling/ablation_ep_latency/pipeline/pipeline_greepy.py
gpu_used = 6
total_expert_per_gpu = 29 # fix as it follow placement

EXPERT_RANGES = None
gpu_config = None
if gpu_used == 6:
    gpu_config = gpu_config_6
    EXPERT_RANGES = EXPERT_RANGES_6
elif gpu_used == 8:
    gpu_config = gpu_config_8
    EXPERT_RANGES = EXPERT_RANGES_8
elif gpu_used == 10:
    gpu_config = gpu_config_10
    EXPERT_RANGES = EXPERT_RANGES_10
elif gpu_used == 12:
    gpu_config = gpu_config_12
    EXPERT_RANGES = EXPERT_RANGES_12
elif gpu_used == 14:
    gpu_config = gpu_config_14
    EXPERT_RANGES = EXPERT_RANGES_14
elif gpu_used == 16:
    gpu_config = gpu_config_16
    EXPERT_RANGES = EXPERT_RANGES_16

EXPERT_RANGE_NAMES = [f"{s}-{e}" for s, e in EXPERT_RANGES] 

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


def random_schedule(topk_id, map_tensor, copy_count):
    """
    把逻辑 expert ID 映射到物理 expert ID，冗余时随机选择
    
    Args:
        topk_id: (batch, k) 逻辑id tensor
        map_tensor: (num_logical, max_copies) 映射表
        copy_count: (num_logical,) 每个逻辑id的副本数量
    
    Returns:
        physical_id: (batch, k) 物理id tensor
    """
    # 获取对应的副本数量
    counts = copy_count[topk_id]  # shape = (batch, k)
    
    # 为每个位置随机生成 [0, count) 的索引
    rand_idx = torch.randint(0, counts.max().item(), size=topk_id.shape, device=topk_id.device)
    rand_idx = torch.minimum(rand_idx, counts - 1)  # 避免超范围
    
    # gather 出物理 ID
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

def save_tensor_to_file(tensor, filepath):
    """
    将tensor保存到文件
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 清空文件内容
    with open(filepath, 'w') as f:
        pass
    
    # 保存tensor数据
    torch.save(tensor, filepath)
    # print(f"Saved tensor to {filepath}")

def load_tensor_from_file(filepath):
    """
    从文件读取tensor
    """
    if os.path.exists(filepath):
        tensor = torch.load(filepath, map_location='cuda', weights_only=True)
        print(f"Loaded tensor from {filepath}")
        return tensor
    else:
        print(f"File {filepath} does not exist")
        return None


# ----------------------------模拟计算性能----------------------------

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["dummy"],               # x轴占位
        x_vals=[0],                      # 固定一个点
        line_arg="expert_range",         # 每条线是一个区间
        line_vals=EXPERT_RANGES,
        line_names=EXPERT_RANGE_NAMES,
        ylabel="Time (ms)",
        plot_name="epmoe-latency",                # 不画图
        args={},
    )
)
def benchmark(dummy, expert_range, num_experts, hidden_size,
              intermediate_size, dtype, topk_id, rep=20):

    start_expert_id, end_expert_id = expert_range
    # print(f"Benchmarking experts {start_expert_id}-{end_expert_id}")

    torch.set_default_device("cuda")
    torch.cuda.manual_seed_all(0)

    batch = topk_id.shape[0]
    hidden_states = torch.randn(batch, hidden_size, dtype=dtype, device="cuda")
    topk_weights = torch.ones(batch, topk_id.shape[1], device="cuda", dtype=dtype)

    model = EPMoE(
        num_experts=num_experts,
        top_k=topk_id.shape[1],
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        params_dtype=dtype,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
    ).cuda()

    # warmup
    for _ in range(5):
        _ = model(hidden_states, topk_id, topk_weights)
    torch.cuda.synchronize()

    # timing
    iters_ms = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for _ in range(rep):
        start_event.record()
        _ = model(hidden_states, topk_id, topk_weights)
        end_event.record()
        torch.cuda.synchronize()
        iters_ms.append(round(start_event.elapsed_time(end_event), 2))

    return sum(iters_ms) / len(iters_ms), min(iters_ms), max(iters_ms)

def run_simulator(topk_id, num_experts=290, hidden_size=5120, intermediate_size=1536,
                  dtype=torch.bfloat16, rep=20, print_data=True, show_plots=False):
    return benchmark.run(
        print_data=print_data,
        show_plots=show_plots,
        save_path=None,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dtype=dtype,
        topk_id=topk_id,
        rep=rep,
    )
    

    
# ======================
# 示例用法
# ======================
if __name__ == "__main__":
    
    # 模拟 topk_id    
    topk = 6
    num_experts = 160

    
    gpu_num = len(gpu_config)    
    # print("gpu_config:\n", gpu_config)
    print("gpu_num: ", gpu_num)
    print("topk_id shape: ", topk_id.shape)
        
    # 构建映射 logical id - physical id
    l2p, p2l, gidx = build_mapping_from_config(gpu_config)    
    map_tensor, copy_count = prepare_mapping_tensor(l2p, device="cuda")        
    expert_gpu_to_phys = build_expert_gpu_to_phys(p2l, gidx, num_experts)
    expert_to_gpus = build_expert_to_gpus(p2l, gidx)        
    
    # print("Logical → Physical:")
    # for k, v in l2p.items():
    #     print(f"Logical {k} → Physical {v}")
        
    # print("Physical → Logical:")
    # for k, v in p2l.items():
    #     print(f"Physical {k} → Logical {v}")            

    topk_id = topk_id.to(torch.int32)                                       
    
    # 路径相对于本文件，确保写入到 pipeline/tmp_mapped_topk 下
    base_dir = os.path.dirname(__file__)
    greedy_filepath = os.path.join(base_dir, "tmp_mapped_topk/greedy.pt")
    random_filepath = os.path.join(base_dir, "tmp_mapped_topk/random.pt")

    # greedy schedule
    assign, load, mapped_topk = greedy_schedule_fast(topk_id, expert_to_gpus, expert_gpu_to_phys)        
    # print("greedy schedule mapped physical IDs:\n", mapped_topk)
    load, gpu_experts = calculate_gpu_load(mapped_topk, gidx, gpu_num)
    print("gpu load summery [greedy schedule]:\n", load)
    
    # 保存greedy schedule结果
    save_tensor_to_file(mapped_topk, greedy_filepath)

    # 在生成 greedy 结果后，调用模拟器性能测试
    try:
        print("\nRunning EP simulator on greedy mapped_topk...")
        dtype = torch.bfloat16
        num_experts_inclue_redundant = total_expert_per_gpu * gpu_used
        print(f"num_experts: {num_experts_inclue_redundant}")
        _ = run_simulator(
            topk_id=mapped_topk,
            num_experts=num_experts_inclue_redundant,
            hidden_size=5120,
            intermediate_size=1536,
            dtype=torch.bfloat16,
            rep=20,
            print_data=True,
            show_plots=False,
        )
    except Exception as e:
        print(f"[warn] EP simulator failed: {e}")
            
    
    # random schedule
    # physical_id = random_schedule(topk_id, map_tensor, copy_count)
    # # print("random schedule mapped physical IDs:\n", physical_id)
    # load, gpu_experts = calculate_gpu_load(physical_id, gidx, gpu_num)
    # print("gpu load summery [random schedule]:\n", load)
    
    # # 保存random schedule结果
    # save_tensor_to_file(physical_id, random_filepath)
    
    
    
    
    
# PYTHONPATH=$PYTHONPATH:/home/zhexiangz/prototype/janus/CUHKSZ/disaggregate python3  /home/zhexiangz/prototype/janus/CUHKSZ/micro_benchmark/scheduling/ablation_ep_latency/pipeline/pipeline_greepy.py