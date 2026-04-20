import torch

def build_dense_mapping_from_dicts(expert_to_gpus, expert_gpu_to_phys, num_experts, device="cuda"):
    """
    根据已有的 expert_to_gpus 和 expert_gpu_to_phys 直接生成稠密映射张量

    Args:
        expert_to_gpus (dict): logical_id -> set(gpu_ids)
        expert_gpu_to_phys (dict): logical_id -> {gpu_id: physical_id}
        num_experts (int): 逻辑专家数量
        num_gpus (int): GPU 数量
        device (str): 张量所在设备

    Returns:
        expert2gpus: (num_experts, max_copies) 每个逻辑专家在不同副本上的 GPU id
        expert2phys: (num_experts, max_copies) 每个逻辑专家在不同副本上的物理 expert id
        copy_count: (num_experts,) 每个逻辑专家的副本数量
        max_copies: int，每个专家的最大副本数
    """
    max_copies = max(len(gpus) for gpus in expert_to_gpus.values())
    expert2gpus = torch.full((num_experts, max_copies), -1, dtype=torch.int32, device=device)
    expert2phys = torch.full((num_experts, max_copies), -1, dtype=torch.int32, device=device)
    copy_count = torch.zeros(num_experts, dtype=torch.int32, device=device)

    for lid, gpus in expert_to_gpus.items():
        for j, g in enumerate(sorted(gpus)):  # 保证顺序确定
            expert2gpus[lid, j] = g
            expert2phys[lid, j] = expert_gpu_to_phys[lid][g]
        copy_count[lid] = len(gpus)

    return expert2gpus, expert2phys, copy_count, max_copies