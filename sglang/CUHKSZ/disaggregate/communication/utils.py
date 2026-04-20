from gc import enable
import torch
import torch.distributed as dist
from typing import Dict, List, Any


def get_ep_group_info(tp_size: int, ep_size: int, moe_node_num: int, enable_ep_intra_node_reduce: bool) -> Dict[str, Any]:
    """计算EP组信息，包括每个组的rank列表和发送策略
    
    Args:
        tp_size: Tensor parallel size (attention ranks)
        ep_size: Expert parallel size (MoE ranks)
        moe_node_num: Number of MoE nodes
        enable_ep_intra_node_reduce: Whether to enable EP intra-node reduce
        
    Returns:
        dict: EP group information including groups, ranks, and send strategy
    """
    world_size = tp_size + ep_size
    
    ep_ranks = list(range(tp_size, world_size))
    att_ranks = list(range(tp_size))
    
    # 获取所有ep group 的 rank
    ranks_per_node = ep_size // moe_node_num
    ep_groups = []
    
    if ranks_per_node <= 0:
        ep_groups = [ep_ranks]
    else:
        for i in range(moe_node_num):
            start_idx = i * ranks_per_node
            end_idx = start_idx + ranks_per_node
            if i == moe_node_num - 1:
                end_idx = ep_size
            group_ranks = ep_ranks[start_idx:end_idx]
            ep_groups.append(group_ranks)        
    
    # MOE -> Attention ：每个attention rank应该从哪些EP rank接收
    send_strategy = {}
    if enable_ep_intra_node_reduce and len(ep_groups) > 0:
        # 每个attention rank从每个EP组中选择一个EP rank
        for att_rank in att_ranks:
            send_strategy[att_rank] = []
            for ep_group in ep_groups:
                if len(ep_group) > 0:
                    # 使用轮询方式选择EP rank
                    selected_ep_rank = ep_group[att_rank % len(ep_group)]
                    send_strategy[att_rank].append(selected_ep_rank)
    else:
        # 传统方式：每个attention rank从所有EP rank接收
        for att_rank in att_ranks:
            send_strategy[att_rank] = ep_ranks
    
    info = {
        'ep_groups': ep_groups,
        'ep_ranks': ep_ranks,
        'att_ranks': att_ranks,
        'send_strategy': send_strategy,
        'enable_ep_intra_node_reduce': enable_ep_intra_node_reduce,
        'moe_node_num': moe_node_num,
        'tp_size': tp_size,
        'ep_size': ep_size
    }
    
    print(f"[EP Group Info] tp_size={tp_size}, ep_size={ep_size}, moe_node_num={moe_node_num}")
    print(f"[EP Group Info] ep_groups={ep_groups}")
    print(f"[EP Group Info] send_strategy (att <- moe) ={send_strategy}")
    
    return info

if __name__ == "__main__":
    tp_size = 1
    ep_size = 1
    moe_node_num = 1
    # enable_ep_intra_node_reduce = True
    enable_ep_intra_node_reduce = False
    info = get_ep_group_info(tp_size, ep_size, moe_node_num, enable_ep_intra_node_reduce)
    # print(info)