from dataclasses import dataclass
from typing import Dict, List
import torch

@dataclass
class ExpertLocationMetadata:
    """EPLB专家位置元数据，每一层存储一个实例"""
    physical_to_logical_map: torch.Tensor  # (num_physical_experts,)
    logical_to_all_physical_map: torch.Tensor  # (num_logical_experts, max_replicas)
    logical_to_all_physical_map_num_valid: torch.Tensor  # (num_logical_experts,)
    logical_to_rank_dispatch_physical_map: torch.Tensor  # (num_logical_experts,)
    logical_to_physical_by_gpu_map: Dict[tuple, int]  # {(logical_id, gpu_id): physical_id}
    
    # 额外信息
    num_physical_experts: int
    num_logical_experts: int
    num_gpus: int
    
    def logical_to_all_physical(self, logical_expert_id: int) -> List[int]:
        """获取指定逻辑专家的所有物理专家ID"""
        return [
            physical_expert_id
            for physical_expert_id in self.logical_to_all_physical_map[logical_expert_id].tolist()
            if physical_expert_id != -1
        ]
