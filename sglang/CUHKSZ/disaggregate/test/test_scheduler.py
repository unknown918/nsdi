#!/usr/bin/env python3
"""
测试程序 for scheduler.py
测试 greedy_expert_scheduling 和 restore_hidden_states 函数
"""

import time
import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

# 导入scheduler模块
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sglang.CUHKSZ.disaggregate.scheduler.schedulerEPLB import random_expert_scheduling, greedy_expert_scheduling_split

@dataclass
class ExpertLocationMetadata:
    """EPLB专家位置元数据，类似SGLang的实现"""
    physical_to_logical_map: torch.Tensor  # (layers, num_physical_experts)
    logical_to_all_physical_map: torch.Tensor  # (layers, num_logical_experts, max_replicas)
    logical_to_all_physical_map_num_valid: torch.Tensor  # (layers, num_logical_experts)
    logical_to_rank_dispatch_physical_map: torch.Tensor  # (layers, num_logical_experts)
    logical_to_physical_by_gpu_map: Dict[tuple, int]  # {(logical_id, gpu_id): physical_id}
    
    # 额外信息
    num_physical_experts: int
    num_logical_experts: int
    num_gpus: int
    
    @property
    def num_layers(self) -> int:
        return self.physical_to_logical_map.shape[0]
    
    def logical_to_all_physical(self, layer_id: int, logical_expert_id: int) -> List[int]:
        """获取指定逻辑专家的所有物理专家ID"""
        return [
            physical_expert_id
            for physical_expert_id in self.logical_to_all_physical_map[
                layer_id, logical_expert_id
            ].tolist()
            if physical_expert_id != -1
        ]

def build_eplb_structures(base_experts: Dict[int, List[int]], device: torch.device, num_layers: int = 1):
    """
    构建EPLB的四个核心数据结构
    
    Args:
        base_experts: 每个GPU上的基础专家分配 {gpu_id: [expert_ids]}
        device: 设备（GPU或CPU）
        num_layers: MoE层数，默认为1
    
    Returns:
        ExpertLocationMetadata: 包含四个核心数据结构的类实例
    """
    
    print("\n" + "="*50)
    print("构建EPLB数据结构")
    print("="*50)
        
    num_gpus = len(base_experts)
    num_logical_experts = max(max(experts) for experts in base_experts.values()) + 1
    num_physical_experts = sum(len(experts) for experts in base_experts.values())
    
    print(f"  num_gpus: {num_gpus}")
    print(f"  num_logical_experts: {num_logical_experts}")
    print(f"  num_physical_experts: {num_physical_experts}")
    print(f"  device: {device}")
    
    # 1. physical_to_logical_map: (num_layers, num_physical_experts)
    # 物理专家ID到逻辑专家ID的映射
    physical_to_logical_map = torch.zeros((num_layers, num_physical_experts), dtype=torch.int64, device=device)
    
    # 2. logical_to_all_physical_map: (num_layers, num_logical_experts, max_replicas)
    # 逻辑专家ID到所有物理专家ID的映射
    max_replicas = max(len(experts) for experts in base_experts.values())
    logical_to_all_physical_map = torch.full(
        (num_layers, num_logical_experts, max_replicas), 
        -1, dtype=torch.int64, device=device
    )
    
    # 3. logical_to_all_physical_map_num_valid: (num_layers, num_logical_experts)
    # 每个逻辑专家有多少个物理副本
    logical_to_all_physical_map_num_valid = torch.zeros(
        (num_layers, num_logical_experts), dtype=torch.int64, device=device
    )
    
    # 4. logical_to_rank_dispatch_physical_map: (num_layers, num_logical_experts)
    # 静态调度时，每个逻辑专家应该调度到哪个物理专家
    logical_to_rank_dispatch_physical_map = torch.full(
        (num_layers, num_logical_experts), -1, dtype=torch.int64, device=device
    )
    
    # 构建映射关系
    physical_expert_id = 0
    logical_to_physical_count = {}  # 记录每个逻辑专家有多少个物理副本
    logical_to_physical_by_gpu_map = {}  # 新增：记录(logical_id, gpu_id) -> physical_id的映射
    
    for gpu_id in range(num_gpus):
        experts_on_gpu = base_experts[gpu_id]
        
        for local_expert_id, logical_expert_id in enumerate(experts_on_gpu):
            # 更新 physical_to_logical_map
            physical_to_logical_map[0, physical_expert_id] = logical_expert_id
            
            # 更新 logical_to_all_physical_map
            if logical_expert_id not in logical_to_physical_count:
                logical_to_physical_count[logical_expert_id] = 0
            
            replica_idx = logical_to_physical_count[logical_expert_id]
            logical_to_all_physical_map[0, logical_expert_id, replica_idx] = physical_expert_id
            logical_to_physical_count[logical_expert_id] += 1
            
            # 更新 logical_to_all_physical_map_num_valid
            logical_to_all_physical_map_num_valid[0, logical_expert_id] = logical_to_physical_count[logical_expert_id]
            
            # 更新 logical_to_rank_dispatch_physical_map (选择第一个物理副本)
            if logical_to_rank_dispatch_physical_map[0, logical_expert_id] == -1:
                logical_to_rank_dispatch_physical_map[0, logical_expert_id] = physical_expert_id
                
            # 新增：更新logical_to_physical_by_gpu_map
            logical_to_physical_by_gpu_map[(logical_expert_id, gpu_id)] = physical_expert_id
            
            physical_expert_id += 1
    
    print(f"  physical_to_logical_map: {physical_to_logical_map}")
    # 形状: (num_layers, num_physical_experts)
    # 作用: 物理专家ID到逻辑专家ID的映射
    # physical_to_logical_map: torch.Tensor

    # 例子:
    # 物理专家ID: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # 逻辑专家ID: [1, 3, 7, 0, 4, 5, 6, 2, 1, 0, 3, 6]
    print(f" \n logical_to_all_physical_map {logical_to_all_physical_map}")
    # 打印每一个logical对应的 py
    for i in range(logical_to_all_physical_map.shape[0]):
        print(f"  layer {i}:")
        for j in range(logical_to_all_physical_map.shape[1]):
            print(f"    logical {j}: {logical_to_all_physical_map[i, j]}")
    
    # 形状: (num_layers, num_logical_experts, max_replicas)
    # 作用: 逻辑专家ID到所有物理专家ID的映射（支持冗余）
    # logical_to_all_physical_map: torch.Tensor

    # 例子:
    # 逻辑专家0 -> 物理专家[0, 8]  # 0是原版，8是冗余
    # 逻辑专家1 -> 物理专家[1, 9]  # 1是原版，9是冗余
    # 逻辑专家2 -> 物理专家[7, 10] # 7是原版，10是冗余
    print(f"  logical_to_all_physical_map_num_valid {logical_to_all_physical_map_num_valid}")
    # 形状: (num_layers, num_logical_experts)
    # 作用: 每个逻辑专家有多少个物理副本
    # logical_to_all_physical_map_num_valid: torch.Tensor

    # 例子:
    # 逻辑专家0: 2个副本（物理专家0和8）
    # 逻辑专家1: 2个副本（物理专家1和9）
    # 逻辑专家2: 2个副本（物理专家7和10）
    # print(f"  logical_to_rank_dispatch_physical_map shape: {logical_to_rank_dispatch_physical_map}")
    # 形状: (num_layers, num_logical_experts)
    # 作用: 静态调度时，每个逻辑专家应该调度到哪个物理专家
    # logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    
    return ExpertLocationMetadata(
        physical_to_logical_map=physical_to_logical_map,
        logical_to_all_physical_map=logical_to_all_physical_map,
        logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
        logical_to_rank_dispatch_physical_map=logical_to_rank_dispatch_physical_map,
        logical_to_physical_by_gpu_map=logical_to_physical_by_gpu_map,  # 新增
        num_physical_experts=num_physical_experts,
        num_logical_experts=num_logical_experts,
        num_gpus=num_gpus
    )


def test_random_expert_scheduling(topk_ids, eplb_structures, layer_id, batch, topk):
        
    # 测试random_expert_scheduling
    print("\n" + "="*50)
    print("测试random_expert_scheduling")
    print("="*50)
    print(f"原始topk_ids:\n{topk_ids}")
    
    # 在函数外部进行层选择和数据访问
    logical_to_all_physical_map = eplb_structures.logical_to_all_physical_map[layer_id]  # (num_logical_experts, max_replicas)
    logical_to_all_physical_map_num_valid = eplb_structures.logical_to_all_physical_map_num_valid[layer_id]  # (num_logical_experts)
    
    # 调用random_expert_scheduling
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    new_topk_ids = random_expert_scheduling(
        topk_ids=topk_ids,
        logical_to_all_physical_map=logical_to_all_physical_map,
        logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid
    )
    end_event.record()
    torch.cuda.synchronize()
    print(f"random_expert_scheduling 时间: {start_event.elapsed_time(end_event)} 毫秒")
    print(f"转换后的topk_ids:\n{new_topk_ids}")
    # 创建延迟函数 f(tokens, experts) -> latency

def test_greedy_expert_scheduling_split(topk_ids, topk_weights, expert_deployment, hidden_states, eplb_structures):
    """测试 greedy_expert_scheduling 函数"""
    print("=" * 50)
    print("测试 greedy_expert_scheduling 函数")
    print("=" * 50)
    
    def latency_function(tokens, experts):
        """简单的延迟函数: tokens * 0.1 + experts * 0.5"""
        return tokens * 0.1 + experts * 0.5
    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    token_allocation = greedy_expert_scheduling_split(
        topk_ids=topk_ids,
        expert_deployment=expert_deployment,
        f=latency_function,
        eplb_structures=eplb_structures
    )
    end_event.record()
    torch.cuda.synchronize()
    print(f"greedy_expert_scheduling 时间: {start_event.elapsed_time(end_event)} 毫秒")
    
    
    print(f"原始topk_ids (logical):\n{topk_ids}")
    print(f"转换后的topk_ids (physical):\n{token_allocation}")
    
    # return new_topk_ids

 

def build_data(B=256, K=6, num_experts=160, gpu_nums=8, redundancy_count=4, device=torch.device("cuda:6" if torch.cuda.is_available() else "cpu")):
    
    # 创建测试数据
    # B, K = 256, 6  # batch_size=4, top_k=2
    # num_experts = 160
    
    # 创建 topk_ids: 每个token选择的前K个专家ID
    # 形状: (B, K) = (4, 2)
    # topk_ids = torch.randint(0, num_experts, (B, K), device=device, dtype=torch.int64)
    
    topk_ids = torch.tensor([[ 0,  6,  1],
            [ 6,  0,  7],
            [ 5,  8,  5],
            [11, 3, 10],
            [ 8,  4,  1]], device='cuda:6')
    # topk_ids = torch.randint(0, 12, (512, 6), device='cuda:6')
    print(f"topk_size: {topk_ids.shape}")
    print(f"topk_ids:\n{topk_ids}")
    topk_weights = torch.rand(B, K, device=device, dtype=torch.float32)
    # print(f"topk_weights:\n{topk_weights}")
    
    
        # 创建 hidden_states (可选)
    hidden_size = 2048
    hidden_states = torch.randn(B, hidden_size, device=device, dtype=torch.float32)
    
    # 创建 expert_deployment: 每个GPU上部署的专家
    # 定义冗余专家个数
    # redundancy_count = 4
    
    # 基础专家分配 - 简化版本
    experts_per_gpu = num_experts // gpu_nums  # 每个GPU的基础专家数量
    base_experts = {}
    for gpu_id in range(gpu_nums):
        start_expert = gpu_id * experts_per_gpu
        end_expert = start_expert + experts_per_gpu
        base_experts[gpu_id] = list(range(start_expert, end_expert))
    
    # 添加冗余专家
    expert_deployment = {}
    for gpu_id in range(gpu_nums):
        # 复制基础专家
        expert_deployment[gpu_id] = base_experts[gpu_id].copy()
        
        # 添加冗余专家：从下一个GPU取前redundancy_count个专家
        next_gpu = (gpu_id + 1) % gpu_nums
        redundant_experts = base_experts[next_gpu][:redundancy_count]
        expert_deployment[gpu_id].extend(redundant_experts)
    
    print(f"\n 冗余专家个数: {redundancy_count}")
    print(f"每个GPU的专家分配:")
    for gpu_id, experts in expert_deployment.items():
        print(f"  GPU {gpu_id}: {experts}")
    
    
    print(f"\n 输入数据:")
    print(f"  topk_ids shape: {topk_ids.shape}")
    print(f"  topk_weights shape: {topk_weights.shape}")
    print(f"  hidden_states shape: {hidden_states.shape}")
    # print(f"  expert_deployment: {expert_deployment}")
    # print(f"  topk_ids:\n{topk_ids}")
    # print(f"  topk_weights:\n{topk_weights}")
        
    return topk_ids, topk_weights, expert_deployment, hidden_states

def main():
    
    # 设置设备
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # batch_size=1024
    # topk = 6
    # num_experts= 160
    # gpu_nums= 8
    # redundancy_count= 40
    
    batch_size=16
    topk = 6
    num_experts= 12
    gpu_nums= 4
    redundancy_count= 2
    
    topk_ids, topk_weights, expert_deployment, hidden_states= build_data(B=batch_size, K=topk, num_experts=num_experts, gpu_nums=gpu_nums, redundancy_count=redundancy_count, device=device)
    
    # 构建EPLB数据结构
    eplb_structures = build_eplb_structures(expert_deployment, device)    
    
    test_random_expert_scheduling(topk_ids, eplb_structures, layer_id=0, batch=256, topk=6)
    test_greedy_expert_scheduling_split(topk_ids, topk_weights, expert_deployment, hidden_states, eplb_structures)
    



if __name__ == "__main__":
    main() 