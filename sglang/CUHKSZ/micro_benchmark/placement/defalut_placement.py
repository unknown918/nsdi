import math
import os
import json
import threading
import time
import numpy as np
from typing import Dict, List
from sglang.srt.utils import get_available_gpu_memory
from sglang.srt.configs.model_config import ModelConfig

class SlotManager:              
    def __init__(self, model_config, ep_size, ep_node_num, ep_node_id):
        
        self.model_config = model_config
        
        self.gpu_ids = list(range(ep_size))
        self.parallel_size = ep_size
        self.gpu_slots = {}       

        self.num_hidden_layers = int(os.environ["NUM_HIDDEN_LAYERS"]) if "NUM_HIDDEN_LAYERS" in os.environ else model_config.hf_config.num_hidden_layers    
        
        # self.expertNum_pLayer = self.model_config.hf_config.num_experts # FIXME
        # 如果num_experts存在，则使用num_experts，否则使用n_routed_experts
        if hasattr(self.model_config.hf_config, 'num_experts'):
            self.expertNum_pLayer = self.model_config.hf_config.num_experts
        else:
            self.expertNum_pLayer = self.model_config.hf_config.n_routed_experts

        self.expertNum_pLayer_pGPU = self.expertNum_pLayer // self.parallel_size
        self.expertSize_in_mb = self.calculate_expert_size()
        
        # 初始化专家映射状态
        self.expert_mapping_ready = False
        self.expert_mapping_data = {}
        
        # 计算专家分布
        self.calculate_slots()

        for key, value in self.gpu_slots.items():
            print(f"-----------------------------------")
            print(f"gpu_id: {key}")
            print(f"total_redundant_slots: {value['total_redundant_slots']}")
            print(f"redundant_slots_layer_wise: {value['redundant_slots_layer_wise']}")
            print(f"redundant_slots_expert_wise: {value['redundant_slots_expert_wise']}")
            print(f"route_experts: {{'experts_per_layer': {dict((layer, [experts[0], experts[-1]]) for layer, experts in value['route_experts']['experts_per_layer'].items())}, 'layers': {value['route_experts']['layers']}}}")
            print(f"redundant_experts: {dict((layer, [experts[0], experts[-1]] if experts else []) for layer, experts in value['redundant_experts'].items())}")

        # 准备专家映射数据
        self.prepare_expert_mapping()
        
#         ut Token (excl. 1st token)------
# Mean TPOT (ms):                          32.69     
# Median TPOT (ms):                        34.26     
# P99 TPOT (ms):                           83.99 
    def prepare_expert_mapping(self):
        """准备专家映射数据"""
        mapping = {}
        
        for gpu_id, gpu_info in self.gpu_slots.items():
            route_experts = {
                str(layer): [int(e) for e in experts] 
                for layer, experts in gpu_info["route_experts"]["experts_per_layer"].items()
            }
            
            redundant_experts = {
                str(layer): [int(e) for e in experts]
                for layer, experts in gpu_info["redundant_experts"].items()
            }
            
            mapping[str(gpu_id)] = {
                "route_experts": route_experts,
                "redundant_experts": redundant_experts
            }
        
        self.expert_mapping_data = mapping
        self.expert_mapping_ready = True
        print("Expert mapping data prepared")
        # print(f"expert_mapping_data: {self.expert_mapping_data}")
    
    def get_expert_mapping(self):
        """获取专家映射数据"""
        return self.expert_mapping_data

    def calculate_expert_size(self):
                    
        hidden_size = self.model_config.hf_config.hidden_size
        moe_intermediate_size = self.model_config.hf_config.moe_intermediate_size
        # num_experts = self.model_config.hf_config.num_experts
        if hasattr(self.model_config.hf_config, 'num_experts'):
            num_experts = self.model_config.hf_config.num_experts
        else:
            num_experts = self.model_config.hf_config.n_routed_experts
            
        print(f"hidden_size: {hidden_size}, moe_intermediate_size: {moe_intermediate_size}, num_experts: {num_experts}")
        
        params_per_expert = hidden_size * moe_intermediate_size * 3           
        params_per_expert = params_per_expert * int(os.getenv("MOCK_EXPERT_SIZE_TIMES")) # TODO just for simulation
        
        # support each parameter is float16 (2 bytes)
        expert_size_bytes = params_per_expert * 2
        expert_size_mb = expert_size_bytes / (1024 * 1024)
        
        return expert_size_mb            
        
    def calculate_slots(self):
        
        # assume all gpu have the same memory! local slot manager only need to know the total memory of one gpu
        total_memory = get_available_gpu_memory("cuda", 0) 
        print(f"total_memory: {total_memory}")
        print(f"memory in gb = {total_memory / 1024}")
        for gpu_id in self.gpu_ids:                                                          
            # convert total_memory from GB to MB
            total_memory_mb = total_memory * 1024
            
            # each ep instance's task
            expertSize_pLayer_pGPU_in_mb = self.expertSize_in_mb * self.expertNum_pLayer_pGPU
            expertSize_pGPU_in_mb = expertSize_pLayer_pGPU_in_mb * self.num_hidden_layers
            
            available_memory = total_memory_mb - int(os.getenv("RESERVE_TENSOR_BUFFER"))
            redundant_memory = available_memory - expertSize_pGPU_in_mb
            print(f"--------------------------------------------")
            print(f"gpu_id: {gpu_id}")
            print(f"total_memory_mb: {total_memory_mb}")
            print(f"expertSize_in_mb: {self.expertSize_in_mb}")
            print(f"expertNum_pLayer_pGPU: {self.expertNum_pLayer_pGPU}")
            print(f"expertSize_pLayer_pGPU_in_mb: {expertSize_pLayer_pGPU_in_mb}")
            print(f"expertSize_pGPU_in_mb: {expertSize_pGPU_in_mb}")
            print(f"available_memory: {available_memory}")  
            print(f"redundant_memory: {redundant_memory}")

            if redundant_memory < 0:
                raise ValueError(f"Not enough memory for GPU {gpu_id}")
            
            redundant_slots = math.floor(redundant_memory / self.expertSize_in_mb)
            redundant_slots_layer_wise = math.floor(redundant_slots / self.num_hidden_layers)
            redundant_slots_expert_wise = redundant_slots - redundant_slots_layer_wise * self.num_hidden_layers                    
            
            print(f"redundant_slots: {redundant_slots}")
            print(f"redundant_slots_layer_wise: {redundant_slots_layer_wise}")
            print(f"redundant_slots_expert_wise: {redundant_slots_expert_wise}")
                                                  
            self.gpu_slots[gpu_id] = {
                "total_redundant_slots": redundant_slots_layer_wise * self.num_hidden_layers,
                "redundant_slots_layer_wise": redundant_slots_layer_wise,
                "redundant_slots_expert_wise": 0,
                "route_experts": self.assign_experts_to_gpu(gpu_id),
                "redundant_experts": self.assign_redundant_experts_to_gpu(gpu_id, redundant_slots_layer_wise, redundant_slots_expert_wise)
            }
            
            # print(f"--------***********--------")
            # for key, value in self.gpu_slots.items():
            #     print(f"gpu_id: {key}")
            #     print(f"total_redundant_slots: {value['total_redundant_slots']}")
            #     print(f"redundant_slots_layer_wise: {value['redundant_slots_layer_wise']}")
            #     print(f"redundant_slots_expert_wise: {value['redundant_slots_expert_wise']}")
            #     print(f"route_experts: {value['route_experts']}")
            #     print(f"redundant_experts: {value['redundant_experts']}")

    def assign_experts_to_gpu(self, gpu_id):
        start_expert_id = gpu_id * self.expertNum_pLayer_pGPU
        end_expert_id = start_expert_id + self.expertNum_pLayer_pGPU
        
        expert_ids = list(range(start_expert_id, end_expert_id))
        experts_per_layer = {layer: expert_ids for layer in range(self.num_hidden_layers)}
        
        return {
            "experts_per_layer": experts_per_layer,
            "layers": self.num_hidden_layers
        }

    def assign_redundant_experts_to_gpu(self, gpu_id, redundant_slots_layer_wise, redundant_slots_expert_wise):        
        redundant_experts_per_layer = {layer: [] for layer in range(self.num_hidden_layers)}
        
        if redundant_slots_layer_wise >= 1:
            redundant_experts_start_id = (self.expertNum_pLayer_pGPU * (gpu_id + 1)) % self.expertNum_pLayer
            
            for layer in range(self.num_hidden_layers):                
                # 确保每个expert id都在合法范围内
                experts = []
                for i in range(redundant_slots_layer_wise):
                    expert_id = (redundant_experts_start_id + i) % self.expertNum_pLayer
                    experts.append(expert_id)
                redundant_experts_per_layer[layer] = experts
       
        # If there are leftover redundant slots that cannot be evenly placed across all layers,
        # do not assign them. This ensures each layer has the same number of redundant experts.
        if redundant_slots_expert_wise >= 1:
            pass

        return redundant_experts_per_layer

    def get_gpu_slots(self, gpu_id=None):
        if gpu_id is not None:
            return self.gpu_slots.get(gpu_id)
        return self.gpu_slots

    def export_layer0_summary(self, output_path):
        """Export per-GPU layer-0 routing and redundant expert start/end to a txt file.
        If output_path is a directory, create a timestamped txt file within it.
        Otherwise, treat it as the exact file path.
        """
        lines = []
        for gpu_id, gpu_info in self.gpu_slots.items():
            route_layer0 = gpu_info["route_experts"]["experts_per_layer"][0]
            route_start = int(route_layer0[0]) if len(route_layer0) > 0 else None
            route_end = int(route_layer0[-1]) if len(route_layer0) > 0 else None
            route_count = len(route_layer0)

            redundant_layer0 = gpu_info["redundant_experts"][0]
            redundant_start = int(redundant_layer0[0]) if len(redundant_layer0) > 0 else None
            redundant_end = int(redundant_layer0[-1]) if len(redundant_layer0) > 0 else None
            redundant_count = len(redundant_layer0)

            lines.append(
                f"gpu_id: {gpu_id}, layer: 0, route_experts: [{route_start}, {route_end}] (count={route_count}), "
                f"redundant_experts: [{redundant_start}, {redundant_end}] (count={redundant_count})"
            )

        file_path = output_path
        # If given path is a directory, write a timestamped file within it
        if os.path.isdir(output_path):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(output_path, f"placement_results_{timestamp}.txt")
        else:
            # Ensure parent directory exists when a file path is provided
            parent_dir = os.path.dirname(output_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

        with open(file_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote layer 0 placement summary to {file_path}")

if __name__ == "__main__":
    
    os.environ["MOCK_EXPERT_SIZE_TIMES"] = "1"
    os.environ["RESERVE_TENSOR_BUFFER"] = "512"
    # os.environ["NUM_HIDDEN_LAYERS"] = "32"
    
    model_config = ModelConfig(
            # model_path="/home/zhexiangz/projec_storage/model_weights/hub/models--deepseek-ai--DeepSeek-V2/snapshots/4461458f186c35188585855f28f77af5661ad489",
            model_path="/home/zhexiangz/projec_storage/model_weights/hub/R1",
            model_override_args="{}"
        )
    
    # Iterate ep_size from 6 to 16, step by 2, and export results to separate logs
    results_dir = "/home/zhexiangz/prototype/janus/CUHKSZ/micro_benchmark/scheduling/placement_results"
    for ep in range(18, 21, 2):
        print(f"\n===== Running placement with ep_size={ep} =====")
        slot_manager = SlotManager(model_config, ep_size=ep, ep_node_num=1, ep_node_id=0)
        out_file = os.path.join(results_dir, f"v2_ep_{ep}.txt")
        slot_manager.export_layer0_summary(out_file)