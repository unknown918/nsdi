import torch
import os
from sglang.CUHKSZ.disaggregate.utils.eplb_metadata import ExpertLocationMetadata

# CUDA_VISIBLE_DEVICES=1 python3 /home/moe/amdissagCore/testmapping.py

def create_mock_mapping():
    """Create a mock mapping structure for testing"""
    return {
        "mapping": {
            "0": {
                "route_experts": {
                    "0": [0, 1, 2],                
                },
                "redundant_experts": {
                    "0": [3, 4],                    
                }
            },
            "1": {
                "route_experts": {
                    "0": [3, 4, 5],                                       
                },
                "redundant_experts": {
                    "0": [6, 7],                                     
                }
            },
            "2": {
                "route_experts": {
                    "0": [6, 7, 8],                                       
                },
                "redundant_experts": {
                    "0": [9, 10],                                     
                }
            },
            "3": {
                "route_experts": {
                    "0": [9, 10, 11],                                       
                },
                "redundant_experts": {
                    "0": [0, 1],                                     
                }
            }
        }
    }

def test_expert_mapping():
    # Set environment variables
    os.environ["MOE_WORKERS_COUNT"] = "4"  # 2 GPUs

    # Initialize the data structures that _process_expert_mapping would use
    expert_placement = []
    expert_placement_per_layer = {}
    expert_location_eplbmetadata = {}

    # Get mock mapping data
    mapping_data = create_mock_mapping()
    mapping = mapping_data.get("mapping", {})
    gpu_nums = int(os.getenv("MOE_WORKERS_COUNT"))

    # Initialize expert_placement list
    for _ in range(gpu_nums):
        expert_placement.append({})

    # Process the mapping data
    for gpu_id, gpu_info in mapping.items():
        gpu_id_int = int(gpu_id)
        route_experts = gpu_info.get("route_experts", {})
        redundant_experts = gpu_info.get("redundant_experts", {})

        # Process each layer
        for layer_str in set(route_experts.keys()) | set(redundant_experts.keys()):
            # Update expert_placement
            if layer_str not in expert_placement[gpu_id_int]:
                expert_placement[gpu_id_int][layer_str] = set()
            
            # Update expert_placement_per_layer
            layer_idx = int(layer_str)
            if layer_idx not in expert_placement_per_layer:
                expert_placement_per_layer[layer_idx] = {}
            if gpu_id_int not in expert_placement_per_layer[layer_idx]:
                expert_placement_per_layer[layer_idx][gpu_id_int] = set()
            
            # Add route_experts
            if layer_str in route_experts:
                expert_ids = set(map(int, route_experts[layer_str]))
                expert_placement[gpu_id_int][layer_str].update(expert_ids)
                expert_placement_per_layer[layer_idx][gpu_id_int].update(expert_ids)

            # Add redundant_experts
            if layer_str in redundant_experts:
                expert_ids = set(map(int, redundant_experts[layer_str]))
                expert_placement[gpu_id_int][layer_str].update(expert_ids)
                expert_placement_per_layer[layer_idx][gpu_id_int].update(expert_ids)

    # Build EPLB metadata for each layer
    for layer_idx, gpu_experts in expert_placement_per_layer.items():
        # Calculate total experts for this layer
        all_experts = set()
        for experts in gpu_experts.values():
            all_experts.update(experts)
        num_logical_experts = max(all_experts) + 1
        num_physical_experts = sum(len(experts) for experts in gpu_experts.values())
        
        # Get current device
        device = "cuda:0"  # For testing purposes
        
        # Initialize mapping tensors
        physical_to_logical_map = torch.zeros(num_physical_experts, dtype=torch.int64, device=device)
        max_replicas = max(len(experts) for experts in gpu_experts.values())
        max_replicas = max(max_replicas, 4)
        logical_to_all_physical_map = torch.full(
            size=(num_logical_experts, max_replicas),  # 修改：使用size参数
            fill_value=-1,  # 修改：使用fill_value参数
            dtype=torch.int64,
            device=device
        )
        logical_to_all_physical_map_num_valid = torch.zeros(
            num_logical_experts, dtype=torch.int64, device=device
        )
        logical_to_rank_dispatch_physical_map = torch.full(
            size=(num_logical_experts,),  # 修改：使用size参数，注意要加逗号表示是元组
            fill_value=-1,  # 修改：使用fill_value参数
            dtype=torch.int64,
            device=device
        )
        logical_to_physical_by_gpu_map = {}

        # Fill mapping relationships
        physical_expert_id = 0
        logical_to_physical_count = {}

        for gpu_id, experts in gpu_experts.items():
            for logical_expert_id in experts:
                # Update physical_to_logical_map
                physical_to_logical_map[physical_expert_id] = logical_expert_id
                
                # Update logical_to_all_physical_map
                if logical_expert_id not in logical_to_physical_count:
                    logical_to_physical_count[logical_expert_id] = 0
                
                replica_idx = logical_to_physical_count[logical_expert_id]
                logical_to_all_physical_map[logical_expert_id, replica_idx] = physical_expert_id
                logical_to_physical_count[logical_expert_id] += 1
                
                # Update logical_to_all_physical_map_num_valid
                logical_to_all_physical_map_num_valid[logical_expert_id] = logical_to_physical_count[logical_expert_id]
                
                # Update logical_to_rank_dispatch_physical_map
                if logical_to_rank_dispatch_physical_map[logical_expert_id] == -1:
                    logical_to_rank_dispatch_physical_map[logical_expert_id] = physical_expert_id
                    
                # Update logical_to_physical_by_gpu_map
                logical_to_physical_by_gpu_map[(logical_expert_id, gpu_id)] = physical_expert_id
                
                physical_expert_id += 1

        # Create ExpertLocationMetadata instance
        expert_location = ExpertLocationMetadata(
            physical_to_logical_map=physical_to_logical_map,
            logical_to_all_physical_map=logical_to_all_physical_map,
            logical_to_all_physical_map_num_valid=logical_to_all_physical_map_num_valid,
            logical_to_rank_dispatch_physical_map=logical_to_rank_dispatch_physical_map,
            logical_to_physical_by_gpu_map=logical_to_physical_by_gpu_map,
            num_physical_experts=num_physical_experts,
            num_logical_experts=num_logical_experts,
            num_gpus=gpu_nums
        )
        
        expert_location_eplbmetadata[layer_idx] = expert_location

    # Print results for verification
    print("\nExpert Placement:")
    for gpu_id, placement in enumerate(expert_placement):
        print(f"\nGPU {gpu_id}:")
        for layer, experts in placement.items():
            print(f"  Layer {layer}: {sorted(list(experts))}")

    print("\nExpert Placement Per Layer:")
    for layer_idx, gpu_experts in expert_placement_per_layer.items():
        print(f"\nLayer {layer_idx}:")
        for gpu_id, experts in gpu_experts.items():
            print(f"  GPU {gpu_id}: {sorted(list(experts))}")

    print("\nEPLB Metadata:")
    for layer_idx, metadata in expert_location_eplbmetadata.items():
        print(f"\nLayer {layer_idx}:")
        print(f"  Num logical experts: {metadata.num_logical_experts}")
        print(f"  Num physical experts: {metadata.num_physical_experts}")
        print(f"  Physical to logical map: {metadata.physical_to_logical_map.tolist()}")
        print(f"  Logical to all physical map: {metadata.logical_to_all_physical_map.tolist()}")
        print(f"  Logical to all physical map num valid: {metadata.logical_to_all_physical_map_num_valid.tolist()}")

if __name__ == "__main__":
    test_expert_mapping() 