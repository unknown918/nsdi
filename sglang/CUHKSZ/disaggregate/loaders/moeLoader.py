import sys
import os
import torch
import json
from safetensors.torch import safe_open
import argparse
from typing import Dict, List, Tuple, Optional
import glob
from safetensors.torch import load_file
import re
from collections import defaultdict


def find_weight_files(model_path: str) -> Tuple[List[str], Optional[Dict]]:
    weight_map = None  # weight map : eg : {"model.layers.0.experts.0.down_proj.weight": "model.layers.0.experts.0.down_proj.safetensors"}
    weight_files = []

    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index_data = json.load(f)
            weight_map = index_data.get("weight_map", {})

        # get all unique files
        unique_files = set(weight_map.values())
        weight_files = [os.path.join(model_path, f) for f in unique_files]

    if not weight_files:
        raise ValueError(f"No weight files found in {model_path}")

    return weight_files, weight_map


def get_file_for_weight(weight_name: str, weight_map: Optional[Dict]) -> Optional[str]:
    if weight_map is None:
        return None

    return weight_map.get(weight_name)


# def load_moe_experts(model_path: str, num_layers: int, start_expert_id, end_expert_id, num_experts=None):
#     """
#     加载模型权重，可以选择性地只加载特定层的特定专家

#     Args:
#         model_path: 模型路径
#         num_layers: 总层数
#         layers_to_experts: 可选，指定每层需要加载的专家ID，格式为{layer_id: [expert_ids]}

#     Returns:
#         dict: 加载的权重字典
#     """
#     weights_dict = {}

#     # 选择性加载特定层的特定专家

#     for layer_id in range(num_layers):
#         strid = start_expert_id[layer_id]
#         endid = end_expert_id[layer_id]
#         print(f"load layer {layer_id} from {strid} to {endid}")
#         for expert_id in range(strid, endid+1):
#             expert_weights = load_specific_expert(model_path, layer_id, expert_id)
#             if expert_weights == {}:
#                 continue
#             weights_dict.update(expert_weights)

#     return weights_dict

def load_moe_experts_old(model_path: str, num_layers: int, logical_experts: List[int]):
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file, 'r') as f:
        weight_map = json.load(f).get("weight_map", {})
    weights_to_load_by_file = defaultdict(list)
    verfiy = defaultdict(list)
    expert_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\..+")
    gate_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.gate\.weight")
    # gate_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.gate\")
    for tensor_name, filename in weight_map.items():
        # match = expert_pattern.match(tensor_name)
        match = expert_pattern.match(tensor_name)
        if match:
            layer_idx, expert_idx = int(match.group(1)), int(match.group(2))
            if layer_idx < num_layers:
                if expert_idx in logical_experts:
                    # start_id, end_id = start_expert_id[layer_idx], end_expert_id[layer_idx]
                    # if start_id <= expert_idx <= end_id:
                    weights_to_load_by_file[filename].append(tensor_name)
                    verfiy[filename].append(
                        tensor_name.replace("model.layers.", "L").replace(".mlp.experts.", ".E").replace(
                            ".down_proj.weight", "").replace("_proj.weight", ""))
        match2 = gate_pattern.match(tensor_name)
        if match2:
            layer_idx = int(match2.group(1))
            if layer_idx < num_layers:
                weights_to_load_by_file[filename].append(tensor_name)
                verfiy[filename].append(tensor_name.replace("model.layers.", "L").replace(".mlp.gate.weight", "gate"))
    weights_dict = {}
    # print(f"verfiy: {verfiy}")
    for filename, tensors_to_load in weights_to_load_by_file.items():
        file_path = os.path.join(model_path, filename)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for tensor_name in tensors_to_load:
                # 如果9 和gate包含在tensor_name中，则打印出来

                weights_dict[tensor_name] = f.get_tensor(tensor_name)
                # if weights_dict[tensor_name].min().item() == weights_dict[tensor_name].max().item():
                #     print("⚠️ tensor is all zeros for ", tensor_name)
                # if "mlp.gate.weight" in tensor_name and "9" in tensor_name:
                #     print(f"tensor_name: {tensor_name}")
                #     print(f"tensor: {weights_dict[tensor_name].min().item()}, {weights_dict[tensor_name].max().item()}")
    return weights_dict


def load_moe_experts(model_path: str, num_layers: int, logical_experts):
    """
    加载 MoE 专家权重。

    参数 logical_experts 支持两种形式：
    - List[int]: 旧接口，表示所有 layer 共享同一批 logical expert id
    - Dict[int, List[int]]: 新接口，按 layer 传入需要的 logical expert id
      例如：{0: [0, 1, 2], 1: [3, 4], ...}
    """
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_file, 'r') as f:
        weight_map = json.load(f).get("weight_map", {})

    weights_to_load_by_file = defaultdict(list)
    verfiy = defaultdict(list)
    expert_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\..+")
    gate_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.gate\.weight")
    # gate_pattern = re.compile(r"model\.layers\.(\d+)\.mlp\.gate\")

    # 兼容两种传参形式：List[int]（所有层一样）或 Dict[int, List[int]]（按层不同）
    if isinstance(logical_experts, dict):
        # layer -> set(expert_id)
        layer2experts = {int(k): set(v) for k, v in logical_experts.items()}

        def need_expert(layer_idx: int, expert_idx: int) -> bool:
            experts = layer2experts.get(layer_idx)
            return experts is not None and expert_idx in experts

    else:
        # 旧接口：所有 layer 共用一份 logical_experts
        logical_expert_set = set(logical_experts)

        def need_expert(layer_idx: int, expert_idx: int) -> bool:
            return expert_idx in logical_expert_set

    for tensor_name, filename in weight_map.items():
        # match = expert_pattern.match(tensor_name)
        match = expert_pattern.match(tensor_name)
        if match:
            layer_idx, expert_idx = int(match.group(1)), int(match.group(2))
            if layer_idx < num_layers and need_expert(layer_idx, expert_idx):
                weights_to_load_by_file[filename].append(tensor_name)
                verfiy[filename].append(
                    tensor_name.replace("model.layers.", "L")
                    .replace(".mlp.experts.", ".E")
                    .replace(".down_proj.weight", "")
                    .replace("_proj.weight", "")
                )
        match2 = gate_pattern.match(tensor_name)
        if match2:
            layer_idx = int(match2.group(1))
            if layer_idx < num_layers:
                weights_to_load_by_file[filename].append(tensor_name)
                verfiy[filename].append(tensor_name.replace("model.layers.", "L").replace(".mlp.gate.weight", "gate"))
    weights_dict = {}
    # print(f"verfiy: {verfiy}")
    for filename, tensors_to_load in weights_to_load_by_file.items():
        file_path = os.path.join(model_path, filename)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for tensor_name in tensors_to_load:
                # 如果9 和gate包含在tensor_name中，则打印出来

                weights_dict[tensor_name] = f.get_tensor(tensor_name)
                # if weights_dict[tensor_name].min().item() == weights_dict[tensor_name].max().item():
                #     print("⚠️ tensor is all zeros for ", tensor_name)
                # if "mlp.gate.weight" in tensor_name and "9" in tensor_name:
                #     print(f"tensor_name: {tensor_name}")
                #     print(f"tensor: {weights_dict[tensor_name].min().item()}, {weights_dict[tensor_name].max().item()}")
    return weights_dict


def analyze_experts(experts_weights):
    if not experts_weights:
        print("no expert weights loaded")
        return

    layers = {}
    for name, tensor in experts_weights.items():
        parts = name.split(".")
        layer_idx = int(parts[2])
        if layer_idx not in layers:
            layers[layer_idx] = []
        layers[layer_idx].append((name, tensor))

    for layer_idx in sorted(layers.keys()):
        print(f"\nlayer {layer_idx} expert weights:")
        layer_weights = layers[layer_idx]

        # calculate the total memory of the layer weights
        total_memory = sum(tensor.nelement() * tensor.element_size() for _, tensor in layer_weights)
        print(f"  Total weights: {len(layer_weights)}, Total memory: {total_memory / (1024 ** 2):.2f} MB")

        # print detailed information of each weight
        # for name, tensor in layer_weights:
        #     print(f"  - {name}, shape: {tensor.shape}, dtype: {tensor.dtype}, size: {tensor.nelement() * tensor.element_size() / (1024**2):.2f} MB")


def load_specific_expert(model_path: str, layer_idx: int, expert_id: int):
    """
    从模型文件中只加载指定层的指定专家权重

    Args:
        model_path: 模型路径
        layer_idx: 要加载的层索引
        expert_id: 要加载的专家ID

    Returns:
        Dict[str, torch.Tensor]: 包含指定层指定专家权重的字典
    """
    weights_dict = {}

    layer_pattern = f"model.layers.{layer_idx}"
    expert_pattern = f"experts.{expert_id}."

    if os.path.isfile(model_path):
        files = [model_path]
    else:
        files = glob.glob(os.path.join(model_path, "*.safetensors"))
        if not files:
            files = glob.glob(os.path.join(model_path, "*.bin"))

    files = sorted(files)

    find_flag = False
    for filename in files:
        # print(f"loading weights from {filename}")
        file_weights = load_file(filename)

        # filter out the weights of the specified layer and expert
        for name, tensor in file_weights.items():
            if layer_pattern in name and expert_pattern in name:
                # print(f"load weight: {name}, shape: {tensor.shape}")
                weights_dict[name] = tensor
                find_flag = True

        if find_flag:
            break

    if not weights_dict:
        if layer_idx == 0:
            return {}  # special for deepseek-v2 (first layer is not moe)
        else:
            raise ValueError(f"failed to load the weights of layer{layer_idx} and expert{expert_id}")

    # print(f"successfully loaded expert{expert_id} from layer{layer_idx}, {len(weights_dict)} parameters")
    return weights_dict


def main():
    parser = argparse.ArgumentParser(description="load MoE expert weights from model weights")
    parser.add_argument("--model_path", type=str,
                        default="/home/moe/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9")
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--analyze", default=True)
    args = parser.parse_args()

    os.environ["NUM_HIDDEN_LAYERS"] = str(args.num_layers)

    # experts_weights = load_moe_experts(args.model_path, args.num_layers)
    specific_expert_weights = load_specific_expert(args.model_path, 2, 1)
    # if args.analyze:
    #     analyze_experts(experts_weights)


if __name__ == "__main__":
    main()
