import os
import re
import json
import time
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from collections import defaultdict
from safetensors.torch import safe_open
from typing import Dict, List, Optional, Tuple

import torch.distributed as dist
from sglang.srt.layers.moe.topk import select_experts
from sglang.CUHKSZ.disaggregate.afd_moe.kernels.ep_moe_kernel import EPMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_available_gpu_memory
from sglang.CUHKSZ.disaggregate.static_placement.placement_generator import generate_expert_mapping

import greedy_schedule


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()

        if "Qwen" in config.architectures[0]:
            num_experts = config.num_experts
        elif "Deepseek" in config.architectures[0]:
            num_experts = config.n_routed_experts
        else:
            raise "Model not supported"

        self.weight = nn.Parameter(
            torch.empty(
                (num_experts, config.hidden_size)
            )
        )
        self.e_score_correction_bias = None

    def forward(self, hidden_states):
        logits = F.linear(hidden_states, self.weight, None)
        return logits


class DeepseekMoE(nn.Module):
    def __init__(
            self,
            config: PretrainedConfig,
            layer_idx: int,
            quant_config: Optional[QuantizationConfig] = None,
            start_expert_id: int = 0,
            end_expert_id: int = 0,
            ep_gpu_used: int = -1,
            expert2gpus_: dict = None,
            expert2phys_: dict = None,
            copy_count_: dict = None,
            max_copies_: dict = None,
            att_dp_num: int = 0,
    ):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.quant_config = quant_config
        self.ep_gpu = ep_gpu_used

        self.expert2gpus_ = expert2gpus_
        self.expert2phys_ = expert2phys_
        self.copy_count_ = copy_count_
        self.max_copies_ = max_copies_

        self.num_physical_experts = len(torch.unique(self.expert2phys_[self.expert2phys_ >= 0]))
        self.all_expert_have_replica_ = all(self.copy_count_ > 1)

        # Use Deepseek-style MoEGate (weight + optional correction bias)
        self.gate = MoEGate(config)

        if "Qwen" in config.architectures[0]:
            self.num_logical_experts = config.num_experts
            self.experts = EPMoE(
                # num_experts=config.n_routed_experts,
                num_experts=self.num_physical_experts,
                params_dtype=config.dtype,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                start_expert_id=start_expert_id,
                end_expert_id=end_expert_id,
                renormalize=config.norm_topk_prob,
                quant_config=quant_config,
                # use_grouped_topk=True,
                # num_expert_group=config.n_group,
                # topk_group=config.topk_group,
                correction_bias=self.gate.e_score_correction_bias,
            )
        elif "Deepseek" in config.architectures[0]:
            self.num_logical_experts = config.n_routed_experts
            self.experts = EPMoE(
                # num_experts=config.n_routed_experts,
                num_experts=self.num_physical_experts,
                params_dtype=config.dtype,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                start_expert_id=start_expert_id,
                end_expert_id=end_expert_id,
                renormalize=config.norm_topk_prob,
                quant_config=quant_config,
                use_grouped_topk=True,
                num_expert_group=config.n_group,
                topk_group=config.topk_group,
                correction_bias=self.gate.e_score_correction_bias,
            )
            if layer_idx == 0: return
        else:
            raise "Model not supported"

        if self.layer_idx == 1:
            print(f"rank={dist.get_rank()} gpu_id={torch.cuda.current_device()}\n"
                  f"start_expert_id={start_expert_id}, end_expert_id={end_expert_id}\n"
                  f"num_physical_experts={self.num_physical_experts}\n")

        self.to(dtype=self.config.torch_dtype)
        self.to("cuda")

        self.my_rank = torch.distributed.get_rank()
        self.att_dp_num = att_dp_num
        self.my_ep_rank = self.my_rank - self.att_dp_num  # MoE rank (0-based within MoE group)
        # self.moe_log = os.environ.get("MOE_LOG")
        #
        # comm_breakdown_str = os.environ.get("COMM_BREAKDOWN", "false")
        # compute_breakdown_str = os.environ.get("COMPUTE_BREAKDOWN", "false")
        # self.COMM_BREAKDOWN = comm_breakdown_str.lower() == "true"
        # self.COMPUTE_BREAKDOWN = compute_breakdown_str.lower() == "true"

    def forward(
            self,
            cat_hidden_states: torch.Tensor,
            cat_topk_ids: torch.Tensor,
            cat_topk_weights: torch.Tensor
    ) -> torch.Tensor:
        raise "shouldn't be here"

    def forward_with_gate(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # if self.COMM_BREAKDOWN:
        #     return hidden_states

        router_logits = self.gate(hidden_states)

        # if self.COMPUTE_BREAKDOWN:
        #     torch.cuda.synchronize()
        #     t_start = time.perf_counter()

        if "Qwen" in self.config.architectures[0]:
            topk_weights, topk_ids = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                use_grouped_topk=False,
                top_k=self.config.num_experts_per_tok,
                renormalize=self.config.norm_topk_prob,
                # topk_group=self.config.topk_group,
                # num_expert_group=self.config.n_group,
                # custom_routing_function=None,
                # correction_bias=self.gate.e_score_correction_bias,
            )
        elif "Deepseek" in self.config.architectures[0]:
            topk_weights, topk_ids = select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                use_grouped_topk=True,
                top_k=self.config.num_experts_per_tok,
                renormalize=self.config.norm_topk_prob,
                topk_group=self.config.topk_group,
                num_expert_group=self.config.n_group,
                custom_routing_function=None,
                correction_bias=self.gate.e_score_correction_bias,
            )

        ids = greedy_schedule.called_experts(topk_ids, self.num_logical_experts)
        mapped_topk = greedy_schedule.greedy_schedule(
            topk_ids,
            ids,
            self.expert2gpus_,
            self.expert2phys_,
            self.copy_count_,
            self.num_logical_experts,
            self.ep_gpu,
            self.all_expert_have_replica_
        )

        output = self.experts(hidden_states, mapped_topk, topk_weights)

        return output


class UnifiedMoE(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
            self,
            config: PretrainedConfig,
            server_args: ServerArgs,
            ep_gpu_used: int,
            quant_config: Optional[QuantizationConfig] = None,
            gpu_global_id=0,
            local_rank=0,
            att_dp_num=0,
    ) -> None:
        super().__init__()
        self.device = "cuda"
        print(f"MoE memory before loading: {get_available_gpu_memory(self.device, local_rank):.2f} GB")

        self.config = config
        self.quant_config = quant_config

        self.start_expert_id = {}
        self.end_expert_id = {}

        self.ep_gpu_used = ep_gpu_used
        self.gpu_global_id = gpu_global_id
        self.att_dp_num = att_dp_num
        self.my_ep_rank = dist.get_rank() - att_dp_num

        self.num_dense_layers = 0
        self.num_logical_experts = 0

        if "Qwen" in self.config.architectures[0]:
            self.num_logical_experts = config.num_experts
            self.num_dense_layers = 0
        elif "Deepseek" in config.architectures[0]:
            self.num_logical_experts = config.n_routed_experts
            self.num_dense_layers = 1

        self.num_layers = config.num_hidden_layers
        self.num_expert_per_gpu = server_args.num_expert_per_gpu

        self.expert_to_gpus = {}
        self.expert_gpu_to_phys = {}
        self.expert2gpus_ = {}
        self.expert2phys_ = {}
        self.copy_count_ = {}
        self.max_copies_ = {}

        for i in range(self.num_dense_layers):
            self.expert2gpus_[i] = {}
            self.expert2phys_[i] = {}
            self.copy_count_[i] = {}
            self.max_copies_[i] = {}

        for i in range(self.num_dense_layers, self.num_layers):
            # no Qwen's expert statistics, fall back to default setting
            self.expert_to_gpus[i], self.expert_gpu_to_phys[i] = generate_expert_mapping(
                num_gpus=self.ep_gpu_used,
                num_expert_per_gpu=self.num_expert_per_gpu,
                num_logical_experts=self.num_logical_experts,
                expert_counts=None,
                co_occurrence_counts=None,
                print_detail=False,
                redundant=True,
            )

            mapping = self.build_dense_mapping_from_dicts(
                self.expert_to_gpus[i],
                self.expert_gpu_to_phys[i],
                self.num_logical_experts,
                "cuda"
            )

            self.expert2gpus_[i] = mapping[0]
            self.expert2phys_[i] = mapping[1]
            self.copy_count_[i] = mapping[2]
            self.max_copies_[i] = mapping[3]

        for layer_id in range(self.config.num_hidden_layers):
            start_expert_id = self.my_ep_rank * self.num_expert_per_gpu
            end_expert_id = start_expert_id + self.num_expert_per_gpu - 1
            self.start_expert_id[layer_id] = start_expert_id
            self.end_expert_id[layer_id] = end_expert_id

        self.layers = nn.ModuleList(
            [
                DeepseekMoE(
                    config,
                    layer_id,
                    quant_config=quant_config,
                    start_expert_id=self.start_expert_id[layer_id],
                    end_expert_id=self.end_expert_id[layer_id],
                    ep_gpu_used=self.ep_gpu_used,
                    expert2gpus_=self.expert2gpus_[layer_id],
                    expert2phys_=self.expert2phys_[layer_id],
                    copy_count_=self.copy_count_[layer_id],
                    max_copies_=self.max_copies_[layer_id],
                    att_dp_num=self.att_dp_num
                )
                for layer_id in range(self.num_layers)
            ]
        )

        self.to(dtype=self.config.torch_dtype)
        self.to("cuda")

    @staticmethod
    def build_dense_mapping_from_dicts(
            expert_to_gpus,
            expert_gpu_to_phys,
            num_experts,
            device="cuda"
    ) -> Tuple:
        max_copies = max(len(gpus) for gpus in expert_to_gpus.values())
        expert2gpus = torch.full((num_experts, max_copies), -1, dtype=torch.int32, device=device)
        expert2phys = torch.full((num_experts, max_copies), -1, dtype=torch.int32, device=device)
        copy_count = torch.zeros(num_experts, dtype=torch.int32, device=device)

        for lid, gpus in expert_to_gpus.items():
            for j, g in enumerate(sorted(gpus)):
                expert2gpus[lid, j] = g
                expert2phys[lid, j] = expert_gpu_to_phys[lid][g]
            copy_count[lid] = len(gpus)

        return expert2gpus, expert2phys, copy_count, max_copies

    @torch.no_grad()
    def forward(
            self,
            layer_idx: int,
            cat_hidden_states: torch.Tensor,
            cat_topk_ids: torch.Tensor,
            cat_topk_weights: torch.Tensor
    ) -> torch.Tensor:
        temp = self.layers[layer_idx](cat_hidden_states, cat_topk_ids, cat_topk_weights)
        return temp

    @torch.no_grad()
    def forward_with_gate(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape[0] > 0, "hidden_states should have non-zero batch size."
        return self.layers[layer_idx].forward_with_gate(hidden_states, layer_idx)

    def get_experts_on_gpu(self, expert2gpus, gpu_id):
        return [eid for eid, gpus in expert2gpus.items() if gpu_id in gpus]

    def load_weights(self, model_path: str) -> None:
        layer2experts_on_gpu = {}
        for layer_id in range(1, self.num_layers):
            expert2gpus = self.expert_to_gpus.get(layer_id, None)
            experts_on_gpu = self.get_experts_on_gpu(expert2gpus, self.my_ep_rank)
            layer2experts_on_gpu[layer_id] = experts_on_gpu

        # get weight dicts for current device
        weights_dict = self.load_moe_experts(
            model_path,
            self.num_layers,
            layer2experts_on_gpu,
        )

        expert_params_mapping = EPMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.num_logical_experts,
        )

        params_dict = dict(self.named_parameters())

        for name, tensor in weights_dict.items():
            if "mlp.gate.weight" in name:
                layer_idx = int(name.split(".")[2])
                gate_param_name = f"layers.{layer_idx}.gate.weight"
                if gate_param_name in params_dict:
                    param = params_dict[gate_param_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, tensor)
                    continue

            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping

                if weight_name not in name:
                    continue

                layer_idx = int(name.split(".")[2])
                mapping_physical_id = self.expert_gpu_to_phys[layer_idx][expert_id][self.my_ep_rank]

                name = name.replace(weight_name, param_name)
                name = name.replace("model.", "").replace("mlp.", "")
                param = params_dict[name]

                weight_loader = param.weight_loader
                weight_loader(param, tensor, name, shard_id=shard_id, expert_id=mapping_physical_id)
                break

            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader(param, tensor)

    @staticmethod
    def load_moe_experts(model_path: str, num_layers: int, logical_experts):
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
                    verfiy[filename].append(
                        tensor_name.replace("model.layers.", "L").replace(".mlp.gate.weight", "gate"))
        weights_dict = {}
        for filename, tensors_to_load in weights_to_load_by_file.items():
            file_path = os.path.join(model_path, filename)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for tensor_name in tensors_to_load:
                    weights_dict[tensor_name] = f.get_tensor(tensor_name)
        return weights_dict
