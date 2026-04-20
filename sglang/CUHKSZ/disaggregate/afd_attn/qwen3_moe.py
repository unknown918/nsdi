# Adapted from qwen2_moe.py

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, Optional, Tuple
import os
import torch
from torch import nn
import asyncio
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
)
from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen2_moe import Qwen2MoeModel
from sglang.srt.utils import add_prefix
from sglang.CUHKSZ.disaggregate.communication.nvshmem_comm import (
    AttnNvshmemCommunicationHandler as AttnUCXCommunicationHandler,
)
from sglang.srt.configs.logger_config import configure_logger

logger = configure_logger(__name__)
Qwen3MoeConfig = None
communicator = AttnUCXCommunicationHandler()


def all_gather(
        input_tensor: torch.Tensor,
        forward_batch: ForwardBatch,
        rank,
        world_size,
        group
):
    if world_size == 1:
        return input_tensor

    all_lens = forward_batch.global_num_tokens
    max_len = max(forward_batch.global_num_tokens)

    padded_tensor = torch.nn.functional.pad(
        input_tensor, (0, 0, 0, max_len - input_tensor.shape[0])
    )

    torch.distributed.all_gather_into_tensor(
        forward_batch.gathered_buffer,
        padded_tensor,
        group=group
    )

    gathered_tensors = torch.concat(
        [
            forward_batch.gathered_buffer[i * max_len: i * max_len + all_lens[i]]
            for i in range(world_size)
        ]
    )

    start_index = 0 if rank == 0 else sum(all_lens[:rank])
    end_index = start_index + all_lens[rank]

    return gathered_tensors, start_index, end_index


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(
            self,
            config: Qwen3MoeConfig,
            layer_id: int,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
            communicator=None
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # (afd) init a2e communicator
        self.communicator = communicator
        self.layer_id = layer_id

        comm_breakdown_str = os.environ.get("COMM_BREAKDOWN", "false")
        self.COMM_BREAKDOWN = comm_breakdown_str.lower() == "true"

    # (afd) forward on A
    async def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        await self.communicator.send_attention_result(0, self.layer_id, hidden_states)
        final_hidden_states = await self.communicator.recv_moe_result(self.layer_id, 0)
        # if self.n_shared_experts is not None:
        #     final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)
        return final_hidden_states


class Qwen3MoeAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            num_kv_heads: int,
            layer_id: int = 0,
            rope_theta: float = 10000,
            rope_scaling: Optional[Dict[str, Any]] = None,
            max_position_embeddings: int = 8192,
            head_dim: Optional[int] = None,
            rms_norm_eps: float = 1e-06,
            attention_bias: bool = False,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.tp_rank = get_tensor_model_parallel_rank()

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            # dtype=config.dtype,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix)
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def _apply_qk_norm(
            self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_by_head = q.reshape(-1, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.reshape(-1, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k

    def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)

        return output


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
            self,
            config: Qwen3MoeConfig,
            layer_id: int,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = ""
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        rms_norm_eps = config.rms_norm_eps
        attention_bias = config.attention_bias
        self.communicator = communicator
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            attention_bias=attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.enable_dp_attention = global_server_args_dict["enable_dp_attention"]
        if self.enable_dp_attention:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.tp_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group().device_group

        self.layer_id = layer_id
        self.mlp = Qwen3MoeSparseMoeBlock(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            communicator=self.communicator
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def stage_attn(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
            residual: Optional[torch.Tensor]
    ):
        if not forward_batch.forward_mode.is_idle():
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )

        start_idx, end_idx = 0, 0
        if self.enable_dp_attention and self.tp_size > 1:
            hidden_states, start_idx, end_idx = all_gather(
                hidden_states, forward_batch, self.tp_rank, self.tp_size, self.tp_group
            )
        return start_idx, end_idx, hidden_states, residual

    def stage_post_process(
            self,
            start_idx: int,
            end_idx: int,
            hidden_states: torch.Tensor,
            residual: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.enable_dp_attention and self.tp_size > 1:
            hidden_states = hidden_states[start_idx:end_idx]
        return hidden_states, residual

    async def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            forward_batch: ForwardBatch,
            residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start_idx, end_idx, hidden_states, residual = self.stage_attn(positions, hidden_states, forward_batch, residual)
        hidden_states = await self.mlp(hidden_states)
        hidden_states, residual = self.stage_post_process(start_idx, end_idx, hidden_states, residual)

        return hidden_states, residual


class Qwen3MoeModel(Qwen2MoeModel):
    def __init__(
            self,
            config: Qwen3MoeConfig,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3MoeDecoderLayer,
        )

    async def async_forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: ForwardBatch,
            input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = input_embeds

        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = await layer(
                positions, hidden_states, forward_batch, residual
            )
        if not forward_batch.forward_mode.is_idle():
            hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
            self,
            config: Qwen3MoeConfig,
            quant_config: Optional[QuantizationConfig] = None,
            prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config
        self.model = Qwen3MoeModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix)
        )
        if global_server_args_dict["enable_dp_attention"]:
            self.lm_head = ReplicatedLinear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(config, skip_all_gather=True)
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )
            self.logits_processor = LogitsProcessor(config)

    @torch.no_grad()
    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: ForwardBatch,
            input_embeds: torch.Tensor = None,
    ) -> LogitsProcessorOutput:
        hidden_states = asyncio.run(self.model.async_forward(input_ids, positions, forward_batch, input_embeds))
        #  hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    # (afd) skip expert loading on attn node
                    if ".experts." in name or ".gate." in name:
                        continue

                    if name not in params_dict:
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = Qwen3MoeForCausalLM
