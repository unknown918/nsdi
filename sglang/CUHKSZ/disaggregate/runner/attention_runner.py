import logging
import os
from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.layers.attention.double_sparsity_backend import DoubleSparseAttnBackend
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.memory_pool import (
    DoubleSparseTokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader import get_model
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    enable_show_time_cost,
    get_available_gpu_memory,
    is_cuda,
    is_hip,
    monkey_patch_p2p_access_check,
    monkey_patch_vllm_gguf_config,
    set_cpu_offload_max_bytes,
)

from sglang.srt.configs.logger_config import configure_logger
from sglang.CUHKSZ.disaggregate.communication.utils import get_ep_group_info
from sglang.CUHKSZ.disaggregate.communication.nvshmem_utils import init_nvshmem_distributed
from sglang.srt.distributed.parallel_state import init_expert_parallel_group

logger = configure_logger(__name__)


class AttentionRunner:
    def __init__(
            self,
            model_config: ModelConfig,
            mem_fraction_static: float,
            gpu_id: int,
            tp_rank: int,
            tp_size: int,
            nccl_port: int,
            server_args: ServerArgs,
            is_draft_worker: bool = False,
    ):
        self.model_config = model_config
        self.mem_fraction_static = mem_fraction_static
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.att_worker_rank = tp_rank
        self.att_worker_size = tp_size
        self.tp_size = tp_size
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.should_log = tp_rank == 0
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.moe_node_num = server_args.moe_node_num

        if (
                self.model_config.attention_arch == AttentionArch.MLA
                and not self.server_args.disable_mla
        ):
            # TODO: add MLA optimization on CPU
            if self.server_args.device != "cpu":
                logger.info("MLA optimization is turned on. Use triton backend.")
                self.server_args.attention_backend = "triton"

        if server_args.show_time_cost:
            enable_show_time_cost()
        if server_args.disable_outlines_disk_cache:
            from outlines.caching import disable_cache
            disable_cache()

        ep_group_info = get_ep_group_info(
            tp_size=server_args.tp_size,
            ep_size=server_args.ep_size,
            moe_node_num=server_args.moe_node_num,
            enable_ep_intra_node_reduce=server_args.enable_ep_intra_node_reduce
        )

        self.num_micro_batch = server_args.num_micro_batch

        global_server_args_dict.update(
            {
                "ucx_peers": None,
                "tp_size": server_args.tp_size,
                "ep_size": server_args.ep_size,
                "attention_backend": server_args.attention_backend,
                "sampling_backend": server_args.sampling_backend,
                "triton_attention_reduce_in_fp32": server_args.triton_attention_reduce_in_fp32,
                "disable_mla": server_args.disable_mla,
                "torchao_config": server_args.torchao_config,
                "enable_nan_detection": server_args.enable_nan_detection,
                "enable_dp_attention": server_args.enable_dp_attention,
                "enable_ep_moe": server_args.enable_ep_moe,
                "device": server_args.device,
                "ep_group_info": ep_group_info,
                "num_micro_batch": server_args.num_micro_batch,
            }
        )
        set_cpu_offload_max_bytes(int(server_args.cpu_offload_gb * 1024 ** 3))
        min_per_gpu_memory = self.init_torch_distributed()

        init_nvshmem_distributed(server_args)

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )
        self.sampler = Sampler()
        self.load_model()

        torchao_applied = getattr(self.model, "torchao_applied", False)
        if torchao_applied:
            apply_torchao_config_to_model(
                self.model, global_server_args_dict["torchao_config"]
            )

        self.torch_tp_applied = False
        self.init_memory_pool(
            min_per_gpu_memory,
            server_args.max_running_requests,
            server_args.max_total_tokens,
        )
        if self.device == "cuda":
            self.init_cublas()
            self.init_attention_backend()
        else:
            self.cuda_graph_runner = None
            self.init_attention_backend()

    def pingpong(self):
        return

    def init_torch_distributed(self):
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()
        torch.cuda.set_device(self.att_worker_rank)
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        if not self.is_draft_worker:
            init_distributed_environment(
                backend="nccl",
                world_size=self.server_args.tp_size + self.server_args.ep_size,
                rank=self.att_worker_rank,
                local_rank=self.gpu_id,
                distributed_init_method="tcp://" + self.server_args.dist_init_addr,
            )
            init_expert_parallel_group(self.server_args.ep_size, self.moe_node_num)
            initialize_model_parallel(
                tensor_model_parallel_size=self.server_args.tp_size,
                expert_model_parallel_size=self.server_args.ep_size
            )
            initialize_dp_attention(
                enable_dp_attention=self.server_args.enable_dp_attention,
                tp_rank=self.att_worker_rank,
                tp_size=self.att_worker_size,
                dp_size=self.server_args.dp_size,
            )
            logger.info("torch.distributed initialized")

        min_per_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.att_worker_size > 1
        )

        self.tp_group = get_tp_group()
        self.attention_tp_group = get_attention_tp_group()

        if self.att_worker_size > 1:
            local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
            if min_per_gpu_memory < local_gpu_memory * 0.9:
                raise ValueError(
                    "The memory capacity is unbalanced. Some GPUs may be occupied by other processes."
                )

        logger.info(f"min_per_gpu_memory: {min_per_gpu_memory}")
        return min_per_gpu_memory

    def load_model(self):
        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
        )
        if self.server_args.load_format == "gguf":
            monkey_patch_vllm_gguf_config()
        monkey_patch_vllm_parallel_state()
        with self.memory_saver_adapter.region():
            self.model = get_model(
                model_config=self.model_config,
                load_config=self.load_config,
                device_config=DeviceConfig(self.device),
            )
        monkey_patch_vllm_parallel_state(reverse=True)
        self.sliding_window_size = (
            self.model.get_attention_sliding_window_size()
            if hasattr(self.model, "get_attention_sliding_window_size")
            else None
        )
        self.dtype = self.model_config.dtype

    def profile_max_num_token(self, total_gpu_memory: int):
        available_gpu_memory = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=self.att_worker_size > 1
        )
        if (
                self.model_config.attention_arch == AttentionArch.MLA
                and not self.server_args.disable_mla
        ):
            cell_size = (
                    (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                    * self.model_config.num_hidden_layers
                    * torch._utils._element_size(self.kv_cache_dtype)
            )
        else:
            cell_size = (
                    self.model_config.get_num_kv_heads(get_attention_tp_size())
                    * self.model_config.head_dim
                    * self.model_config.num_hidden_layers
                    * 2
                    * torch._utils._element_size(self.kv_cache_dtype)
            )
        rest_memory = available_gpu_memory - total_gpu_memory * (
                1 - self.mem_fraction_static
        )
        max_num_token = int(rest_memory * (1 << 30) // cell_size)
        return max_num_token

    def init_memory_pool(
            self,
            total_gpu_memory: int,
            max_num_reqs: Optional[int] = None,
            max_total_tokens: Optional[int] = None,
    ):
        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if is_hip():  # Using natively supported format
                self.kv_cache_dtype = torch.float8_e5m2fnuz
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if is_cuda():
                self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

        self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)

        if max_num_reqs is None:
            max_num_reqs = min(
                max(
                    int(
                        self.max_total_num_tokens / self.model_config.context_len * 512
                    ),
                    2048,
                ),
                4096,
            )

        if not self.spec_algorithm.is_none():
            if self.is_draft_worker:
                self.max_total_num_tokens = self.server_args.draft_runner_cache_size
            else:
                self.server_args.draft_runner_cache_size = (
                        self.max_total_num_tokens
                        + max_num_reqs * self.server_args.speculative_num_steps
                        + 100
                )

        if max_total_tokens is not None:
            if max_total_tokens > self.max_total_num_tokens:
                logging.warning(
                    f"max_total_tokens={max_total_tokens} is larger than the profiled value "
                    f"{self.max_total_num_tokens}. "
                    f"Use the profiled value instead."
                )
            self.max_total_num_tokens = min(self.max_total_num_tokens, max_total_tokens)

        if self.max_total_num_tokens <= 0:
            raise RuntimeError(
                "Not enough memory. Please try to increase --mem-fraction-static."
            )

        self.req_to_token_pool = ReqToTokenPool(
            size=max_num_reqs + 1,
            max_context_len=self.model_config.context_len + 4,
            device=self.device,
            enable_memory_saver=self.server_args.enable_memory_saver,
        )
        if (
                self.model_config.attention_arch == AttentionArch.MLA
                and not self.server_args.disable_mla
        ):
            self.token_to_kv_pool = MLATokenToKVPool(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
            )
        elif self.server_args.enable_double_sparsity:
            self.token_to_kv_pool = DoubleSparseTokenToKVPool(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                heavy_channel_num=self.server_args.ds_heavy_channel_num,
                enable_memory_saver=self.server_args.enable_memory_saver,
            )
        else:
            self.token_to_kv_pool = MHATokenToKVPool(
                self.max_total_num_tokens,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.model_config.num_hidden_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
            )
        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        logger.AttentionRunner("Init attention backend: %s" % self.server_args.attention_backend)

        def creator():
            if self.server_args.attention_backend == "flashinfer":
                return FlashInferAttnBackend(self)
            elif self.server_args.attention_backend == "triton":
                assert self.sliding_window_size is None, (
                    "Window attention is not supported in the triton attention backend. "
                    "Please use `--attention-backend flashinfer`."
                )
                assert not self.model_config.is_encoder_decoder, (
                    "Cross attention is not supported in the triton attention backend. "
                    "Please use `--attention-backend flashinfer`."
                )
                if self.server_args.enable_double_sparsity:
                    return DoubleSparseAttnBackend(self)
                else:
                    return TritonAttnBackend(self)
            elif self.server_args.attention_backend == "torch_native":
                return TorchNativeAttnBackend(self)
            else:
                raise ValueError(
                    f"Invalid attention backend: {self.server_args.attention_backend}"
                )

        logger.info(f"[MBO] start init attention backend, MboAttentionBackend {(self.num_micro_batch > 1)}")
        if self.num_micro_batch > 1:
            from sglang.srt.layers.attention.mbo_backend import MboAttnBackend
            self.attn_backend = MboAttnBackend.init_new(creator, self.num_micro_batch)
        else:
            self.attn_backend = creator()

    def forward_decode(self, forward_batch: ForwardBatch):
        self.attn_backend.init_forward_metadata(forward_batch)
        return self.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

    def forward_extend(self, forward_batch: ForwardBatch):
        self.attn_backend.init_forward_metadata(forward_batch)
        if self.is_generation:
            if forward_batch.input_embeds is None:
                return self.model.forward(
                    forward_batch.input_ids, forward_batch.positions, forward_batch
                )
            else:
                return self.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    input_embeds=forward_batch.input_embeds.bfloat16(),
                )
        else:
            # Only embedding models have get_embedding parameter
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                get_embedding=True,
            )

    def forward_idle(self, forward_batch: ForwardBatch):
        return self.model.forward(
            forward_batch.input_ids, forward_batch.positions, forward_batch
        )

    def forward_layer(self, forward_batch: ForwardBatch, layer_idx: int) -> torch.Tensor:
        self.attn_backend.init_forward_metadata(forward_batch)
        attn_out = self.model.forward_layer(
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            layer_idx=layer_idx
        )

        slices_to_moe, input_split_sizes = self.route_tokens_to_moe(attn_out)
        if len(slices_to_moe) > 0:
            moe_input = torch.cat(slices_to_moe, dim=0)
            moe_output = torch.empty_like(moe_input)
            dist.all_to_all_single(
                moe_output,
                moe_input,
                input_split_sizes=input_split_sizes,
                output_split_sizes=input_split_sizes
            )
            moe_output = self.moe_model.forward_with_gate(layer_idx, moe_output)
            attn_input = torch.empty_like(moe_output)
            dist.all_to_all_single(
                attn_input,
                moe_output,
                input_split_sizes=input_split_sizes,
                output_split_sizes=input_split_sizes
            )
            hidden_state_next_layer = torch.cat(
                torch.split(attn_input, input_split_sizes, dim=0),
                dim=0
            )
        else:
            hidden_state_next_layer = attn_out
        return hidden_state_next_layer

    def forward(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        tp_size = get_tensor_model_parallel_world_size()
        local_rank = get_tensor_model_parallel_rank()

        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(forward_batch)
        elif forward_batch.forward_mode.is_extend():
            return self.forward_extend(forward_batch)
        elif forward_batch.forward_mode.is_idle():
            return self.forward_idle(forward_batch)
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

    def sample(
            self, logits_output: LogitsProcessorOutput, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        # Apply logit bias
        sampling_info = forward_batch.sampling_info
        if sampling_info.sampling_info_done:
            # Overlap mode: the function update_regex_vocab_mask was executed
            # in process_batch_result of the last batch.
            if sampling_info.grammars:
                sampling_info.sampling_info_done.wait()
        else:
            # Normal mode: Put CPU-heavy tasks here. They will be overlapped with the forward pass.
            sampling_info.update_regex_vocab_mask()
            sampling_info.update_penalties()

        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
        )
        return next_token_ids

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(self.model_config.hf_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        return rope_scaling.get("type", None) == "mrope"
