import torch
import dataclasses
from typing import Dict, List, Optional, Sequence
from srt.batch_overlap.operations import execute_overlapped_operations
from srt.batch_overlap.operations_strategy import OperationsStrategy
from srt.layers.attention import AttentionBackend
from srt.layers.attention.mbo_backend import MboAttnBackend
from srt.model_executor.cuda_graph_runner import CudaGraphRunner
from srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)

from sglang.srt.configs.logger_config import configure_logger
logger = configure_logger(__name__)

import asyncio

# use tokens number to balance each micro-batch
def compute_split_index(batch: ForwardBatch):
    if batch.forward_mode.is_decode():
        # logger.Model("[MBO] decode mode")
        return _compute_split_index_decode(
            batch.input_ids.shape[0],
            batch.seq_lens,
            batch.num_micro_batch,
        )
    elif batch.forward_mode.is_extend():
        # logger.Model("[MBO] extend mode") 
        return _compute_split_index_extend(
            batch.input_ids.shape[0],
            batch.extend_seq_lens,
            batch.num_micro_batch,
        )
    else:
        raise NotImplementedError(f"Unsupported forward mode {batch.forward_mode} for MBO.")


def _compute_split_index_decode(
    num_tokens: int,
    seq_lens: torch.Tensor,
    num_micro_batch: int,
):
    num_seqs = seq_lens.shape[0]
    assert num_tokens == num_seqs, f"num_tokens {num_tokens} != num_seqs {num_seqs} in decode mode"
    assert num_seqs >= num_micro_batch, f"num_seqs {num_seqs} < num_micro_batch {num_micro_batch}, we dont support empty micro-batch"
    
    seq_ends = []
    token_ends = []
    cur_tokens = 0
    cur_seqs = 0
    base = len(seq_lens) // num_micro_batch
    for i in range(num_micro_batch - 1):
        start_seq_idx = cur_seqs
        end_seq_idx = cur_seqs + base
        cur_seqs += base
        cur_tokens += base
        seq_ends.append(cur_seqs)
        token_ends.append(cur_tokens)
    # last batch takes all remaining
    seq_ends.append(len(seq_lens))
    token_ends.append(num_tokens)
    assert seq_ends[-1] == num_seqs
    assert token_ends[-1] == num_tokens
    return seq_ends, token_ends


def _compute_split_index_extend(
    num_tokens: int,
    seq_lens: torch.Tensor,
    num_micro_batch: int,
):
    seq_lens = seq_lens.tolist()
    assert len(seq_lens) >= num_micro_batch, f"len(seq_lens) {len(seq_lens)} < num_micro_batch {num_micro_batch}, we dont support empty micro-batch"

    seq_ends = []
    token_ends = []
    cur_tokens = 0
    cur_seqs = 0
    base = len(seq_lens) // num_micro_batch
    for i in range(num_micro_batch - 1):
        start_seq_idx = cur_seqs
        end_seq_idx = cur_seqs + base
        cur_seqs += base
        cur_tokens += sum(seq_lens[start_seq_idx:end_seq_idx])
        seq_ends.append(cur_seqs)
        token_ends.append(cur_tokens)
    # last batch takes all remaining
    seq_ends.append(len(seq_lens))
    token_ends.append(num_tokens)
    assert seq_ends[-1] == len(seq_lens), f"seq_ends[-1] {seq_ends[-1]} != len(seq_lens {len(seq_lens)}"
    assert token_ends[-1] == num_tokens, f"token_ends[-1] {token_ends[-1]} != num_tokens {num_tokens}"
    return seq_ends, token_ends
    
class MboForwardBatchPreparer:
    @classmethod
    def prepare(cls, batch: ForwardBatch):
        # logger.Model("[MBO]: preparing ForwardBatch for MBO")
        # logger.Model(f"[MBO]: seq lens {batch.seq_lens}")
        # logger.Model(f"[MBO]: extend prefix lens {batch.extend_prefix_lens}")
        # logger.Model(f"[MBO]: extend seq lens {batch.extend_seq_lens}")
        # logger.Model(f"[MBO]: num tokens {batch.input_ids.shape[0]}")
        mbo_split_seq_index, mbo_split_token_index = compute_split_index(batch)
        # logger.Model(f"MBO split token indices: {mbo_split_token_index}")
        # logger.Model(f"MBO split seq indices: {mbo_split_seq_index}")
        assert batch.attn_backend.__class__.__name__ == "MboAttnBackend", f"Expected MboAttnBackend, got {batch.attn_backend.__class__.__name__}"
        children = []
        for child_idx in range(batch.num_micro_batch):
            # using the same attn backend for each child
            attn_backend_child = batch.attn_backend.children[child_idx]
            assert child_idx < len(mbo_split_seq_index), f"child_idx {child_idx} >= len(mbo_split_seq_index) {len(mbo_split_seq_index)}"
            child = cls.filter_batch(
                batch,
                start_seq_index=0 if child_idx == 0 else mbo_split_seq_index[child_idx - 1],
                end_seq_index=mbo_split_seq_index[child_idx],
                start_tok_index=0 if child_idx == 0 else mbo_split_token_index[child_idx - 1],
                end_tok_index=mbo_split_token_index[child_idx],
                output_attn_backend=attn_backend_child,
            )
            children.append(child)
        batch.mbo_children = children

    @classmethod
    def filter_batch(
        cls,
        batch: ForwardBatch,
        start_seq_index: int,
        end_seq_index: int,
        start_tok_index: int,
        end_tok_index: int,
        output_attn_backend: AttentionBackend,
    ):
        # logger.Model(f"MBO filtering ForwardBatch [{start_seq_index}:{end_seq_index}]")
        num_tokens = batch.input_ids.shape[0]
        num_seqs = batch.batch_size
        seq_lens_sum = batch.seq_lens[start_seq_index:end_seq_index].sum().item()
        # logger.Model(f"MBO [{start_seq_index}:{end_seq_index}] seq_lens_sum {seq_lens_sum}")

        # small checks to avoid divide child again
        assert batch.mbo_parent_token_range is None, "Input batch should not be already divided for MBO."

        output_dict = dict()

        for key in [
            "input_ids",
            "positions",
            "out_cache_loc",
        ]:
            old_value = getattr(batch, key)
            assert (
                old_value.shape[0] == num_tokens
            ), f"{key=} {old_value=} {num_tokens=} {batch=}"
            output_dict[key] = old_value[start_tok_index:end_tok_index]

        for key in [
            "req_pool_indices",
            "seq_lens",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
            "extend_prefix_lens_cpu",
            "extend_seq_lens_cpu",
            "extend_logprob_start_lens_cpu",
            "lora_paths",
        ]:
            old_value = getattr(batch, key)
            if old_value is None:
                continue
            assert (
                len(old_value) == num_seqs
            ), f"{key=} {old_value=} {num_seqs=} {batch=}"
            output_dict[key] = old_value[start_seq_index:end_seq_index]

        for key in [
            "forward_mode",
            "return_logprob",
            "req_to_token_pool",
            "token_to_kv_pool",
            "can_run_dp_cuda_graph",
            "spec_info",
            "spec_algorithm",
            "capture_hidden_mode",
            "mrope_positions",  # only used by qwen2-vl, thus not care
        ]:
            output_dict[key] = getattr(batch, key)

        # print(f"MBO global num tokens {batch.global_num_tokens}")
        # print(f"MBO gathered_buffer is none {(batch.gathered_buffer is None)}")

        num_tokens = end_tok_index - start_tok_index
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size > 1:
            local_num_tokens = torch.tensor([num_tokens], dtype=torch.int64, device="cuda")
            global_num_tokens = torch.empty(tp_size, dtype=torch.int64, device="cuda")
            torch.distributed.all_gather_into_tensor(
                global_num_tokens,
                local_num_tokens,
                group=get_tp_group().device_group,
            )

            global_num_tokens = global_num_tokens.tolist()
            max_len = max(global_num_tokens)
            hidden_size = batch.gathered_buffer.shape[1]
            gather_buffer = torch.zeros(
                (max_len * tp_size, hidden_size),
                dtype=batch.gathered_buffer.dtype,
                device=batch.gathered_buffer.device,
            )
        else:
            global_num_tokens = batch.global_num_tokens
            gather_buffer = batch.gathered_buffer

        output_dict.update(
            dict(
                batch_size=end_seq_index - start_seq_index,
                seq_lens_sum=seq_lens_sum,
                extend_num_tokens=None,
                attn_backend=output_attn_backend,
                num_micro_batch=1, # individe
                mbo_parent_token_range=(start_tok_index, end_tok_index),
                mbo_children=None,
                sampling_info=None,
                image_inputs=None,
                global_num_tokens=global_num_tokens,
                gathered_buffer=gather_buffer,
            )
        )

        errors = []
        for field in dataclasses.fields(ForwardBatch):
            if getattr(batch, field.name) is not None and field.name not in output_dict:
                errors.append(
                    f"Field {field.name} has value, but is not yet supported (value={getattr(batch, field.name)} batch={batch})"
                )
        if len(errors) > 0:
            raise Exception(f"{len(errors)} errors happen:\n" + "\n\n".join(errors))

        return ForwardBatch(**output_dict)
    

# -------------------------------- Execution ---------------------------------------

async def model_forward_mbo(
    layers,
    num_micro_batch: int,
    mbo_stream_manager,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    hidden_states: torch.Tensor,
    residual: Optional[torch.Tensor],
):
    # logger.Model("[MBO]: using multi-batch overlap feature")
    # logger.Model(f"num_micro_batch={num_micro_batch}, hidden_states shape={hidden_states.shape}, residual shape={residual.shape if residual is not None else None}")
    inputs = dict(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
        residual=residual,
    )

    operations_strategy = OperationsStrategy.init_new_mbo(
        layers, forward_batch.forward_mode
    )
    inputs_arr = _model_forward_mbo_split_inputs(**inputs)
    del inputs

    outputs_arr = await execute_overlapped_operations(
        inputs_arr=inputs_arr,
        operations_arr=[operations_strategy.operations] * num_micro_batch,
        delta_stage=operations_strategy.mbo_delta_stages,
        num_micro_batch=num_micro_batch,
        mbo_stream_manager=mbo_stream_manager,
    )
    return _model_forward_mbo_merge_outputs(outputs_arr)


def _model_forward_mbo_split_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
) -> List[Dict]:
    return [
        dict(
            **_model_forward_filter_inputs(
                hidden_states=hidden_states,
                residual=residual,
                positions=positions,
                output_forward_batch=output_forward_batch,
            ),
        )
        for _, output_forward_batch in enumerate(
            forward_batch.mbo_children
        )
    ]


def _model_forward_filter_inputs(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    positions: torch.Tensor,
    output_forward_batch: ForwardBatch,
) -> Dict:
    token_slice = slice(*output_forward_batch.mbo_parent_token_range)
    # logger.Model(f"MBO filtering inputs for tokens {token_slice}")
    return dict(
        hidden_states=hidden_states[token_slice],
        residual=None if residual is None else residual[token_slice],
        positions=positions[token_slice],
        forward_batch=output_forward_batch,
    )


def _model_forward_mbo_merge_outputs(outputs):
    def _handle_key(name):
        values = [output[name] for output in outputs]
        assert all((value is None) == (values[0] is None) for value in values)
        if values[0] is None:
            return None
        return torch.concat(values, dim=0)

    return _handle_key("hidden_states"), _handle_key("residual")