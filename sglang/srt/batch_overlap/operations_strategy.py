

from dataclasses import dataclass
from typing import List, Optional

import torch

from srt.batch_overlap import operations
from srt.batch_overlap.operations import Operation
from srt.model_executor.forward_batch_info import ForwardMode


@dataclass
class OperationsStrategy:
    operations: List[Operation]
    mbo_delta_stages: Optional[int] = None

    @classmethod
    def concat(cls, items: List["OperationsStrategy"]) -> "OperationsStrategy":
        return OperationsStrategy(
            operations=[x for item in items for x in item.operations],
            mbo_delta_stages=_assert_all_same(
                [item.mbo_delta_stages for item in items]
            ),
        )

    @staticmethod
    def init_new_mbo(
        layers: torch.nn.ModuleList,
        forward_mode: ForwardMode,
    ) -> "OperationsStrategy":
        layer_name = layers[0].__class__.__name__
        if layer_name == "DeepseekV2DecoderLayer":
            return OperationsStrategy.concat(
                [
                    _compute_moe_deepseek_v2_mbo(layer)
                    for layer in layers
                ]
            )
        else:
            raise NotImplementedError


def _assert_all_same(items: List):
    assert all(item == items[0] for item in items)
    return items[0]


# -------------------------------- Strategy for DeepSeek ---------------------------------------


# TODO: maybe use different strategy for different forward modes
def _compute_moe_deepseek_v2_mbo(layer):
    return OperationsStrategy(
        mbo_delta_stages=1,
        operations=[
            layer.op_attn,
            operations.YieldOperation(),
            layer.mlp.op_commit_a2e,
            operations.YieldOperation(),
            layer.op_wait_exp,
            operations.YieldOperation(),
            layer.mlp.op_wait_e2a,
            layer.op_post_process,
            operations.YieldOperation()
        ]
    )


# -------------------------------- Strategy for Qwen3 ---------------------------------------

def _compute_moe_qwen3_layer_operations_strategy_mbo(
    layer: torch.nn.Module,
    forward_mode: ForwardMode,
) -> OperationsStrategy:
    pass

def _compute_moe_qwen3_prefill(layer):
    pass

def _compute_moe_qwen3_decode(layer):
    pass
