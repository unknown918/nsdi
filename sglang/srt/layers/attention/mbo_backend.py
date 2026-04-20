
import torch
from typing import Callable, List, Optional
from srt.batch_overlap import micro_batch_overlap
from srt.layers.attention import AttentionBackend
from srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

"""
MboAttnBackend: Attention backend for micro-batch overlap (MBO) processing.
TODO: currently not support cuda graph for MBO attention backend.
"""
class MboAttnBackend(AttentionBackend):
    def __init__(self, primary: AttentionBackend, children: List[AttentionBackend]):
        super().__init__()
        self.primary = primary
        self.children = children
        self.num_micro_batch = len(children)

    @classmethod
    def init_new(cls, creator: Callable[[], AttentionBackend], num_micro_batch: int):
        return cls(
            primary=creator(),
            children=[creator() for _ in range(num_micro_batch)],
        )

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        self.primary.init_forward_metadata(forward_batch=forward_batch)
        if forward_batch.mbo_children is not None:
            for child, forward_batch_child in zip(
                self.children, forward_batch.mbo_children, strict=True
            ):
                # assert forward_batch_child.batch_size > 0, "Child forward batch should have non-zero batch size."
                if forward_batch_child.batch_size > 0:
                    child.init_forward_metadata(forward_batch=forward_batch_child)

    def forward_extend(self, *args, **kwargs):
        return self.primary.forward_extend(*args, **kwargs)

    def forward_decode(self, *args, **kwargs):
        return self.primary.forward_decode(*args, **kwargs)

    def get_indexer_metadata(self, layer_id: int, forward_batch: "ForwardBatch"):
        return self.primary.get_indexer_metadata(layer_id, forward_batch)