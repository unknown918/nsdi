"""
NVSHMEM utility functions for disaggregated attention-expert communication.
Extracted from triton_dist.utils to avoid heavy triton_dist dependency.
"""

import os
import torch
import nvshmem.core

from torch import Tensor
import torch.distributed as dist
from sglang.srt.server_args import ServerArgs
from cuda.core.experimental import Device, system
from sglang.srt.configs.logger_config import configure_logger
from sglang.srt.managers.schedule_batch import global_server_args_dict
import nvshmem.bindings.nvshmem as bindings

logger = configure_logger(__name__)


def init_nvshmem_by_torch_process_group(pg: torch.distributed.ProcessGroup):
    """Initialize NVSHMEM using an existing torch.distributed process group.

    Env vars:
        NVSHMEM_DUMMY_PES  – number of dummy PEs that will join NVSHMEM but
                             are not part of the torch process group.  NVSHMEM
                             nranks = pg.size() + NVSHMEM_DUMMY_PES.
        NVSHMEM_UID_FILE   – when set, rank 0 writes the NVSHMEM unique-id to
                             this path so dummy PEs can bootstrap without
                             torch.distributed.
    """
    num_ranks = pg.size()
    rank_id = pg.rank()
    dummy_pes = int(os.environ.get("NVSHMEM_DUMMY_PES", 0))
    nvshmem_nranks = num_ranks + dummy_pes

    broadcast_objects = [nvshmem.core.get_unique_id(empty=rank_id != 0)]
    torch.distributed.broadcast_object_list(broadcast_objects, src=0, group=pg)
    torch.distributed.barrier(group=pg)

    uid = broadcast_objects[0]

    # Save UID for dummy PEs
    uid_file = os.environ.get("NVSHMEM_UID_FILE")
    if uid_file and rank_id == 0:
        import pickle
        with open(uid_file, "wb") as f:
            pickle.dump(uid, f)
        print(f"[nvshmem_utils] UID written to {uid_file}", flush=True)

    nvshmem.core.init(
        device=Device(torch.cuda.current_device()),
        uid=uid,
        rank=rank_id,
        nranks=nvshmem_nranks,
        initializer_method="uid",
    )


def nvshmem_create_tensor(shape, dtype) -> Tensor:
    """Allocate a symmetric NVSHMEM tensor visible to all ranks."""
    torch.cuda.synchronize()
    tensor = nvshmem.core.tensor(shape, dtype=dtype)
    torch.cuda.synchronize()
    return tensor


def nvshmem_free_tensor_sync(tensor):
    """Free an NVSHMEM symmetric tensor."""
    torch.cuda.synchronize()
    nvshmem.core.free_tensor(tensor)
    torch.cuda.synchronize()


def init_nvshmem_distributed(server_args: ServerArgs):
    """Initialize NVSHMEM buffers for attention <-> expert communication."""
    # Initialize NVSHMEM using the existing torch distributed group
    world_group = dist.group.WORLD
    init_nvshmem_by_torch_process_group(world_group)
    logger.info(f"[rank={dist.get_rank()}] NVSHMEM initialized.")

    # Allocate symmetric buffers for A2E and E2A communication
    # Use model config for hidden_dim to ensure consistency with expert side
    from sglang.srt.configs.model_config import ModelConfig
    model_cfg = ModelConfig(server_args.model_path, model_override_args="{}")
    hidden_dim = model_cfg.hf_config.hidden_size
    max_tokens = 16384
    buf_numel = max_tokens * hidden_dim
    # a2e buffers: metadata is now packed in the signal value, no header needed
    a2e_buf_numel = buf_numel
    logger.info(
        f"NVSHMEM buffer size: hidden_dim={hidden_dim}, max_tokens={max_tokens}, buf_numel={buf_numel}, a2e_buf_numel={a2e_buf_numel}",
    )

    num_micro_batch = server_args.num_micro_batch
    ep_group_info = global_server_args_dict["ep_group_info"]
    max_ep_peers = max(len(v) for v in ep_group_info["send_strategy"].values())

    # Per-micro-batch A2E buffers and signals
    # IMPORTANT: allocation order must match moe_runner exactly (symmetric)
    a2e_bufs = []
    a2e_sigs = []
    for _ in range(num_micro_batch):
        a2e_bufs.append(
            {
                "src": nvshmem_create_tensor((a2e_buf_numel,), torch.bfloat16),
                "dst": nvshmem_create_tensor((a2e_buf_numel,), torch.bfloat16),
            }
        )
        a2e_sigs.append({"sig": nvshmem.core.buffer(8)})

    # Per-micro-batch × per-peer E2A buffers and signals
    e2a_dst_slots = []
    e2a_src = []
    e2a_sigs = []
    for _ in range(num_micro_batch):
        e2a_dst_slots.append(
            [nvshmem_create_tensor((buf_numel,), dtype=torch.bfloat16) for _ in range(max_ep_peers)])
        e2a_src.append(nvshmem_create_tensor((buf_numel,), torch.bfloat16))
        e2a_sigs.append([nvshmem.core.buffer(8) for _ in range(max_ep_peers)])

    global_server_args_dict["nvshmem_a2e_bufs"] = a2e_bufs
    global_server_args_dict["nvshmem_a2e_sigs"] = a2e_sigs
    global_server_args_dict["nvshmem_e2a_dst_slots"] = e2a_dst_slots
    global_server_args_dict["nvshmem_e2a_src"] = e2a_src
    global_server_args_dict["nvshmem_e2a_sigs"] = e2a_sigs

    logger.info(
        f"[rank={dist.get_rank()}] NVSHMEM buffers allocated, buffer size: {buf_numel}, e2a_peers: {max_ep_peers}"
    )


class TorchStreamWrapper:
    """Wrap a torch.cuda.Stream for use with nvshmem APIs."""

    def __init__(self, pt_stream: torch.cuda.Stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)
