"""
NVSHMEM-based communication handlers for disaggregated attention-expert.

Protocol:
  A2E: putmem_signal with metadata packed in signal value (zero-copy)
  E2A: putmem_signal data (zero-copy from compute output)
  Receiver: signal_wait, extract metadata from signal, read data from buffer

Signal value layout (64-bit):
  [63:40] iteration (24 bits, monotonic — used for CMP_GE ordering)
  [39:20] layer_index (20 bits)
  [19:0]  tokens     (20 bits, max ~1M tokens)
"""

import ctypes
import os
import torch
import torch.distributed as dist
import nvshmem.core
import nvshmem.bindings.nvshmem as bindings

from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.configs.logger_config import configure_logger
from .nvshmem_utils import (
    nvshmem_create_tensor,
    nvshmem_free_tensor_sync,
    TorchStreamWrapper,
)

logger = configure_logger(__name__)

DEBUG_NVSHMEM_VERIFY = os.environ.get("DEBUG_NVSHMEM_VERIFY", "0") == "1"

# ── Signal value packing ────────────────────────────────────────────────
ITER_SHIFT = 40
LAYER_SHIFT = 20
FIELD_MASK = (1 << 20) - 1  # 0xFFFFF, covers both layer and tokens fields


def _pack_signal(iteration: int, layer_index: int, tokens: int) -> int:
    """Pack iteration + metadata into a 64-bit signal value."""
    return (iteration << ITER_SHIFT) | (layer_index << LAYER_SHIFT) | tokens


def _signal_threshold(iteration: int) -> int:
    """Minimum signal value for CMP_GE — any layer/tokens satisfies it."""
    return iteration << ITER_SHIFT


# ── Reading signal value back to CPU (slow path only) ───────────────────
_cudart = None
_pinned_signal = None


def _get_cudart():
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so")
    return _cudart


def _get_pinned_signal():
    global _pinned_signal
    if _pinned_signal is None:
        _pinned_signal = torch.empty(1, dtype=torch.int64, pin_memory=True)
    return _pinned_signal


def _read_signal_value(sig_buffer) -> tuple:
    """Read 8-byte signal from GPU, sync, unpack → (layer_index, tokens).

    Uses cudaMemcpyAsync D2H from the signal buffer's device pointer to
    pinned host memory.  Only called on the slow path (first MoE layer per
    scheduler step).
    """
    pinned = _get_pinned_signal()
    cudart = _get_cudart()
    sh = torch.cuda.current_stream().cuda_stream
    cudart.cudaMemcpyAsync(
        ctypes.c_void_p(pinned.data_ptr()),
        ctypes.c_void_p(sig_buffer.handle),
        ctypes.c_size_t(8),
        ctypes.c_int(2),   # cudaMemcpyDeviceToHost
        ctypes.c_void_p(sh),
    )
    torch.cuda.current_stream().synchronize()
    val = pinned.item()
    layer_index = (val >> LAYER_SHIFT) & FIELD_MASK
    tokens = val & FIELD_MASK
    return layer_index, tokens


# ── Helpers ──────────────────────────────────────────────────────────────
def _current_stream_handle():
    """Get current CUDA stream handle — safe for multi-GPU processes."""
    return torch.cuda.current_stream().cuda_stream


def _current_sw():
    """Get TorchStreamWrapper for current stream — safe for multi-GPU."""
    return TorchStreamWrapper(torch.cuda.current_stream())


class AttnNvshmemCommunicationHandler:
    """Attention side: sends hidden_states to MoE, receives results back."""

    def __init__(self):
        self.rank = dist.get_rank()
        self.att_tp_size = global_server_args_dict["tp_size"]
        self.ep_size = global_server_args_dict["ep_size"]
        self.moe_workers = list(range(self.att_tp_size, self.att_tp_size + self.ep_size))
        self.num_micro_batch = global_server_args_dict.get("num_micro_batch", 1)

        self.map_att_to_moe = [[] for _ in range(self.att_tp_size)]
        for ep_rank in self.moe_workers:
            self.map_att_to_moe[ep_rank % self.att_tp_size].append(ep_rank)
        self.send_targets = self.map_att_to_moe[self.rank]
        print(f"[AttnNvshmem] rank={self.rank} send_targets={self.send_targets}")

        self.ep_group_info = global_server_args_dict.get("ep_group_info", None)
        self.expected_ep_ranks = self.ep_group_info["send_strategy"].get(self.rank, [])
        print(f"[AttnNvshmem] rank={self.rank} expected_ep_ranks={self.expected_ep_ranks}")

        self.a2e_bufs = global_server_args_dict["nvshmem_a2e_bufs"]
        self.a2e_sigs = global_server_args_dict["nvshmem_a2e_sigs"]
        self.e2a_dst_slots = global_server_args_dict["nvshmem_e2a_dst_slots"]
        self.e2a_sigs = global_server_args_dict["nvshmem_e2a_sigs"]

        self._iteration = {i: 0 for i in range(self.num_micro_batch)}
        self.shape = {}

        print(f"[AttnNvshmem] rank={self.rank} num_micro_batch={self.num_micro_batch}")

    async def send_attention_result(self, batch_id: int, layer_index: int, gpu_hidden_state: torch.Tensor):
        """Send hidden_states via putmem_signal.  Metadata is packed in the
        signal value — no separate header needed.
        Data is staged into NVSHMEM src buffer (required for IB RDMA)."""
        shape = gpu_hidden_state.shape
        self.shape[batch_id] = shape
        tokens = shape[0]

        if DEBUG_NVSHMEM_VERIFY:
            gpu_hidden_state = torch.full_like(gpu_hidden_state, layer_index)

        self._iteration[batch_id] += 1
        flag_val = _pack_signal(self._iteration[batch_id], layer_index, tokens)

        flat = gpu_hidden_state.reshape(-1)
        data_numel = flat.numel()
        data_bytes = data_numel * flat.element_size()

        # Copy into NVSHMEM symmetric buffer — required for IBRC transport
        src_buf = self.a2e_bufs[batch_id]["src"]
        src_buf[:data_numel].copy_(flat)

        dst_buf = self.a2e_bufs[batch_id]["dst"]
        sig = self.a2e_sigs[batch_id]["sig"]
        sh = _current_stream_handle()

        for dst_rank in self.send_targets:
            bindings.putmem_signal_on_stream(
                dst_buf.data_ptr(),
                src_buf.data_ptr(),
                data_bytes,
                sig.handle,
                flag_val,
                bindings.Signal_op.SIGNAL_SET,
                dst_rank, sh,
            )

    async def send_skip_signal(self, batch_id: int):
        """Send a skip signal (tokens=0) — signal-only, no data transfer."""
        self._iteration[batch_id] += 1
        flag_val = _pack_signal(self._iteration[batch_id], 0, 0)

        dst_buf = self.a2e_bufs[batch_id]["dst"]
        sig = self.a2e_sigs[batch_id]["sig"]
        sh = _current_stream_handle()
        for dst_rank in self.send_targets:
            # Zero-byte data; dst_buf address is just a valid placeholder
            bindings.putmem_signal_on_stream(
                dst_buf.data_ptr(),
                dst_buf.data_ptr(),
                0,
                sig.handle,
                flag_val,
                bindings.Signal_op.SIGNAL_SET,
                dst_rank, sh,
            )

    async def recv_moe_result(self, layer_index: int, batch_id: int):
        """Receive MoE results via signal_wait."""
        shape = self.shape[batch_id]
        numel = 1
        for s in shape:
            numel *= s

        self._iteration[batch_id] += 1
        flag_val = self._iteration[batch_id]

        n_peers = len(self.expected_ep_ranks)
        if n_peers == 1:
            sw = _current_sw()
            nvshmem.core.signal_wait(
                self.e2a_sigs[batch_id][0], flag_val,
                bindings.Cmp_type.CMP_GE, stream=sw,
            )
            result = self.e2a_dst_slots[batch_id][0][:numel].reshape(shape)
        else:
            sw = _current_sw()
            accumulator = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
            for i in range(n_peers):
                nvshmem.core.signal_wait(
                    self.e2a_sigs[batch_id][i], flag_val,
                    bindings.Cmp_type.CMP_GE, stream=sw,
                )
                buf = self.e2a_dst_slots[batch_id][i][:numel].reshape(shape)
                accumulator.add_(buf)
            result = accumulator
        return result


class MoENvshmemCommunicationHandler:
    """MoE side: receives hidden_states from attention, sends results back."""

    def __init__(self, config):
        self.hidden_dim = config.hidden_size
        self.rank = dist.get_rank()
        self.att_tp_size = global_server_args_dict["tp_size"]
        self.ep_size = global_server_args_dict["ep_size"]
        self.ep_rank = self.rank - self.att_tp_size
        self.comm_error_threehold = int(os.environ.get("COMM_ERROR_THREEHOLD", 16))
        self.num_micro_batch = global_server_args_dict.get("num_micro_batch", 1)

        self.total_layers = config.num_hidden_layers
        self.last_layer = 0

        self._shutdown_requested = False
        self._shutdown_reason = ""

        self.recv_att_peer = self.rank % self.att_tp_size

        self.ep_group_info = global_server_args_dict.get("ep_group_info", None)
        if self.ep_group_info is None:
            raise ValueError("ep_group_info shouldn't be None")
        self.send_targets = []
        for att_rank, ep_ranks in self.ep_group_info["send_strategy"].items():
            if self.rank in ep_ranks:
                self.send_targets.append(att_rank)
        print(f"[MoENvshmem] rank={self.rank} recv_from={self.recv_att_peer} send_targets={self.send_targets}")

        self.slot_index_at = {}
        for att_rank in self.send_targets:
            peer_list = self.ep_group_info["send_strategy"][att_rank]
            self.slot_index_at[att_rank] = peer_list.index(self.rank)
        print(f"[MoENvshmem] rank={self.rank} slot_index_at={self.slot_index_at}")

        self.a2e_bufs = global_server_args_dict["nvshmem_a2e_bufs"]
        self.a2e_sigs = global_server_args_dict["nvshmem_a2e_sigs"]
        self.e2a_dst_slots = global_server_args_dict["nvshmem_e2a_dst_slots"]
        self.e2a_src = global_server_args_dict["nvshmem_e2a_src"]
        self.e2a_sigs = global_server_args_dict["nvshmem_e2a_sigs"]

        self._iteration = {i: 0 for i in range(self.num_micro_batch)}

        # Header cache: avoid CPU-GPU sync on layers after the first.
        # Within one forward pass (scheduler step), tokens is constant and
        # layer_index is sequential, so we only read the signal value (with
        # stream.synchronize()) on the first MoE layer and derive the rest.
        self._cached_tokens = {}      # batch_id -> token count
        self._next_layer_idx = {}     # batch_id -> next expected layer index

        print(f"[MoENvshmem] rank={self.rank} num_micro_batch={self.num_micro_batch}")

    @property
    def should_shutdown(self):
        return self._shutdown_requested

    def request_shutdown(self, reason=""):
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self._shutdown_reason = reason
            print(f"[MoENvshmem] rank={self.rank} shutdown requested: {reason}")

    def recv_attention_result(self, batch_id):
        """Receive A2E via signal_wait.  Metadata is unpacked from the signal
        value on the first MoE layer; subsequent layers use cached values."""
        self._iteration[batch_id] += 1
        threshold = _signal_threshold(self._iteration[batch_id])

        try:
            sw = _current_sw()
            nvshmem.core.signal_wait(
                self.a2e_sigs[batch_id]["sig"], threshold,
                bindings.Cmp_type.CMP_GE, stream=sw,
            )
        except Exception as e:
            print(f"[MoENvshmem][recv][ERROR] rank={self.rank} err={e}")
            self.request_shutdown(f"signal_wait failed: {e}")
            return None, None

        if batch_id in self._cached_tokens:
            # Fast path: no CPU-GPU sync — use cached metadata
            tokens = self._cached_tokens[batch_id]
            layer_index = self._next_layer_idx[batch_id]
            self._next_layer_idx[batch_id] = layer_index + 1
            # Last MoE layer → clear cache so next step does full unpack
            if layer_index >= self.total_layers - 1:
                del self._cached_tokens[batch_id]
                del self._next_layer_idx[batch_id]
        else:
            # Slow path: first MoE layer — read signal value from GPU
            sig = self.a2e_sigs[batch_id]["sig"]
            layer_index, tokens = _read_signal_value(sig)
            if tokens > 0:
                self._cached_tokens[batch_id] = tokens
                self._next_layer_idx[batch_id] = layer_index + 1

        self.last_layer = layer_index

        if tokens == 0:
            empty = torch.empty(0, self.hidden_dim, dtype=torch.bfloat16, device="cuda")
            return layer_index, empty

        buf = self.a2e_bufs[batch_id]["dst"]
        data_numel = tokens * self.hidden_dim
        hidden_state = buf[:data_numel].reshape(tokens, self.hidden_dim)
        return layer_index, hidden_state

    def send_moe_result(self, layer_index, batch_id, result_state):
        """Send E2A via putmem_signal.  Data staged into NVSHMEM buffer
        (required for IB RDMA — source must be symmetric heap memory)."""
        try:
            self._iteration[batch_id] += 1
            flag_val = self._iteration[batch_id]

            if DEBUG_NVSHMEM_VERIFY:
                result_state = torch.full_like(result_state, layer_index)

            flat = result_state.reshape(-1)
            actual_numel = flat.numel()
            data_bytes = actual_numel * 2

            # Copy into NVSHMEM symmetric buffer — required for IBRC transport
            e2a_src = self.e2a_src[batch_id]
            e2a_src[:actual_numel].copy_(flat)

            sh = _current_stream_handle()
            for dst_rank in self.send_targets:
                slot_idx = self.slot_index_at[dst_rank]
                actual_dst = self.e2a_dst_slots[batch_id][slot_idx]
                bindings.putmem_signal_on_stream(
                    actual_dst[:actual_numel].data_ptr(),
                    e2a_src.data_ptr(), data_bytes,
                    self.e2a_sigs[batch_id][slot_idx].handle,
                    flag_val, bindings.Signal_op.SIGNAL_SET,
                    dst_rank, sh,
                )

        except Exception as e:
            print(f"[MoENvshmem][send][ERROR] rank={self.rank} err={e}")
            self.request_shutdown(f"send failed: {e}")
