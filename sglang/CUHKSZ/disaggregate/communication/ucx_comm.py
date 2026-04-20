from cgitb import reset
from functools import reduce
import os
from re import U
import threading
import queue
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
import time
import socket
import asyncio

from sglang.srt.distributed.parallel_state import get_ep_group
# from janus.srt.configs.logger_config import configure_logger
from sglang.srt.configs.logger_config import configure_logger
from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.managers.schedule_batch import global_server_args_dict

import numpy as np
import cupy as cp
from torch.utils.dlpack import to_dlpack
from torch.utils.dlpack import from_dlpack


logger = configure_logger(__name__)

DEBUG_UCX_VERIFY = os.environ.get("DEBUG_UCX_VERIFY", "0") == "1"

class AttnUCXCommunicationHandler:
    def __init__(self):
        self.rank = dist.get_rank()
        self.ucx_peers = global_server_args_dict['ucx_peers']
        print(f"[AttnUCXCommunicationHandler __init__] ucx_peers: {self.ucx_peers}")
        self.att_tp_size = global_server_args_dict['tp_size']
        self.ep_size = global_server_args_dict['ep_size']        
        self.attn_workers = [i for i in range(self.att_tp_size)]
        self.moe_workers = [i for i in range(self.att_tp_size, self.att_tp_size + self.ep_size)]                
        self.shape = {}
                
        # ⭐️ each attention only need to send results to some moe worker as their output are same        
        self.map_att_to_moe = [[] for _ in range(self.att_tp_size)]
        for ep_rank in self.moe_workers:
            self.map_att_to_moe[ep_rank % self.att_tp_size].append(ep_rank)            
        print(f"⭐️ attention rank {self.rank} need to send results to moe workers: ", self.map_att_to_moe)

        # ⭐️ each attention only need to recv results from some ep ranks as their output are same if already reduced result
        self.ep_group_info = global_server_args_dict.get("ep_group_info", None)        
        self.expected_ep_ranks = self.ep_group_info['send_strategy'].get(self.rank, [])
        print(f"[AttnUCXCommunicationHandler] rank={self.rank} expected_ep_ranks={self.expected_ep_ranks}")
                    
    async def send_attention_result(self, batch_id: int, layer_index: int, gpu_hidden_state: torch.Tensor):            
                
        # First send: send metadata (layer_index and shape)
        shape = gpu_hidden_state.shape
        self.shape[batch_id] = shape        
        tokens = shape[0]
        assert tokens != 0, "[AttnUCX] tokens should not be zero"

        # use cpu numpy for header to avoid cuda async issue
        header = np.array([layer_index, tokens], dtype=np.int64)
        targets = self.map_att_to_moe[self.rank]                                
        
        # logger.Model(f"[batch {batch_id}] att {self.rank} send meta to {targets} content {header.tolist()}")
        meta_send_tasks = [self.ucx_peers[dst][batch_id].send(header) for dst in targets]
        await asyncio.gather(*meta_send_tasks)

        # fake a layer idx hidden_states to verify
        if DEBUG_UCX_VERIFY:
            gpu_hidden_state = torch.full_like(gpu_hidden_state, layer_index)

        # send tensor
        torch.cuda.current_stream().synchronize()
        data_send_tasks = [self.ucx_peers[dst][batch_id].send(gpu_hidden_state) for dst in targets]
        await asyncio.gather(*data_send_tasks)

    async def recv_moe_result(self, layer_index: int, batch_id: int):                                
        expected_ep_ranks = self.expected_ep_ranks
        accumulator = torch.zeros(self.shape[batch_id], dtype=torch.bfloat16, device="cuda")
        staging_buffers = [
            torch.empty(self.shape[batch_id], dtype=torch.bfloat16, device="cuda")
            for _ in expected_ep_ranks
        ]
        
        async def recv_and_accumulate(peer_rank: int, buf: torch.Tensor):
            try:
                await self.ucx_peers[peer_rank][batch_id].recv(buf)
                accumulator.add_(buf)
            except Exception as e:
                print(f"[Attn][recv][ERROR] rank={self.rank} peer={peer_rank} err={e}")
                raise
                
        await asyncio.gather(*[recv_and_accumulate(peer, staging_buffers[idx]) for idx, peer in enumerate(expected_ep_ranks)])
        
        if DEBUG_UCX_VERIFY:
            expected = torch.tensor(layer_index, dtype=accumulator.dtype, device=accumulator.device)
            wrong_count = (accumulator != expected).sum()
            total = accumulator.numel()
            logger.Model(f"[UCX_VERIFY] layer={layer_index} wrong={wrong_count}/{total}")

        return accumulator


class MoEUCXCommunicationHandler:    
    def __init__(self, hidden_dim=5120): # 5120 Only for deepseek2
        self.hidden_dim = hidden_dim
        self.rank = dist.get_rank()
        self.att_tp_size = global_server_args_dict['tp_size']
        self.ep_size = global_server_args_dict['ep_size']
        self.ucx_peers = global_server_args_dict['ucx_peers']
        self.comm_error_threehold = int(os.environ.get("COMM_ERROR_THREEHOLD", 16))
        print(f"[MoEUCXCommunicationHandler __init__] ucx_peers: {self.ucx_peers}")
        self.ep_rank = self.rank - self.att_tp_size
        self.error_time = 0
        self.attn_workers = [i for i in range(self.att_tp_size)]
        self.moe_workers = [i for i in range(self.att_tp_size, self.att_tp_size + self.ep_size)]

        self.total_layers = int(os.environ.get("NUM_HIDDEN_LAYERS", 60))
        self.last_layer = 0

        # shutdown flags
        self._shutdown_requested = False
        self._shutdown_reason = ""
        
        # ⭐️ each ep only need to send results to some attn ranks as their output are same if already reduced result
        self.ep_group_info = global_server_args_dict.get("ep_group_info", None)        
        if self.ep_group_info is None: raise ValueError("ep_group_info shouldnt be None")        
        self.send_targets = []
        for att_rank, ep_ranks in self.ep_group_info['send_strategy'].items():
            if self.rank in ep_ranks:
                self.send_targets.append(att_rank)
        print(f"[MoEUCXCommunicationHandler] rank={self.rank} send_targets={self.send_targets}")     
        
        print(f"⭐️ moe rank {self.rank} need to recv results from attention workers: ", self.rank % self.att_tp_size)              

    @property
    def should_shutdown(self):
        return self._shutdown_requested
    
    def request_shutdown(self, reason=""):
        if not self._shutdown_requested:
            self._shutdown_requested = True
            self._shutdown_reason = reason
            print(f"[MoE] rank={self.rank} shutdown requested: {reason}")

    async def recv_attention_result(self, batch_id):                                
        # First receive: get metadata
        # Assuming max shape dimensions is 8, so header size is 1 (idx) + 1 (ndim) + 8 (dims) = 10
        
        recv_att_peers = self.rank % self.att_tp_size

        header = np.zeros(2, dtype=np.int64)
        # meta_recv_tasks = [self.ucx_peers[recv_att_peers].recv(head_tensor)]
        # await asyncio.gather(*meta_recv_tasks)        
        try:
            meta_recv_tasks = [self.ucx_peers[recv_att_peers][batch_id].recv(header)]
            await asyncio.gather(*meta_recv_tasks)
            # print(f"[MoE][recv] rank={self.rank} header ok: vals={head_tensor.tolist()}")
        except Exception as e:
            print(f"[MoE][recv][ERROR] rank={self.rank} header from {recv_att_peers} err={e}")
            self.request_shutdown(f"recv header failed from {recv_att_peers}: {e}")
            return None, None

        # Parse headers and prepare for second receive         
        idx = int(header[0])
        tokens = int(header[1])
        flag = -1 
        if tokens == 0: 
            flag = 0            
            tokens = self.comm_error_threehold
            idx = (self.last_layer + 1) % self.total_layers
        
        self.last_layer = idx
        shape = (tokens, self.hidden_dim)        
        # Pre-allocate tensor for receiving data
        # hidden_state = torch.empty(shape, dtype=torch.bfloat16, device="cuda")        
        hidden_state = torch.full(shape, -1, dtype=torch.bfloat16, device="cuda")        

        data_recv_tasks = [self.ucx_peers[recv_att_peers][batch_id].recv(hidden_state)]        
        # logger.Model(f"[batch {batch_id}] layer_index {idx} recv moe result from attn rank {recv_att_peers} tokens {tokens}")    
        # Second receive: get the actual tensor data
        # await asyncio.gather(*data_recv_tasks)
        try:
            await asyncio.gather(*data_recv_tasks)
            # print(f"[MoE][recv] rank={self.rank} tensor ok from {recv_att_peers} min={hidden_state.min().item() if hidden_state.numel()>0 else 'n/a'} max={hidden_state.max().item() if hidden_state.numel()>0 else 'n/a'}")
        except Exception as e:
            print(f"❌ [MoE][recv][ERROR] rank={self.rank} tensor from {recv_att_peers} err={e}")
            print(f"Error ❌ {self.rank} recv meta from {recv_att_peers} idx {idx} tokens {tokens} shape {shape}")
            self.request_shutdown(f"recv tensor failed from {recv_att_peers}: {e}")
            return None, None

        if DEBUG_UCX_VERIFY:
            expected = torch.tensor(idx, dtype=hidden_state.dtype, device=hidden_state.device)
            wrong_count = (hidden_state != expected).sum()
            total = hidden_state.numel()
            if self.ep_rank == 0:
                print(f"[UCX_VERIFY] layer={idx} batch={batch_id} wrong={wrong_count}/{total}")
                
        return idx, hidden_state                    
    
    async def send_moe_result(self, layer_index, batch_id, result_state):
        try:
            # 使用预计算的 send_targets，避免每次查找
            send_targets = self.send_targets
            # print(f"[MoE][send] rank={self.rank} -> targets={send_targets}")
            if DEBUG_UCX_VERIFY:
                result_state = torch.full_like(result_state, layer_index)           
            
            # print(f"*** rank={self.rank} send moe result to att {result_state.shape}")
            torch.cuda.current_stream().synchronize()
            data_send_tasks = [self.ucx_peers[dst][batch_id].send(result_state) for dst in send_targets]
            await asyncio.gather(*data_send_tasks)
                
        except Exception as e:
            print(f"[MoE][send][ERROR] rank={self.rank} err={e}")
            self.request_shutdown(f"send failed to {self.send_targets}: {e}")
        # data_send_tasks = [self.ucx_peers[dst].send(result_state) for dst in self.attn_workers]
        # await asyncio.gather(*data_send_tasks)