import os
import asyncio
import socket
import time
import torch
import torch.distributed as dist
import ucxx
# import pprint
# from janus.CUHKSZ.disaggregate.afd_moe.qwen_moe_layer import QwenMoEModel
from sglang.CUHKSZ.disaggregate.afd_moe.deepseek_v2_layer import DeepseekV2Model
from sglang.srt.configs.model_config import ModelConfig
from runner import Node


class MoeRunner(Node):
    def __init__(self, rank, world_size, attn_ranks, model_path):
        super().__init__(rank, world_size)
        self.world_size = world_size
        self.attn_ranks = attn_ranks
        self.model_config = ModelConfig(
            model_path,
            model_override_args="{}"
        )
        self.model_config.hf_config.num_hidden_layers = 10    
        self.load_moe()

    async def connect_workers(self, my_port):
        print(f"[MoeRunner {self.rank}] listening on port {my_port}")
        async def handler(ep):
            buf = torch.empty((1,), dtype=torch.int32, device=self.device)
            await ep.recv(buf)
            worker_rank = int(buf.item())
            self.ucx_peers[worker_rank] = ep
            print(f"[MoeRunner {self.rank}] connected to Worker {worker_rank}")
        self.listener = ucxx.create_listener(handler, port=my_port)
        while len(self.ucx_peers) < len(self.attn_ranks):
            await asyncio.sleep(0.1)
        print(f"[MoeRunner {self.rank}] all workers connected")
        
    async def handle_request(self, worker_rank):
        buf = torch.empty((1024,), dtype=torch.float32, device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        while True:
            await self.recv_tensor(worker_rank, buf, stream)
            request_data = self.parse_request(buf)
            output = await self.process_request(request_data)
            await self.send_tensor(output, worker_rank, stream)        

    async def run(self, my_ip, my_port):
        await self.connect_workers(my_port)
        tasks = [self.handle_request(worker_rank) for worker_rank in self.ucx_peers]
        await asyncio.gather(*tasks)
        
    def load_moe(self):
        rank = dist.get_rank()
        ep_gpu_used = self.world_size - len(self.attn_ranks)        
        from sglang.CUHKSZ.disaggregate.managers.slot_manager import SlotManager
        slot_manager = SlotManager(self.model_config, ep_gpu_used, 1, 1)
        gpu_slots = slot_manager.get_gpu_slots(rank)
        self.model = DeepseekV2Model(
            self.model_config.hf_config,
            gpu_slots=gpu_slots,
            ep_gpu_used=ep_gpu_used,
            gpu_global_id=rank
        )
        self.model.load_weights(self.model_config.model_path)
        print("[DEBUG] moe model loaded")