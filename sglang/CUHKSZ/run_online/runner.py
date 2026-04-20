import os
import asyncio
import socket
import time
import torch
import torch.distributed as dist
import ucxx
import pprint
# from janus.CUHKSZ.disaggregate.afd_moe.qwen_moe_layer import QwenMoEModel
from sglang.CUHKSZ.disaggregate.afd_moe.deepseek_v2_layer import DeepseekV2Model
from sglang.srt.configs.model_config import ModelConfig


BASE_PORT = 14000
MB = 1024*1024

def get_local_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

def ip2int_list(ip: str):
    return [int(x) for x in ip.split(".")]

def int_list2ip(lst):
    return ".".join(str(x) for x in lst)

class Node:
    def __init__(self, rank, world_size, device="cuda"):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.ucx_peers = {}
        self.listener = None

    async def send_tensor(self, tensor, dst):
        await self.ucx_peers[dst].send(tensor)

    async def recv_tensor(self, src, buf):
        await self.ucx_peers[src].recv(buf)
        return buf


def role_layout(world_size, num_workers, num_servers):
    worker_ranks = list(range(num_workers))
    server_ranks = list(range(num_workers, num_workers + num_servers))
    return worker_ranks, server_ranks

class Scheduler:
    def __init__(self, world_size, worker_ranks, server_ranks):
        self.world_size = world_size
        self.worker_ranks = worker_ranks
        self.server_ranks = server_ranks

    def run(self, rank):
        role = 0 if rank in self.worker_ranks else 1
        ip = get_local_ip()
        port = BASE_PORT + rank
        info_tensor = torch.tensor([rank, role] + ip2int_list(ip) + [port], dtype=torch.int32,
                                   device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        gather_buf = [torch.zeros_like(info_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_buf, info_tensor)

        addr_book = {}
        for t in gather_buf:
            lst = t.cpu().tolist()
            r, role_code = lst[0], lst[1]
            ip_str = int_list2ip(lst[2:6])
            port_num = lst[6]
            addr_book[r] = {"rank": r, "role": "worker" if role_code==0 else "server",
                            "ip": ip_str, "port": port_num}
        if rank == 0:
            pprint.pprint(addr_book)
        return addr_book

