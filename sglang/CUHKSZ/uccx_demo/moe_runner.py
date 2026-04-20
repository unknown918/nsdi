import threading
import queue
import os
import gc
import json
import logging
import time
import os
from typing import List, Optional, Tuple, Dict
import queue
import threading
import signal
import sys
import zmq
import time
import numpy as np

import torch
import torch.distributed as dist

current_dir = os.path.dirname(__file__)
 

class MoeRunner:
    def __init__(
        self,        
        device: str,        
        gpu_local_id: int,
        gpu_global_id: int,
    ):        
        self.device = device
        self.moeLayers = None
        self.gpu_local_id = gpu_local_id
        self.gpu_global_id = gpu_global_id                        
        print(f"gpu_global_id={self.gpu_global_id}")
        print(f"set device to cuda:{self.gpu_local_id}")
        torch.cuda.set_device(f"cuda:{self.gpu_local_id}")
                            
    def init_distributed_environment(self, dist_init_method, dp_size, ep_size, gpu_local_id, gpu_global_id):        
        att_tp_size = dp_size        
        torch.cuda.set_device(gpu_local_id)
                               
        world_size = att_tp_size + ep_size
        rank = att_tp_size + gpu_global_id                        
        
        nccl_master_ip = os.getenv("NCCL_MASTER_IP")
        nccl_master_port = os.getenv("NCCL_MASTER_PORT")
        dist_init_method = f"tcp://{nccl_master_ip}:{nccl_master_port}"
    
        print(f"init_distributed_environment begin, init_method={dist_init_method}, world_size={world_size}, rank={rank}")
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=dist_init_method,
            world_size=world_size,           
            rank=rank,
        )            
        print(f"init_distributed_environment success, init_method={dist_init_method}, world_size={world_size}, rank={rank}")
        
        # 获取当前设备
        # current_device = torch.cuda.current_device()        
        # tensor = torch.tensor([1], device=f"cuda:{current_device}")
                
        # world_size=att_tp_size + ep_size
        # tensor = torch.tensor([1], device=f"cuda:{current_device}")
        # tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        # dist.all_gather(tensor_list, tensor)
        
        
        data = torch.tensor([rank], dtype=torch.float32).cuda()
# 存放所有 rank 的结果
        gather_list = [torch.zeros_like(data) for _ in range(3)]
# 执行 all_gather
        dist.all_gather(gather_list, data)
        # reduce
        # dist.reduce(data, dst=0)
        print(f"data={data}")