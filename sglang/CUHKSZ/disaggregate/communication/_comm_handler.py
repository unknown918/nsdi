from functools import reduce
import os
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
import ucxx

# from janus.srt.configs.logger_config import configure_logger
from sglang.srt.configs.logger_config import configure_logger
from sglang.srt.distributed import tensor_model_parallel_all_reduce

logger = configure_logger(__name__)


class CommunicationHandler(ABC):
    @abstractmethod
    def send(self, data: Any, dst_rank: int) -> None:
        pass

    @abstractmethod
    def rece(self, data: Any, src_rank: int) -> None:
        pass

    def isend(self, data: Any, dst_rank: int):
        pass

    def irecv(self, data: Any, src_rank: int):
        pass

    @abstractmethod
    def init_distributed(self, *args, **kwargs):
        pass


# class UCXCommunicationHandler(CommunicationHandler):
#     def __init__(self, device: str = "cuda"):
#         # This is a placeholder for the UCX communication handler.
#         # The 'ucp' library will be imported and used in subsequent phases.
#         self.device = device
#         self.rank = 0
#         self.world_size = 0
#
#     def init_distributed(self, *args, **kwargs):
#         pass
#
#     def send(self, data: Any, dst_rank: int) -> None:
#         pass
#
#     def rece(self, src_rank: int) -> Any:
#         pass


class UCXCommunicationHandler(CommunicationHandler):
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.rank = None
        self.world_size = None
        self.endpoints: Dict[int, ucxx.Endpoint] = {}
        self.listener = None
        self._initialized = False
        self.role = None
        self.m_ranks = []
        self.n_ranks = []

    def init_distributed(
        self, rank, world_size, m_ranks, n_ranks, master_addr, master_port, **kwargs
    ):
        # For one-time initialization, asyncio.run() is acceptable.
        # The main application logic should manage its own event loop for communication.
        asyncio.run(
            self._init_distributed_async(
                rank, world_size, m_ranks, n_ranks, master_addr, master_port
            )
        )

    async def _init_distributed_async(
        self, rank, world_size, m_ranks, n_ranks, master_addr, master_port
    ):
        if self._initialized:
            return

        self.rank = rank
        self.world_size = world_size
        self.m_ranks = m_ranks
        self.n_ranks = n_ranks

        if self.rank in self.m_ranks:
            self.role = "m"
        elif self.rank in self.n_ranks:
            self.role = "n"
        else:
            raise ValueError(f"Rank {self.rank} is not in m_ranks or n_ranks")

        # Bootstrap using torch.distributed with 'gloo' backend
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        ucxx.init()

        address_tensor = torch.zeros(40, dtype=torch.uint8)

        if self.role == "n":  # Server
            async def server_handler(endpoint):
                # Client sends its rank as the first message
                rank_buffer = bytearray(4)
                await endpoint.recv(rank_buffer)
                client_rank = int.from_bytes(rank_buffer, "little")
                self.endpoints[client_rank] = endpoint
                logger.info(
                    f"UCX Server rank {self.rank} connected with client rank {client_rank}"
                )

            self.listener = await ucxx.create_listener(server_handler)
            ip = socket.gethostbyname(socket.gethostname())
            addr_str = f"{ip}:{self.listener.port}"
            address_tensor = torch.tensor(
                list(map(ord, addr_str.ljust(40))), dtype=torch.uint8
            )

        # Exchange addresses
        all_addresses_tensor_list = [
            torch.zeros_like(address_tensor) for _ in range(world_size)
        ]
        dist.all_gather(all_addresses_tensor_list, address_tensor)

        server_addresses = {}
        for r, addr_tens in enumerate(all_addresses_tensor_list):
            if r in self.n_ranks:
                addr_str = "".join(map(chr, addr_tens)).strip()
                if addr_str:
                    ip, port_str = addr_str.split(":")
                    server_addresses[r] = (ip, int(port_str))

        if self.role == "m":  # Client
            for server_rank, (ip, port) in server_addresses.items():
                ep = await ucxx.create_endpoint(ip, port)
                self.endpoints[server_rank] = ep
                # Send my rank to identify myself
                rank_bytes = self.rank.to_bytes(4, "little")
                await ep.send(bytearray(rank_bytes))
                logger.info(
                    f"UCX Client rank {self.rank} connected to server rank {server_rank}"
                )

        dist.barrier()
        self._initialized = True
        logger.info(f"UCXCommunicationHandler initialized for rank {self.rank}")

    def send(self, data: torch.Tensor, dst_rank: int):
        # Blocking send is an anti-pattern for high-performance async networking.
        # The calling code should be refactored to use the async 'isend' method.
        raise NotImplementedError(
            "Blocking 'send' is not supported. Use 'await isend(...)' instead."
        )

    def rece(self, data: torch.Tensor, src_rank: int):
        # Blocking receive is an anti-pattern for high-performance async networking.
        # The calling code should be refactored to use the async 'irecv' method.
        raise NotImplementedError(
            "Blocking 'rece' is not supported. Use 'await irecv(...)' instead."
        )

    async def isend(self, data: torch.Tensor, dst_rank: int):
        ep = self.endpoints.get(dst_rank)
        if not ep:
            raise RuntimeError(
                f"Endpoint to rank {dst_rank} not found for rank {self.rank}."
            )
        await ep.send(data)

    async def irecv(self, buffer: torch.Tensor, src_rank: int):
        if self.role == "n":  # Server, wait for client to connect if not already
            while src_rank not in self.endpoints:
                await asyncio.sleep(0.01)  # busy wait for connection

        ep = self.endpoints.get(src_rank)
        if not ep:
            raise RuntimeError(
                f"Endpoint from rank {src_rank} not found for rank {self.rank}."
            )
        await ep.recv(buffer)


class TorchDistCommunicationHandler(CommunicationHandler):
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.rank = 0
        self.world_size = 0
        self.att_tp_size = 0
        self.ep_size = 0
        
    def send(self, data, dst_rank):
        dist.send(data, dst=dst_rank)

    def rece(self, data, src_rank):
        dist.recv(data, src=src_rank)
    
    def irecv(self, data, src_rank):
        dist.irecv(data, src=src_rank)
        
    def isend(self, data, dst_rank):
        dist.isend(data, dst=dst_rank)

    # TODO m->n (wy): after m->n: 把send_attention_result的逻辑抽象成一个函数写在TorchDistCommunicationHandler中

    def batch_isend(self, data1, data2, data3, dst_rank):
        req_ops = [
            dist.P2POp(dist.isend, data1, peer=dst_rank),
            dist.P2POp(dist.isend, data2, peer=dst_rank),
            dist.P2POp(dist.isend, data3, peer=dst_rank)
        ]
        reqs = dist.batch_isend_irecv(req_ops)
        for req in reqs:
            req.wait()

    def batch_irecv(self, data1, src_rank):
        req_ops = [
            dist.P2POp(dist.irecv, data1, peer=src_rank),
        ]               
        #print(f"[DEBUG] batch_irecv: rank {self.rank}, src_rank: {src_rank}, req_ops: {req_ops}")
        reqs = dist.batch_isend_irecv(req_ops)
        for req in reqs:
            req.wait()
        #print(f"[DEBUG] batch_irecv: rank {self.rank}, src_rank: {src_rank}")
    
    def init_distributed(self, backend, init_method, world_size, rank):
        logger.CommunicationHandler(f"init torch PG begin, world_size={world_size}, rank={rank}, init_method={init_method}")

        current_dir = os.path.dirname(__file__)
        from dotenv import load_dotenv
        ib_env_path = os.path.join(current_dir, "../userConfig/ib.env")
        load_dotenv(ib_env_path)          

        print("[DEBUG] init_distributed_environment begin, backend={}, init_method={}, world_size={}, rank={}".format(backend, init_method, world_size, rank))
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,           
            rank=rank,
        )    
        logger.CommunicationHandler(f"init torch PG success, world_size={world_size}, rank={rank}, init_method={init_method}")

class AttentionCommunicator:    
    def __init__(self, comm_handler: CommunicationHandler):        
        self.communicator = comm_handler
        self.rank = dist.get_rank()        
        
        # TODO m->n (wy): 这里暂时写死了从环境变量读，写完m2n之后要统一所有的attn/moe把从环境变量读改为从config读（涉及到调用链上面的很多处修改，需要看在哪里init）
        self.att_dp_size = int(os.environ['ATTN_WORKERS_COUNT'])
        self.ep_size = int(os.environ['MOE_WORKERS_COUNT'])
        self.attn_workers = [i for i in range(self.att_dp_size)]
        self.moe_workers = [i for i in range(self.att_dp_size, self.att_dp_size + self.ep_size)]  
        # TODO m->n (wy): 给AttentionCommunicator初始化的时候添加一个tensor_pool
        self.tensor_pool = None
        
    def send_attention_result(self, metadata: torch.Tensor, gpu_hidden_state: torch.Tensor, gpu_topk_ids: torch.Tensor, gpu_topk_weights: torch.Tensor):
                            
            # step 1: send metadata
            logger.CommunicationHandler(f"send_attention_result before send 1: rank {self.rank}, moe_workers: {self.moe_workers}")
            
            # torch.cuda.synchronize()
            # timestamp = time.time()
            # logger.TimeStamp(f"att send meta: {timestamp}   ")      

            # TODO m->n (wy): after m->n: 把send_attention_result的逻辑抽象成一个函数写在TorchDistCommunicationHandler中
            # TODO m->n (wy): 可以抽象出一个moe rank和global rank的转换函数
            # TODO m->n (wy): 这个metadatalist每次创建是不是会有很多额外耗时，用pool来管理？
            dst_metadata_list = []
            for dst in self.moe_workers:
                only_moe_rank = dst - self.att_dp_size
                dst_metadata = metadata.clone()
                # 如果没有token需要处理，则设置为0
                dst_metadata[3] = gpu_topk_ids[only_moe_rank].shape[0] if only_moe_rank in gpu_topk_ids else 0
                dst_metadata_list.append(dst_metadata)
                # logger.CommunicationHandler(f"准备发送到 {dst}: metadata: {dst_metadata_list[-1]}")

            # 创建发送操作列表，避免出现异步发送但是引用同一个metadata的问题    
            send_ops = []
            for i, dst in enumerate(self.moe_workers):
                send_ops.append(dist.P2POp(dist.isend, dst_metadata_list[i], peer=dst))

            reqs = dist.batch_isend_irecv(send_ops)
            for req in reqs:
                req.wait()
            logger.CommunicationHandler(f"send_attention_result after send 1: rank {self.rank} sync")

            # torch.cuda.synchronize()
            # timestamp = time.time()
            # logger.TimeStamp(f"att send data: {timestamp}   ")      
            
            # # step 2: send real data
            logger.CommunicationHandler(f"send_attention_result before send 2: rank {self.rank}")
            # TODO 5.11 (wy): 改成发给多个moe节点   
            send_ops = []
            # 只遍历gpu_topk_ids中有的moe节点，并且转换成global rank
            for dst in gpu_topk_ids.keys():
                global_moe_rank = dst + self.att_dp_size
                send_ops.append(dist.P2POp(dist.isend, gpu_hidden_state[dst], peer=global_moe_rank))
                send_ops.append(dist.P2POp(dist.isend, gpu_topk_ids[dst], peer=global_moe_rank))
                send_ops.append(dist.P2POp(dist.isend, gpu_topk_weights[dst], peer=global_moe_rank))
                logger.CommunicationHandler(f"loop send to {global_moe_rank}: send_attention_result: rank {self.rank}, dst: moe {dst}")
            reqs = dist.batch_isend_irecv(send_ops)
            for req in reqs:
                req.wait()
            logger.CommunicationHandler(f"send_attention_result after send 2: rank {self.rank} sync")
    
    def send_attention_result_without_gate(self, metadata: torch.Tensor, gpu_hidden_states: Dict[int, torch.Tensor]):
        """
        Send attention results to MoE workers without gate information.
        This method sends metadata and hidden states to all MoE workers for full data transfer.
        
        Args:
            metadata: Tensor containing [layer_idx, hidden_dim, topk, num_tokens]
            gpu_hidden_states: Dictionary mapping GPU IDs to hidden state tensors
        """
        # step 1: send metadata to all MoE workers
        logger.CommunicationHandler(f"send 1: rank {self.rank}, moe_workers: {self.moe_workers}")
        
        # Create metadata for each MoE worker
        dst_metadata_list = []
        for dst in self.moe_workers:
            only_moe_rank = dst - self.att_dp_size
            dst_metadata = metadata.clone()
            # Set the number of tokens to process for this MoE worker
            # Since we're doing full data transfer, all workers get the same number of tokens
            dst_metadata[3] = metadata[3]  # num_tokens
            dst_metadata_list.append(dst_metadata)

        # Send metadata to all MoE workers
        send_ops = []
        for i, dst in enumerate(self.moe_workers):
            send_ops.append(dist.P2POp(dist.isend, dst_metadata_list[i], peer=dst))

        reqs = dist.batch_isend_irecv(send_ops)
        for req in reqs:
            req.wait()
        # step 2: send hidden states to all MoE workers
        logger.CommunicationHandler(f"send 2")
        
        send_ops = []
        # Send hidden states to all MoE workers (full data transfer)
        
        for dst in self.moe_workers:
            print(f"dst: {dst}")
            hidden_state = gpu_hidden_states[dst - self.att_dp_size] 
            send_ops.append(dist.P2POp(dist.isend, hidden_state, peer=dst))   
            print(f"hidden_state: {hidden_state.shape}, to rank {dst}")
        
        reqs = dist.batch_isend_irecv(send_ops)
        for req in reqs:
            req.wait()
        logger.CommunicationHandler(f"send 3: finished")
    
    def parse_list(self, hidden_states: Dict[int, torch.Tensor]):
        new_dict = {}
        if self.tensor_pool:
            pass
        else:
            new_dict = {key: torch.zeros(tensor.shape, device="cuda", dtype=torch.bfloat16) 
            for key, tensor in hidden_states.items()}
        return new_dict


    def rece_moe_result(self, hidden_states: Dict[int, torch.Tensor]):  
        moe_result = self.parse_list(hidden_states)
        recv_ops = []
        for src in hidden_states.keys():
            global_moe_rank = src + self.att_dp_size
            recv_ops.append(dist.P2POp(dist.irecv, moe_result[src], peer=global_moe_rank))
            # logger.CommunicationHandler(f"rece_moe_result: rank {self.rank}, src: {src}, global_moe_rank: {global_moe_rank}")
        reqs = dist.batch_isend_irecv(recv_ops)
        for req in reqs:
            req.wait()

        # torch.cuda.synchronize()
        # timestamp = time.time()
        # logger.TimeStamp(f"att rece data: {timestamp}   ")      

        logger.CommunicationHandler(f"rece_moe_result: rank {self.rank}")

        # moeResult = torch.randn(shape[0], shape[1], dtype=torch.bfloat16).cuda()
        # self.communicator.rece(moeResult, src_rank)

        return moe_result


class MoeCommunicator:    
    def __init__(self, comm_handler: CommunicationHandler, tensor_pool=None):        
        self.communicator = comm_handler
        self.rank = 0        
        # TODO m->n (wy): after m->n: 这里暂时写死了从环境变量读，写完m2n之后要统一所有的attn/moe把从环境变量读改为从config读（涉及到调用链上面的很多处修改，需要看在哪里init）
        self.att_dp_size = int(os.environ['ATTN_WORKERS_COUNT'])
        self.ep_size = int(os.environ['MOE_WORKERS_COUNT'])
        self.attn_workers = [i for i in range(self.att_dp_size)]
        self.moe_workers = [i for i in range(self.att_dp_size, self.att_dp_size + self.ep_size)]  

        # TODO m->n (wy): m -> n这里接收参数需要修改成一个list[list[int]]，可以不放init里面
        # TODO m->n (wy): 或者用metadata存当前对应的metadata，但是再开一个list存每个attn节点需要的top_k expert以及需要处理的token数量
        self.metadata = torch.zeros(4, dtype=torch.int32, device="cuda")
        logger.CommunicationHandler(f"MoeCommunicator init: metadata: {self.metadata}, self.rank: {self.rank}")
        self.tensor_pool = tensor_pool
    
    def parse_metadata(self):
        layer_idx, hidden_dim, topk, num_tokens = self.metadata
        logger.CommunicationHandler(f"layer_idx: {layer_idx}, hidden_dim: {hidden_dim}, topk: {topk}, num_tokens: {num_tokens}")

        # use tensor pool to create tensors
        # TODO m->n (wy): 这里创建tensor需要修改，得依据list建立三个巨大tensor
        if self.tensor_pool:
            hidden_state = self.tensor_pool.get_tensor((num_tokens, hidden_dim), torch.bfloat16)
            topk_ids = self.tensor_pool.get_tensor((num_tokens, topk), torch.int32)
            topk_weights = self.tensor_pool.get_tensor((num_tokens, topk), torch.bfloat16)
        else:
            hidden_state = torch.zeros((num_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
            topk_ids = torch.zeros((num_tokens, topk), dtype=torch.int32, device="cuda")
            topk_weights = torch.zeros((num_tokens, topk), dtype=torch.bfloat16, device="cuda")
            
        return layer_idx, hidden_state, topk_ids, topk_weights
    
    def rece_attention_result(self, src_rank: int):        
    #         print(f"rece_attn: rank {self.rank}, torch.cuda.current_device()={torch.cuda.current_device()}")
    #         logger.CommunicationHandler(f"rcee_attn: rank {self.rank}, torch.cuda.current_device()={torch.cuda.current_device()}")              
            
            # TODO for wy
            # 你需要接收多个attention传过来的结果
            # 并直接用三个足够大的tensor来接收，hidden_state, topk_ids, topk_weights
            #             
            # receive metadata
            # self.communicator.rece(self.metadata, src_rank)
            # logger.CommunicationHandler(f"rece_attention_result before 1st recv: metadata: {self.metadata}, self.rank: {self.rank}")

            # TODO m->n (wy): 这里需要修改!!需要改成用一个list接收，当前只是一个写死attn 0的写法
            recv_ops = []
            for src in self.attn_workers:
                recv_ops.append(dist.P2POp(dist.irecv, self.metadata, peer=src))
            reqs = dist.batch_isend_irecv(recv_ops)
            for req in reqs:
                req.wait()

            # torch.cuda.synchronize()
            # timestamp = time.time()
            # logger.TimeStamp(f"moe rece meta: {timestamp}   ")              
            logger.CommunicationHandler(f"rece 1, rank {self.rank}")

            # TODO m->n (wy): 修改成对于list中metadata[3]的判断
            if self.metadata[3] == 0:
                return "FREE_COMPUTE"
            
            # else: do real data recv

            # TODO m->n (wy): 修改parse_metadata，用三个足够大的tensor来接收来自多个attn节点的hidden_state, topk_ids, topk_weights
            layer_idx, hidden_state, topk_ids, topk_weights = self.parse_metadata()

            # print(f"rece_attention_result before 2nd recv: hidden_state: {hidden_state.shape}, layer_idx: {layer_idx}")
            # logger.CommunicationHandler(f"rece_attention_result before 2nd recv: hidden_state: {hidden_state.shape}, topk_ids: {topk_ids.shape}, topk_weights: {topk_weights.shape}, self.rank: {self.rank}")
            # receive hidden_state and router_logits
            # 清零
            self.metadata = torch.zeros(4, dtype=torch.int32, device="cuda")
            # logger.CommunicationHandler(f"rece_attention_result metadata after clear: {self.metadata}")
            
            # TODO m->n (wy): 修改，当前的实现是从一个attn节点接收
            #print(f"[DEBUG] rece 2: hidden_state: {hidden_state.shape}, topk_ids: {topk_ids.shape}, topk_weights: {topk_weights.shape}")
            self.communicator.batch_irecv(hidden_state, src_rank)

            # torch.cuda.synchronize()
            # timestamp = time.time()
            # logger.TimeStamp(f"moe rece data: {timestamp}   ")                  
            logger.CommunicationHandler(f"rece 2, layer_idx: {layer_idx}, hidden_state: {hidden_state.shape}")
            data = {"layer_idx": layer_idx, "hidden_state": hidden_state}
            return data
        
    def send_moe_result(self, data: Any, dst_rank: int):
         # TODO for wy
            # 这里你已经拿到了对上面你拼接起来的大数据块的计算结果了！
            # 你需要按照你接收时候的拆分方法，把他们再逐个发回去   

        # TODO m->n (wy): 改成只给当前moe节点发过real data的attn节点发送moe结果，而不是全部attn节点
        send_ops = []
        for dst in self.attn_workers:
            send_ops.append(dist.P2POp(dist.isend, data, peer=dst))
        reqs = dist.batch_isend_irecv(send_ops)

        # torch.cuda.synchronize()
        # timestamp = time.time()
        # logger.TimeStamp(f"moe send data: {timestamp}   ")      

        for req in reqs:
            req.wait()        
        logger.CommunicationHandler(f"send 3: finished")

    def activate_default_PG(self, role = "receiver"):
         # TODO m->n (wy): 这里还默认了只从rank 0接收，如果不改多个attn的时候会出问题，也要改！！
        current_device_id = torch.cuda.current_device()
        # sync_data = torch.randn(1, 1, dtype=torch.float16).cuda() 
        sync_data = torch.zeros(1, dtype=torch.float16, device="cuda")
        if role == "receiver":
            reqs = dist.batch_isend_irecv([dist.P2POp(dist.irecv, sync_data, peer=0)])
        else:
            reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, sync_data, peer=0)])
        for req in reqs:
            req.wait()
        #print("[DEBUG] activate_default_PG sync")
        logger.CommunicationHandler(f"activate_default_PG: rank {self.rank} Sync, current_device_id: {current_device_id}")

    def init_moe_distributed(self, dist_init_method, att_tp_size, ep_size, gpu_local_id, gpu_global_id):                
        torch.cuda.set_device(gpu_local_id)
                               
        world_size = att_tp_size + ep_size
        rank = att_tp_size + gpu_global_id
        
        self.world_size = world_size
        self.att_tp_size = att_tp_size
        self.ep_size = ep_size

        self.communicator.init_distributed(backend="nccl", init_method=dist_init_method, world_size=world_size, rank=rank)

        self.rank = dist.get_rank()
        #print("[DEBUG] activate_default_PG begin")
        self.activate_default_PG(role="receiver")
        #print("[DEBUG] activate_default_PG end")

        logger.CommunicationHandler(f"rank {rank} Sync")