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

# from janus.srt.configs.logger_config import configure_logger
from sglang.srt.configs.logger_config import configure_logger
from sglang.srt.distributed import tensor_model_parallel_all_reduce

logger = configure_logger(__name__)

class CommunicationHandler(ABC):    
    @abstractmethod
    def send(self, data: Any, dst_rank: int) -> None: pass
    
    @abstractmethod
    def rece(self, src_rank: int) -> Any: pass      

    @abstractmethod
    def init_distributed(self): pass

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
    def __init__(self, comm_handler: CommunicationHandler, ucx_peers: dict):        
        self.communicator = comm_handler
        self.rank = dist.get_rank()        
        self.ucx_peers = ucx_peers
        
        # TODO m->n (wy): 这里暂时写死了从环境变量读，写完m2n之后要统一所有的attn/moe把从环境变量读改为从config读（涉及到调用链上面的很多处修改，需要看在哪里init）
        self.att_dp_size = int(os.environ['ATTN_WORKERS_COUNT'])
        self.ep_size = int(os.environ['MOE_WORKERS_COUNT'])
        self.attn_workers = [i for i in range(self.att_dp_size)]
        self.moe_workers = [i for i in range(self.att_dp_size, self.att_dp_size + self.ep_size)]  
        # TODO m->n (wy): 给AttentionCommunicator初始化的时候添加一个tensor_pool
        self.tensor_pool = None
        
    async def send_attention_result(self, metadata: torch.Tensor, gpu_hidden_state: torch.Tensor, gpu_topk_ids: torch.Tensor, gpu_topk_weights: torch.Tensor):
                            
            # step 1: send metadata
            # logger.CommunicationHandler(f"send_attention_result before send 1: rank {self.rank}, moe_workers: {self.moe_workers}")
            
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
            send_tasks = []
            for i, dst in enumerate(self.moe_workers):
                send_tasks.append(self.ucx_peers[dst].send(dst_metadata_list[i]))

            await asyncio.gather(*send_tasks)
            # logger.CommunicationHandler(f"send_attention_result after send 1: rank {self.rank} sync")

            # torch.cuda.synchronize()
            # timestamp = time.time()
            # logger.TimeStamp(f"att send data: {timestamp}   ")      
            
            # # step 2: send real data
            # logger.CommunicationHandler(f"send_attention_result before send 2: rank {self.rank}")
            # TODO 5.11 (wy): 改成发给多个moe节点   
            send_tasks = []
            # 只遍历gpu_topk_ids中有的moe节点，并且转换成global rank
            for dst in gpu_topk_ids.keys():
                global_moe_rank = dst + self.att_dp_size
                send_tasks.append(self.ucx_peers[global_moe_rank].send(gpu_hidden_state[dst]))
                send_tasks.append(self.ucx_peers[global_moe_rank].send(gpu_topk_ids[dst]))
                send_tasks.append(self.ucx_peers[global_moe_rank].send(gpu_topk_weights[dst]))
                # logger.CommunicationHandler(f"loop send to {global_moe_rank}: send_attention_result: rank {self.rank}, dst: moe {dst}")
            if send_tasks:
                await asyncio.gather(*send_tasks)
            # logger.CommunicationHandler(f"send_attention_result after send 2: rank {self.rank} sync")
    
    async def send_attention_result_without_gate(self, metadata: torch.Tensor, gpu_hidden_states: Dict[int, torch.Tensor]):
        """
        Send attention results to MoE workers without gate information.
        This method sends metadata and hidden states to all MoE workers for full data transfer.
        
        Args:
            metadata: Tensor containing [layer_idx, hidden_dim, topk, num_tokens]
            gpu_hidden_states: Dictionary mapping GPU IDs to hidden state tensors
        """
        # step 1: send metadata to all MoE workers
        # logger.CommunicationHandler(f"send 1: rank {self.rank}, moe_workers: {self.moe_workers}")
        
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
        send_tasks = []
        for i, dst in enumerate(self.moe_workers):
            send_tasks.append(self.ucx_peers[dst].send(dst_metadata_list[i]))

        await asyncio.gather(*send_tasks)

        # step 2: send hidden states to all MoE workers
        # logger.CommunicationHandler(f"send 2")
        
        send_tasks = []
        # Send hidden states to all MoE workers (full data transfer)
        
        for dst in self.moe_workers:
            # print(f"dst: {dst}")
            hidden_state = gpu_hidden_states[dst - self.att_dp_size] 
            send_tasks.append(self.ucx_peers[dst].send(hidden_state))
            # print(f"hidden_state: {hidden_state.shape}, to rank {dst}")
        
        await asyncio.gather(*send_tasks)
        # logger.CommunicationHandler(f"send 3: finished")
    
    def parse_list(self, hidden_states: Dict[int, torch.Tensor]):
        new_dict = {}
        if self.tensor_pool:
            pass
        else:
            new_dict = {key: torch.zeros(tensor.shape, device="cuda", dtype=torch.bfloat16) 
            for key, tensor in hidden_states.items()}
        return new_dict


    async def rece_moe_result(self, hidden_states: Dict[int, torch.Tensor]):  
        moe_result = self.parse_list(hidden_states)
        recv_tasks = []
        for src in hidden_states.keys():
            global_moe_rank = src + self.att_dp_size
            recv_tasks.append(self.ucx_peers[global_moe_rank].recv(moe_result[src]))
            # logger.CommunicationHandler(f"rece_moe_result: rank {self.rank}, src: {src}, global_moe_rank: {global_moe_rank}")
        if recv_tasks:
            await asyncio.gather(*recv_tasks)

        # torch.cuda.synchronize()
        # timestamp = time.time()
        # logger.TimeStamp(f"att rece data: {timestamp}   ")      

        # logger.CommunicationHandler(f"rece_moe_result: rank {self.rank}")

        # moeResult = torch.randn(shape[0], shape[1], dtype=torch.bfloat16).cuda()
        # self.communicator.rece(moeResult, src_rank)

        return moe_result


class MoeCommunicator:    
    def __init__(self, comm_handler: CommunicationHandler, rank: int, tensor_pool=None):  
        # print("INIT!!!", torch.cuda.current_device())      
        self.communicator = comm_handler
        self.rank = rank
        # TODO m->n (wy): after m->n: 这里暂时写死了从环境变量读，写完m2n之后要统一所有的attn/moe把从环境变量读改为从config读（涉及到调用链上面的很多处修改，需要看在哪里init）
        self.att_dp_size = int(os.environ['ATTN_WORKERS_COUNT'])
        self.ep_size = int(os.environ['MOE_WORKERS_COUNT'])
        self.attn_workers = [i for i in range(self.att_dp_size)]
        self.moe_workers = [i for i in range(self.att_dp_size, self.att_dp_size + self.ep_size)]  

        # TODO m->n (wy): m -> n这里接收参数需要修改成一个list[list[int]]，可以不放init里面
        # TODO m->n (wy): 或者用metadata存当前对应的metadata，但是再开一个list存每个attn节点需要的top_k expert以及需要处理的token数量
        self.metadata = {
            rank: torch.zeros(4, dtype=torch.int32, device="cuda")
            for rank in self.attn_workers
        }
        logger.CommunicationHandler(f"MoeCommunicator init: metadata: {self.metadata}, self.rank: {self.rank}")
        self.tensor_pool = tensor_pool
    
    def parse_metadata(self):
        # This method will be refactored within rece_attention_result for M-to-N logic.
        # The original logic is preserved here for reference but will be superseded.
        layer_idx, hidden_dim, topk, num_tokens = self.metadata[0] # Placeholder for old logic
        # logger.CommunicationHandler(f"layer_idx: {layer_idx}, hidden_dim: {hidden_dim}, topk: {topk}, num_tokens: {num_tokens}")

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
    
    async def rece_attention_result(self):        
            # M-to-N implementation: Receive from all attention workers concurrently.
            
            # 1. Concurrently receive metadata from all attention workers.
            recv_tasks = []
            for src in self.attn_workers:
                recv_tasks.append(self.ucx_peers[src].recv(self.metadata[src]))
            await asyncio.gather(*recv_tasks)
            
            # logger.CommunicationHandler(f"rece 1, rank {self.rank}, all metadata received.")

            # Check for end-of-request signal before proceeding.
            # If any worker sends a sentinel, we treat it as the end of the request.
            for rank, meta in self.metadata.items():
                if meta[0].item() == -1:
                    # logger.CommunicationHandler(f"rece END_OF_REQUEST signal from rank {rank}.")
                    # Reset metadata for the next valid request
                    for m in self.metadata.values():
                        m.zero_()
                    return "END_OF_REQUEST", None, None

            # 2. Parse all metadata to figure out total size and slices.
            total_tokens = 0
            attn_slices = {}
            active_attn_workers = []
            
            # Assuming hidden_dim and topk are the same across all workers for a given layer.
            layer_idx, hidden_dim, topk = -1, -1, -1

            for rank, meta in self.metadata.items():
                num_tokens = meta[3].item()
                if num_tokens > 0:
                    if layer_idx == -1: # First active worker sets the common metadata
                        layer_idx = meta[0].item()
                        hidden_dim = meta[1].item()
                        topk = meta[2].item()

                    attn_slices[rank] = slice(total_tokens, total_tokens + num_tokens)
                    total_tokens += num_tokens
                    active_attn_workers.append(rank)

            if total_tokens == 0:
                return "FREE_COMPUTE", None, None

            # 3. Allocate large tensors to hold aggregated data.
            # TODO: Integrate with self.tensor_pool
            agg_hidden_state = torch.empty((total_tokens, hidden_dim), dtype=torch.bfloat16, device="cuda")
            
            # 4. Concurrently receive actual data into slices of the large tensors.
            recv_tasks = []
            for src in active_attn_workers:
                data_slice = attn_slices[src]
                recv_tasks.append(
                    self.ucx_peers[src].recv(agg_hidden_state[data_slice])
                )
            
            await asyncio.gather(*recv_tasks)

            # logger.CommunicationHandler(f"rece 2, layer_idx: {layer_idx}, total_tokens: {total_tokens}")

            # 5. Reset metadata buffers for next iteration.
            for meta in self.metadata.values():
                meta.zero_()

            data = {
                "layer_idx": layer_idx, 
                "hidden_state": agg_hidden_state,
                "attn_slices": attn_slices,
                "active_attn_workers": active_attn_workers
            }
            return "COMPUTE", data, None # last param is for compatibility
        
    async def send_moe_result(self, moe_output, attn_slices, active_attn_workers):
        # M-to-N implementation: Scatter results back to the source attention workers.
        
        send_tasks = []
        for dst in active_attn_workers:
            data_slice = attn_slices[dst]
            send_tasks.append(
                self.ucx_peers[dst].send(moe_output[data_slice])
            )

        if send_tasks:
            await asyncio.gather(*send_tasks)

        # torch.cuda.synchronize()
        # timestamp = time.time()
        # logger.TimeStamp(f"moe send data: {timestamp}   ")      

        # logger.CommunicationHandler(f"send 3: finished sending results to {len(active_attn_workers)} workers.")

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

        # self.rank = dist.get_rank()
        print("RANK!!:", self.rank)
        assert self.rank == dist.get_rank()

        # ===== UCX (server n-side) bootstrap: listener → all_gather address → barrier → wait clients =====
        if True:
            # Fully refactored UCX bootstrap, inspired by xnet4.py
            async def _moe_ucx_bootstrap():
                # os.environ["CUDA_VISIBLE_DEVICES"] = "5"
                # 1. Determine own address and broadcast it via NCCL's all_gather first
                def ip2int_list(ip: str): return [int(x) for x in ip.split('.')]
                
                # base_port = int(os.environ.get("MOE_UCX_BASE_PORT", "14000"))
                base_port = 43325
                my_port = base_port + self.rank
                # Using the host's real IP is crucial for RDMA-based transports like rdmacm
                # my_ip = socket.gethostbyname(socket.gethostname())
                my_ip = "127.0.0.1"
                # my_ip = os.environ.get("NCCL_MASTER_IP")
                my_role = 1 # 1 for MoE

                info_tensor = torch.tensor(
                    [rank, my_role] + ip2int_list(my_ip) + [my_port],
                    dtype=torch.int32,
                    device="cuda",
                )
                gather_buf = [torch.zeros_like(info_tensor) for _ in range(world_size)]
                
                # Broadcast/receive all addresses BEFORE creating any UCX listeners/endpoints
                dist.all_gather(gather_buf, info_tensor)

                # Barrier to ensure all processes have the full address book before proceeding
                dist.barrier()
                print(f"[UCX][MOE][rank={rank}] all_gather and barrier done.")

                # Serialize listener creation among MoE workers to avoid race conditions.
                # if rank > self.moe_workers[0]:
                #     # This is not the first MoE worker, so wait for a signal from the previous one.
                #     wait_tensor = torch.zeros(1, dtype=torch.int8, device="cuda")
                #     dist.recv(wait_tensor, src=rank - 1)

                # 2. Now, create the listener on the port that was just advertised
                self.ucx_peers = {}
                async def handler(ep):
                    print("gpu_local_id: ", gpu_local_id)
                    print(torch.cuda.current_device())
                    handshake = torch.empty((1,), dtype=torch.int32, device=f"cuda:{gpu_local_id}")
                    # print("i am here")
                    print(ep)
                    await ep.recv(handshake)
                    # print("i am here2")
                    client_rank = int(handshake.item())
                    self.ucx_peers[client_rank] = ep
                    print(f"[UCX][MOE][rank={rank}] Accepted client {client_rank}, total: {len(self.ucx_peers)}")

                # self.listener = ucxx.create_listener(handler, port=my_port)
                self.listener = ucxx.create_listener(handler, port=0)
                print(f"[UCX][MOE][rank={rank}] My port is: ", self.listener.port)
                
                # Write address to a shared file for Attention workers to discover
                addr_file = "/gemini/space/wxy/janus/log/port_log.txt"
                with open(addr_file, "a") as f:
                    f.write(f"{rank},{my_ip},{self.listener.port}\n")
                # print(f"[UCX][MOE][rank={rank}] Listening on {my_ip}:{my_port}")
                
                # If there is a next MoE worker, send it a signal to proceed.
                # if rank < self.moe_workers[-1]:
                #     signal_tensor = torch.ones(1, dtype=torch.int8, device="cuda")
                #     dist.send(signal_tensor, dst=rank + 1)

                # 3. Wait for all Attention clients to connect
                print(f"[UCX][MOE][rank={rank}] Waiting for {self.att_dp_size} clients...")
                while len(self.ucx_peers) < self.att_dp_size:
                    await asyncio.sleep(0.1)

                # 4. Final barrier for synchronization
                print(f"[UCX][MOE][rank={rank}] All clients connected. Entering final barrier.")
                dist.barrier()
                print(f"[UCX][MOE][rank={rank}] UCX Bootstrap complete. Peers: {list(self.ucx_peers.keys())}")

            try:
                import asyncio
                import ucxx
                import socket
                asyncio.run(_moe_ucx_bootstrap())
            except Exception as e:
                print(f"[UCX][MOE][rank={rank}] UCX bootstrap FAILED: {e}")
                raise
            
            return # Skip NCCL p2p warmup

        # Original NCCL p2p warmup (non-UCX)
        self.nccl_p2p_p_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")