# import sys
# import os
# import torch
# import json
# import torch.multiprocessing as mp
# from flask import Flask, request, jsonify
# import threading
# import signal

# from janus.CUHKSZ.uccx_demo.moe_runner import MoeRunner


# def init_moe_runner(ep_size, gpu_local_id=0, gpu_global_id=0):   
#     model_runner = MoeRunner(        
#         device="cuda",
#         gpu_local_id=gpu_local_id, # 局部的gpu id
#         gpu_global_id=gpu_global_id, # 全局的gpu id               
#     )       
    
#     # model_runner.init_distributed_environment(
#     #     dist_init_method, 
#     #     ep_size, 
#     #     gpu_local_id,
#     #     gpu_global_id
#     # )
        
#     return model_runner

# class MoEController:
#     def __init__(self, ep_size: int, dp_size: int, ep_node_num: int, ep_node_id: int):
#         self.ep_size = ep_size
#         self.dp_size = dp_size
#         self.ep_node_num = ep_node_num
#         self.ep_node_id = ep_node_id
#         self.processes = []          
        
#     def start_moe_runners(self):
#         mp.set_start_method('spawn', force=True)
#         ep_per_node = self.ep_size // self.ep_node_num # 8 / 2 = 4
#         for gpu_local_id in range(ep_per_node): # 
#             gpu_global_id = gpu_local_id + self.ep_node_id * ep_per_node # 这里的gpu_id是全局id
#             p = mp.Process(
#                 target=init_moe_runner, 
#                 args=(self.ep_size, gpu_local_id, gpu_global_id)
#             )
#             p.start()
#             self.processes.append(p)
#             print(f"Started MoE runner on GPU {gpu_global_id}")

# def run_controller():   
    
    
#     ep_size = int(os.getenv("MOE_WORKERS_COUNT"))    
#     dp_size = int(os.getenv("ATTN_WORKERS_COUNT"))
#     ep_node_num = int(os.getenv("MOE_WORKERS_NODE"))
#     ep_node_id = int(os.getenv("MOE_WORKERS_NODE_ID"))
#     controller = MoEController(ep_size, dp_size, ep_node_num, ep_node_id)
#     print(f"run_controller ep_size={ep_size}, dp_size={dp_size}, ep_node={ep_node_num}, ep_node_id={ep_node_id}")
#     controller.start_moe_runners()
    
#     return controller
