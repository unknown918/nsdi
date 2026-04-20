'''
Run: 
Example:
    >>> NODE=0 python rdma_test.py
    >>> NODE=1 python rdma_test.py
# TODO: write a bash script to run on all nodes
'''
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

test_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# test_sizes = [126, 127, 128, 129, 130]

def print_env():
    pass
    # print(f"RANK: {os.environ.get('RANK', 'not set')}")
    # print(f"NCCL_PROTO: {os.environ.get('NCCL_PROTO', 'not set')}")
    # print(f"NCCL_ALGO: {os.environ.get('NCCL_ALGO', 'not set')}")
    # print(f"NCCL_P2P: {os.environ.get('NCCL_P2P', 'not set')}")
    # print(f"NCCL_BUFFSIZE: {os.environ.get('NCCL_BUFFSIZE', 'not set')}")
    # print(f"NCCL_MIN_NCHANNELS: {os.environ.get('NCCL_MIN_NCHANNELS', 'not set')}")
    # print(f"NCCL_P2P_LEVEL: {os.environ.get('NCCL_P2P_LEVEL', 'not set')}")
    # print(f"NCCL_P2P_PXN_LEVEL: {os.environ.get('NCCL_P2P_PXN_LEVEL', 'not set')}")
    # print(f"NCCL_P2P_NET_CHUNKSIZE: {os.environ.get('NCCL_P2P_NET_CHUNKSIZE', 'not set')}")
    
def init_process_group(local_rank, global_rank):
    # os.environ['MASTER_ADDR'] = '172.20.67.12'  # a100
    os.environ['MASTER_PORT'] = '8888'
    os.environ['NCCL_DEBUG'] = 'INFO'
    timeout = datetime.timedelta(seconds=60)

    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=global_rank,
        timeout=timeout,
        world_size=int(os.environ["WORLD_SIZE"]),
    )
    torch.cuda.set_device(local_rank)
    print(f"Rank {global_rank} initialized")
    print_env()


def read_config(node):
    global attn_global_ranks, moe_global_ranks
    with open("config.json", "r") as f:
        config = json.load(f)
    
    cur_node_nprocs = 0
    cur_node_rank0 = 0

    def get_ranks(nodes, start_rank=0, check_current=False):
        result = []
        current_rank = start_rank
        for _, node_info in enumerate(nodes):
            nprocs = node_info.get('nprocs', 0)
            node_rank = node_info.get('node_rank')
            if check_current and node_rank == node:
                nonlocal cur_node_nprocs, cur_node_rank0
                cur_node_nprocs = nprocs
                cur_node_rank0 = current_rank        
            result.extend(range(current_rank, current_rank + nprocs))
            current_rank += nprocs
        return result

    attn_global_ranks = get_ranks(config["attn_nodes"], check_current=True)
    moe_global_ranks = get_ranks(config["moe_nodes"], start_rank=len(attn_global_ranks), check_current=True)
    return cur_node_nprocs, cur_node_rank0

def warmup_attn_worker_communication(rank):
    warmup_tensor = torch.ones((1, 2048), dtype=torch.bfloat16).cuda() * rank
    send_ops = []
    for dst in moe_workers:
        send_op = dist.P2POp(dist.isend, warmup_tensor, dst)
        send_ops.append(send_op)

    reqs = dist.batch_isend_irecv(send_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()

def warmup_moe_worker_communication(rank):
    recv_ops = []
    recv_tensors = []
    for src in attn_workers:
        recv_tensor = torch.zeros((1, 2048), dtype=torch.bfloat16).cuda()
        recv_tensors.append(recv_tensor)
        recv_op = dist.P2POp(dist.irecv, recv_tensor, src)
        recv_ops.append(recv_op)
    
    reqs = dist.batch_isend_irecv(recv_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    # print(f"MOE worker {rank} received from {src}, value: {recv_tensors}")
    for i, src in enumerate(attn_workers):
        print(f"MOE worker {rank} received from {src}, value: {recv_tensors[i][0][0]}")

def measure_attn_communication(rank, iterations=10):
    # sizes = [508, 509, 510, 511, 512, 513, 514, 515, 516]
    # sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #  sizes = [126, 127, 128, 129, 130]
    sizes = test_sizes
    all_times = {}  
    
    for size in sizes:
        print(f"\nTesting with tensor size: ({size}, 2048)")
        test_tensor = torch.ones((size, 2048), dtype=torch.bfloat16).cuda() * rank
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        for iter in range(iterations):
            start_event.record()
            
            send_ops = []
            for dst in moe_workers:
                send_op = dist.P2POp(dist.isend, test_tensor, dst)
                send_ops.append(send_op)

            reqs = dist.batch_isend_irecv(send_ops)
            for req in reqs:
                req.wait()
            
            torch.cuda.synchronize()
            
            end_event.record()
            end_event.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)
            dist.barrier()
        
        all_times[size] = times
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"ATTN worker {rank} - Size ({size}, 2048) stats - avg: {avg_time:.2f}ms, min: {min_time:.2f}ms, max: {max_time:.2f}ms")
        print_env()
    plot_communication_times_multi_size(all_times, f"attn_worker_{rank}", "Attention Worker Communication Time")
    return all_times

def measure_moe_communication(rank, iterations=10):
    # sizes = [508, 509, 510, 511, 512, 513, 514, 515, 516]
    # sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    # sizes = [126, 127, 128, 129, 130]
    sizes = test_sizes

    all_times = {}  
    all_received_values = {}
    
    for size in sizes:
        print(f"\nTesting with tensor size: ({size}, 2048)")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        times = []
        received_values = []
        
        for iter in range(iterations):
            recv_ops = []
            recv_tensors = []
            for src in attn_workers:
                recv_tensor = torch.zeros((size, 2048), dtype=torch.bfloat16).cuda()
                recv_tensors.append(recv_tensor)
                recv_op = dist.P2POp(dist.irecv, recv_tensor, src)
                recv_ops.append(recv_op)
            
            start_event.record()
            
            reqs = dist.batch_isend_irecv(recv_ops)
            for req in reqs:
                req.wait()
            
            torch.cuda.synchronize()
            
            end_event.record()
            end_event.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event)
            times.append(elapsed_time)
            
            if iter == 0:
                for tensor in recv_tensors:
                    received_values.append(tensor[0][0].item())
            
            dist.barrier()
        
        all_times[size] = times
        all_received_values[size] = received_values
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        print(f"MOE worker {rank} - Size ({size}, 2048) stats - avg: {avg_time:.2f}ms, min: {min_time:.2f}ms, max: {max_time:.2f}ms")
        print_env()
        if iter == 0:
            for i, src in enumerate(attn_workers):
                print(f"MOE worker {rank} received from {src}, value: {received_values[i]}")
    
    plot_communication_times_multi_size(all_times, f"moe_worker_{rank}", "MOE Worker Communication Time")
    return all_times, all_received_values

def plot_communication_times_multi_size(all_times, filename_prefix, title):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    save_dir = f"communication_plots_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    data = [times for size, times in sorted(all_times.items())]
    labels = [f"({size}, 2048)" for size, _ in sorted(all_times.items())]
    
    bp = plt.boxplot(data, labels=labels)
    
    for i, box in enumerate(bp['boxes']):
        box_x = box.get_xdata().mean()
        q1 = np.percentile(data[i], 25)
        median = np.percentile(data[i], 50)
        q3 = np.percentile(data[i], 75)
        plt.text(box_x, median, f'{median:.1f}', 
                 horizontalalignment='center', 
                 verticalalignment='bottom',
                 fontsize=8)
        plt.text(box_x, q1, f'{q1:.1f}', 
                 horizontalalignment='center', 
                 verticalalignment='top',
                 fontsize=7)
        plt.text(box_x, q3, f'{q3:.1f}', 
                 horizontalalignment='center', 
                 verticalalignment='bottom',
                 fontsize=7)
    
    plt.title(f"{title} by Tensor Size")
    plt.xlabel('Tensor Size')
    plt.ylabel('Communication Time (ms)')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename_prefix}_boxplot.png")
    plt.close()
    
    plt.figure(figsize=(15, 8))
    sizes = sorted(all_times.keys())
    avg_times = [sum(all_times[size])/len(all_times[size]) for size in sizes]
    
    plt.plot(sizes, avg_times, 'o-', linewidth=2)
    for i, (size, avg) in enumerate(zip(sizes, avg_times)):
        plt.text(size, avg, f'{avg:.1f}', 
                 horizontalalignment='center', 
                 verticalalignment='bottom',
                 fontsize=8)
    
    plt.title(f"{title} - Average Time by Tensor Size (Linear Scale)")
    plt.xlabel('First Dimension Size')
    plt.ylabel('Average Communication Time (ms)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename_prefix}_avg_time_linear.png")
    plt.close()
    
    plt.figure(figsize=(15, 8))
    plt.plot(sizes, avg_times, 'o-', linewidth=2)
    plt.title(f"{title} - Average Time by Tensor Size (Log Scale)")
    plt.xlabel('First Dimension Size')
    plt.ylabel('Average Communication Time (ms)')
    plt.xscale('log')  
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename_prefix}_avg_time_log.png")
    plt.close()
    
    with open(f"{save_dir}/{filename_prefix}_raw_data.txt", "w") as f:
        f.write("Size,Iteration,Time(ms)\n")
        for size in sorted(all_times.keys()):
            times = all_times[size]
            for i, t in enumerate(times):
                f.write(f"{size},{i+1},{t:.4f}\n")
    
    print(f"Plots saved to {save_dir}/")

def attn_worker(rank):
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ['NCCL_DEBUG'] = 'INFO'
    print(f"ATTN worker {rank} of {world_size}")
    
    # warm up
    for _ in range(5):
        warmup_attn_worker_communication(rank)
    
    dist.barrier()
    print(f"ATTN worker {rank} barrier done")
    measure_attn_communication(rank)

def moe_worker(rank):
    world_size = int(os.environ["WORLD_SIZE"])
    os.environ['NCCL_DEBUG'] = 'INFO'
    print(f"MOE worker {rank} of {world_size}")
    
    # warm up
    for _ in range(5):
        warmup_moe_worker_communication(rank)
    
    dist.barrier()
    print(f"MOE worker {rank} barrier done")
    measure_moe_communication(rank)

def run_node(rank, cur_node_nprocs, cur_node_rank0, attn_ranks, moe_ranks):
    global_rank = cur_node_rank0 + rank
    os.environ["RANK"] = str(global_rank)
    init_process_group(rank, global_rank)

    global attn_workers, moe_workers
    attn_workers = attn_ranks
    moe_workers = moe_ranks
    
    if global_rank in attn_ranks:
        attn_worker(global_rank)
    else:
        moe_worker(global_rank)

def main():
    node = int(os.environ["NODE"])
    cur_node_nprocs, cur_node_rank0 = read_config(node)
    os.environ["WORLD_SIZE"] = str(len(attn_global_ranks) + len(moe_global_ranks))
    mp.spawn(run_node, args=(cur_node_nprocs, cur_node_rank0, attn_global_ranks, moe_global_ranks), 
             nprocs=cur_node_nprocs, join=True)

# def monitor_channels():
#     import os
#     channels = int(os.environ.get('NCCL_MIN_NCHANNELS', 1))
#     print(f"Using {channels} NCCL channels")

if __name__ == "__main__":
    main()
    print(attn_global_ranks)
    print(moe_global_ranks)
