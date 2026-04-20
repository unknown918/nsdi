import asyncio
import time
import torch
import ucxx
from runner import Node

class AttenRunner(Node):
    def __init__(self, rank, world_size, addr_book, num_iter=1000):
        super().__init__(rank, world_size)
        self.addr_book = addr_book
        self.num_iter = num_iter

    async def run(self):
        print(f"[Worker {self.rank}] device={self.device}, connecting to servers...")
        # 建立与所有 Server 的连接
        for srv_rank, info in self.addr_book.items():
            if info["role"] != "server":
                continue
            addr, port = info["ip"], info["port"]
            ep = None
            while ep is None:
                try:
                    ep = await ucxx.create_endpoint(addr, port)
                    self.ucx_peers[srv_rank] = ep
                    print(f"[Worker {self.rank}] connected to Server {srv_rank} at {addr}:{port}")
                except Exception:
                    await asyncio.sleep(0.1)
            handshake = torch.tensor([self.rank], dtype=torch.int32, device=self.device)
            await ep.send(handshake)

        for data_size_mb in [1, 2, 4]:
            elements = (data_size_mb * MB) // 4
            send_tensor = torch.rand((elements,), dtype=torch.float32, device=self.device)
            data_size_bytes = elements * 4

            # 为每个 server 分配独立 recv buffer + stream
            recv_tensors = {srv_rank: torch.empty_like(send_tensor, device=self.device)
                            for srv_rank in self.ucx_peers}
            streams = {srv_rank: torch.cuda.Stream(device=self.device) for srv_rank in self.ucx_peers}

            print(f"[Worker {self.rank}] Start {data_size_mb}MB ping-pong to all servers")

            total_send_ns = 0
            total_recv_ns = 0
            total_pingpong_ns = 0

            for i in range(self.num_iter):
                ping_start = time.perf_counter_ns()

                # send 阶段
                send_start = time.perf_counter_ns()
                send_tasks = [asyncio.create_task(self.send_tensor(send_tensor, dst))
                            for dst in self.ucx_peers]
                await asyncio.gather(*send_tasks)
                send_end = time.perf_counter_ns()

                # recv 阶段
                recv_start = time.perf_counter_ns()
                recv_tasks = [asyncio.create_task(self.recv_tensor(src, recv_tensors[src]))
                            for src in self.ucx_peers]
                await asyncio.gather(*recv_tasks)
                recv_end = time.perf_counter_ns()

                ping_end = time.perf_counter_ns()

                total_send_ns += send_end - send_start
                total_recv_ns += recv_end - recv_start
                total_pingpong_ns += ping_end - ping_start

                # 计算带宽
                num_servers = len(self.ucx_peers)
                send_bw = (data_size_bytes * num_servers) / ((send_end - send_start)/1e9) / (1024**3)  # GB/s
                recv_bw = (data_size_bytes * num_servers) / ((recv_end - recv_start)/1e9) / (1024**3)  # GB/s

                if i % 100 == 0 and self.rank == 0:
                    print(f"[Worker {self.rank}] {data_size_mb}MB iter {i} to {num_servers} servers:")
                    print(f"  Send: {(send_end-send_start)/1e3:.1f} us, "
                        f"Recv: {(recv_end-recv_start)/1e3:.1f} us, \n"
                        f"Send BW: {8*send_bw:.2f} Gb/s, "
                        f"Recv BW: {8*recv_bw:.2f} Gb/s, "
                        f"Total: {(ping_end-ping_start)/1e3:.1f} us")