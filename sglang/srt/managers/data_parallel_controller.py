# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A controller that dispatches requests to multiple data parallel workers."""

import logging
import multiprocessing as mp
import signal
import threading
from enum import Enum, auto

import psutil
import setproctitle
import zmq

from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.scheduler import run_scheduler_process, run_moe_scheduler_process
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import bind_port, configure_logger, get_zmq_socket
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DataParallelController:
    def __init__(self, server_args, port_args) -> None:
        # Parse args
        self.max_total_num_tokens = None
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

        self.context = zmq.Context(1 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        self.scheduler_procs = []
        self.workers = [None] * server_args.dp_size

        if not server_args.enable_dp_attention:
            dp_port_args = self.launch_dp_schedulers(server_args, port_args)
        else:
            dp_port_args = self.launch_dp_attention_schedulers(server_args, port_args)

        if server_args.node_rank == 0:
            for dp_rank in range(server_args.dp_size):
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    dp_port_args[dp_rank].scheduler_input_ipc_name,
                    True,
                )

        self.max_req_input_len = None

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        dp_port_args = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            dp_port_args.append(tmp_port_args)

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank),
            )
            threads.append(thread)
            base_gpu_id += server_args.tp_size

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return dp_port_args

    def launch_dp_attention_schedulers(self, server_args, port_args):
        self.launch_tensor_parallel_group(server_args, port_args, 0, None)
        dp_port_args = []
        for dp_rank in range(server_args.dp_size):
            dp_port_args.append(PortArgs.init_new(server_args, dp_rank))
        return dp_port_args

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        # Launch tensor parallel scheduler processes
        scheduler_pipe_readers = []
        total_tp = server_args.tp_size
        tp_per_node_capacity = 8
        full_nodes = total_tp // tp_per_node_capacity
        remaining_tp = total_tp % tp_per_node_capacity
        gpu_id_counter = server_args.base_gpu_id + base_gpu_id

        if server_args.node_rank < full_nodes:
            num_workers = tp_per_node_capacity
        elif server_args.node_rank == full_nodes:
            num_workers = remaining_tp
            
        for local_tp_rank in range(num_workers):
            rank_port_args = port_args
            tp_rank = global_tp_start = server_args.node_rank * tp_per_node_capacity + local_tp_rank  # 全局 TP rank
            if server_args.enable_dp_attention:
                # dp attention has different sharding logic
                _, _, dp_rank = compute_dp_attention_world_info(
                    server_args.enable_dp_attention,
                    tp_rank,
                    server_args.tp_size,
                    server_args.dp_size,
                )
                # compute zmq ports for this dp rank
                rank_port_args = PortArgs.init_new(server_args, dp_rank)
                # Data parallelism resues the tensor parallelism group,
                # so all dp ranks should use the same nccl port.
                rank_port_args.nccl_port = port_args.nccl_port

            reader, writer = mp.Pipe(duplex=False)
            gpu_id = local_tp_rank
            proc = mp.Process(
                target=run_scheduler_process,
                args=(server_args, rank_port_args, gpu_id, tp_rank, dp_rank, writer),
            )
            proc.start()
            self.scheduler_procs.append(proc)
            scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def round_robin_scheduler(self, req):
        self.workers[self.round_robin_counter].send_pyobj(req)
        self.round_robin_counter = (self.round_robin_counter + 1) % len(self.workers)

    def shortest_queue_scheduler(self, input_requests):
        raise NotImplementedError()

    def event_loop(self):
        while True:
            while True:
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break

                if isinstance(
                    recv_req,
                    (
                        TokenizedGenerateReqInput,
                        TokenizedEmbeddingReqInput,
                    ),
                ):
                    self.dispatching(recv_req)
                else:
                    # Send other control messages to first worker of tp group
                    for worker in self.workers[:: self.server_args.tp_size]:
                        worker.send_pyobj(recv_req)


class ExpertParallelController:
    def __init__(self, server_args, port_args) -> None:
        self.server_args = server_args
        self.port_args = port_args
        self.scheduler_procs = []
        self.workers = [None] * server_args.ep_size

        ep_port_args = self.launch_expert_parallel_workers(server_args, port_args)

    def launch_expert_parallel_workers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        ep_port_args = []

        # For each expert, create a new worker and start its process
        
        import math
        attention_node_num = math.ceil(server_args.tp_size / 8)
        moe_node_num = server_args.nnodes - attention_node_num
        ep_size_per_node = server_args.ep_size // moe_node_num
        
        ep_rank_range = range(
            server_args.expert_node * ep_size_per_node,
            (server_args.expert_node + 1) * ep_size_per_node,
        )                        
        print(f"‼️ attention_node_num={attention_node_num}, moe_node_num={moe_node_num}, ep_size_per_node={ep_size_per_node}, ep_rank_range={ep_rank_range}")
        # for ep_rank in range(server_args.ep_size):
        for ep_rank in ep_rank_range:
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name
            ep_port_args.append(tmp_port_args)

            # Create a socket for each expert
            sockets.append(bind_port(tmp_port_args.nccl_port))

            # Launch an individual expert process
            thread = threading.Thread(
                target=self.launch_expert_instance,
                args=(server_args, tmp_port_args, base_gpu_id, ep_rank),
            )
            threads.append(thread)
            base_gpu_id = base_gpu_id + 1
        # Free all sockets before starting the threads
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return ep_port_args

    def launch_expert_instance(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        ep_rank: int,
    ):
        # Each expert just launches an instance with its own parameters
        
        rank_port_args = port_args
        # For each expert, you simply start the process
        reader, writer = mp.Pipe(duplex=False)
        # gpu_id = server_args.dp_size % 8 + ep_rank
        # gpu_id = ep_rank % 8 # 
        gpu_id = base_gpu_id
        print(f"Launch EP{ep_rank} starting at GPU #{gpu_id}.")
        proc = mp.Process(
            target=run_moe_scheduler_process,
            args=(server_args, rank_port_args, gpu_id, ep_rank, writer),
        )
        proc.start()
        self.scheduler_procs.append(proc)

    def event_loop(self):
        while True:
            pass

def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        controller = DataParallelController(server_args, port_args)
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)

def run_expert_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    setproctitle.setproctitle("sglang::expert_parallel_controller")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        controller = ExpertParallelController(server_args, port_args)
        pipe_writer.send(
            {
                "status": "ready",
                # "max_total_num_tokens": controller.max_total_num_tokens,
                # "max_req_input_len": controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or ExpertParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"ExpertParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
