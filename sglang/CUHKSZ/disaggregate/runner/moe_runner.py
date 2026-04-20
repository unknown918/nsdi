import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.server_args import ServerArgs
from sglang.CUHKSZ.disaggregate.afd_moe.moe import UnifiedMoE
from sglang.srt.configs.logger_config import configure_logger
from sglang.srt.model_loader import get_model
import torch.distributed as dist
from sglang.srt.distributed import (
    init_distributed_environment,
    set_custom_all_reduce,
)
import os
from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.utils import monkey_patch_vllm_gguf_config
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.CUHKSZ.disaggregate.communication.utils import get_ep_group_info
from sglang.CUHKSZ.disaggregate.communication.nvshmem_utils import init_nvshmem_distributed
from sglang.srt.distributed.parallel_state import init_expert_parallel_group
from sglang.srt.distributed.parallel_state import get_ep_group

logger = configure_logger(__name__)


class MoERunner:
    def __init__(
            self,
            server_args: ServerArgs,
            port_args,
            gpu_id,
            ep_rank
    ):
        self.moeLayers = None
        self.server_args = server_args
        self.model_config = ModelConfig(
            server_args.model_path,
            model_override_args="{}"
        )
        self.rank = ep_rank
        self.local_rank = gpu_id

        self.num_micro_batch = server_args.num_micro_batch

        self.moe_node_num = server_args.moe_node_num
        self.enable_ep_intra_node_reduce = server_args.enable_ep_intra_node_reduce
        torch.cuda.set_device(gpu_id)

        self.ep_group_info = get_ep_group_info(
            tp_size=server_args.tp_size,
            ep_size=server_args.ep_size,
            moe_node_num=server_args.moe_node_num,
            enable_ep_intra_node_reduce=server_args.enable_ep_intra_node_reduce
        )
        global_server_args_dict["ep_group_info"] = self.ep_group_info

        self.init_torch_distributed()
        init_nvshmem_distributed(server_args)
        self.load_moe(self.local_rank)
        from sglang.CUHKSZ.disaggregate.communication.nvshmem_comm import MoENvshmemCommunicationHandler
        self.communicator = MoENvshmemCommunicationHandler(self.model_config)

        self.my_epgroup_id = self.get_my_epgroup_id()
        self.ep_group = get_ep_group(self.my_epgroup_id)

        comm_breakdown_str = os.environ.get("COMM_BREAKDOWN", "false")
        self.COMM_BREAKDOWN = comm_breakdown_str.lower() == "true"

        self.forward_loop()

        if self.communicator.should_shutdown:
            print(f"[MoeRunner] rank={self.rank} exiting due to comm error: {self.communicator._shutdown_reason}")
        else:
            print(f"[MoeRunner] rank={self.rank} exiting normally")

    def get_my_epgroup_id(self):
        ep_groups = self.ep_group_info['ep_groups']
        for i, ep_group in enumerate(ep_groups):
            print(f"[MoeRunner] rank={self.rank} group_ranks={ep_groups}")
            if self.rank in ep_group:
                return i
        print(f"[MoeRunner] rank={self.rank} my_epgroup_id=None")
        return None

    def _warmup_forward(self):
        """Run dummy forward_with_gate on default stream to warmup
        CUBLAS handles and Triton JIT caches."""
        hidden_dim = self.model_config.hf_config.hidden_size
        dummy = torch.randn(1, hidden_dim, dtype=torch.bfloat16, device="cuda")
        first_moe_layer = 1
        with torch.no_grad():
            _ = self.model.forward_with_gate(first_moe_layer, dummy)
        torch.cuda.synchronize()
        print(f"[MoeRunner] rank={self.rank} warmup done", flush=True)

    def pingpong(self):
        return

    def forward_loop(self):
        """Single-thread loop that processes all micro-batches sequentially.
        Multi-threading is not viable because .item() does cudaDeviceSynchronize
        which deadlocks across threads. The micro-batch overlap is driven by
        the attention side's MBO; MoE side just needs to service requests for
        each batch_id in order."""
        # Warmup: Triton JIT + CUBLAS handle init on default stream
        self._warmup_forward()

        while True:
            if self.communicator.should_shutdown:
                break

            for batch_id in range(self.num_micro_batch):
                if self.communicator.should_shutdown:
                    break

                layer_index, hidden_state = self.communicator.recv_attention_result(batch_id)

                if hidden_state is None:
                    self.communicator.request_shutdown("recv failed")
                    break

                # Skip sentinel: tokens=0 means attention did not use MBO
                if hidden_state.shape[0] == 0:
                    continue

                output = self.model.forward_with_gate(layer_index, hidden_state)
                if self.enable_ep_intra_node_reduce:
                    output = self.ep_group.all_reduce(output)

                self.communicator.send_moe_result(layer_index, batch_id, output)

    def init_torch_distributed(self):
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)

        init_distributed_environment(
            backend="nccl",
            world_size=self.server_args.tp_size + self.server_args.ep_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method="tcp://" + self.server_args.dist_init_addr,
        )

        init_expert_parallel_group(self.server_args.ep_size, self.moe_node_num)
        moe_ranks = list(range(self.server_args.tp_size, self.server_args.tp_size + self.server_args.ep_size))

    def load_moe(self, local_rank):
        rank = dist.get_rank()
        ep_gpu_used = self.server_args.ep_size
        att_dp_num = self.server_args.dp_size

        self.model = UnifiedMoE(
            self.model_config.hf_config,
            self.server_args,
            ep_gpu_used=ep_gpu_used,
            gpu_global_id=rank,
            local_rank=local_rank,
            att_dp_num=att_dp_num
        )
        self.model.load_weights(self.model_config.model_path)
        # self.load_config = LoadConfig(
        #     load_format=self.server_args.load_format,
        #     download_dir=self.server_args.download_dir,
        # )
        # if self.server_args.load_format == "gguf":
        #     monkey_patch_vllm_gguf_config()
        # monkey_patch_vllm_parallel_state()
        # memory_saver_adapter = TorchMemorySaverAdapter.create(
        #     enable=self.server_args.enable_memory_saver
        # )
        # with memory_saver_adapter.region():
        #     self.model = get_model(
        #         model_config=self.model_config,
        #         load_config=self.load_config,
        #         device_config=DeviceConfig("cuda"),
        #     )
        # monkey_patch_vllm_parallel_state(reverse=True)
