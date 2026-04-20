import torch
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.server_args import ServerArgs
from sglang.CUHKSZ.disaggregate.afd_moe.deepseek_v2 import UnifiedMoE
from sglang.srt.configs.logger_config import configure_logger
import torch.distributed as dist
from sglang.srt.distributed import (
    init_distributed_environment,
    set_custom_all_reduce,
)
from sglang.CUHKSZ.disaggregate.runner.runner import build_address_book
import asyncio
import ucxx
import os
from sglang.CUHKSZ.disaggregate.communication.ucx_comm import MoEUCXCommunicationHandler

logger = configure_logger(__name__)


class MoeRunner:
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

        self.init_torch_distributed()
        self.init_ucx_distributed()
        self.load_moe(self.local_rank)
        self.communicator = MoEUCXCommunicationHandler()
        asyncio.run(self.forward_ucx())

    def init_ucx_distributed(self):
        self.addr_book = build_address_book(self.rank, self.local_rank, "moe")

        async def _moe_ucx_bootstrap():
            self.ucx_peers = {}

            async def handler(ep):
                handshake = torch.empty((1,), dtype=torch.int32, device="cuda")
                await ep.recv(handshake)
                client_rank = int(handshake.item())
                self.ucx_peers[client_rank] = ep

            self.listener = ucxx.create_listener(handler, port=self.addr_book[self.rank]["port"])
            while len(self.ucx_peers) < self.server_args.tp_size:
                await asyncio.sleep(0.1)
            dist.barrier()
            global_server_args_dict.update(
                {
                    "ucx_peers": self.ucx_peers,
                    "tp_size": self.server_args.tp_size,
                    "ep_size": self.server_args.ep_size,
                }
            )
            print(
                f"[MoeRunner init_ucx_distributed] tp_size: {global_server_args_dict['tp_size']}, ep_size: {global_server_args_dict['ep_size']}")
            print(f"[MoeRunner init_ucx_distributed] ucx_peers: {global_server_args_dict['ucx_peers']}")

        try:
            asyncio.run(_moe_ucx_bootstrap())
        except Exception as e:
            raise e

    def pingpong(self):
        return

    async def forward_ucx(self):
        while True:
            layer_index, hidden_state = await self.communicator.recv_attention_result()
            # print("[MoeRunner forward_ucx] layer_idx: ", layer_index)
            output = self.model.forward_with_gate(layer_index, hidden_state)
            # output = hidden_state
            await self.communicator.send_moe_result(output)

    def forward(self, layer_idx, hidden_state):
        output = self.model.forward_with_gate(layer_idx, hidden_state)
        return output

    def init_torch_distributed(self):
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)

        print(f"[MoeRunner init_torch_distributed] rank: {self.rank}, local_rank: {self.local_rank}")
        init_distributed_environment(
            backend="nccl",
            world_size=self.server_args.tp_size + self.server_args.ep_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method="tcp://" + self.server_args.dist_init_addr,
        )
        print("rank {} set device {}".format(self.rank, self.local_rank))
        torch.cuda.set_device(self.local_rank)
        print(f"[MoeRunner] After set_device, current_device={torch.cuda.current_device()}")

    def load_moe(self, local_rank):
        rank = dist.get_rank()
        ep_gpu_used = self.server_args.ep_size
        from sglang.CUHKSZ.disaggregate.managers.slot_manager import SlotManager

        slot_manager = SlotManager(self.model_config, ep_gpu_used, 1, 1)
        gpu_slots = slot_manager.get_gpu_slots(rank)

        att_dp_num = self.server_args.dp_size

        self.model = UnifiedMoE(
            self.model_config.hf_config,
            gpu_slots=gpu_slots,
            ep_gpu_used=ep_gpu_used,
            gpu_global_id=rank,
            local_rank=local_rank,
            att_dp_num=att_dp_num
        )
        self.model.load_weights(self.model_config.model_path)
        print("[DEBUG] moe model loaded")
