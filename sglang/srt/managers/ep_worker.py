import threading
from typing import Optional

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.CUHKSZ.disaggregate.runner.moe_runner import ExpertModelRunner
from sglang.srt.server_args import ServerArgs

from sglang.srt.configs.logger_config import configure_logger

logger = configure_logger(__name__)

class EpModelWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        ep_rank: int,
        nccl_port: int,
        is_draft_worker: bool = False,
    ):
        self.ep_rank = ep_rank
        self.model_config = ModelConfig(
            (
                server_args.expert_model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            dtype=server_args.dtype,
            quantization=server_args.quantization,
        )
        self.model_runner = ExpertModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=gpu_id,
            ep_rank=ep_rank,
            nccl_port=nccl_port,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
        )
        self.device = self.model_runner.device

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: Optional[threading.Event] = None,
        skip_sample: bool = False,
    ):
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        if launch_done:
            launch_done.set()

        if skip_sample:
            next_token_ids = None
        else:
            next_token_ids = self.model_runner.sample(logits_output, model_worker_batch)

        return logits_output, next_token_ids
