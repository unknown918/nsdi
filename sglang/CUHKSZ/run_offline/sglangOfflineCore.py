"""
A simplified inference script that directly uses SGlang core components without Engine.
"""

import sys
sys.path.append("/zhangzhexiang/")

import os
import time
import torch
import logging
from typing import List, Optional

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import configure_logger, suppress_other_loggers

class CoreInference:
    def __init__(
        self,
        model_path: str,
        tp_rank: int = 0,
        tp_size: int = 1,
        max_total_tokens: Optional[int] = None,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        load_format: str = "auto",
        mem_fraction_static: float = 0.95,
    ):
        """Initialize core components for inference.
        
        Args:
            model_path: Path or name of the model
            tp_rank: Tensor parallel rank
            tp_size: Number of tensor parallel devices
            max_total_tokens: Maximum number of total tokens
            trust_remote_code: Whether to trust remote code
            dtype: Data type for model weights
            load_format: Model loading format
            mem_fraction_static: Memory fraction for static allocation
        """
        # Create minimal server args with all necessary parameters
        server_args = ServerArgs(
            model_path=model_path,
            tp_size=tp_size,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            load_format=load_format,
            device="cuda",
            attention_backend="flashinfer",
            sampling_backend="pytorch",
            max_total_tokens=max_total_tokens,
            max_running_requests=256,
            disable_custom_all_reduce=False,
            triton_attention_reduce_in_fp32=False,
            disable_mla=False,
            torchao_config=None,
            enable_nan_detection=False,
            enable_dp_attention=False,
            enable_ep_moe=False,
            cpu_offload_gb=0,
            enable_memory_saver=False,
            kv_cache_dtype="auto",
            disable_cuda_graph=False,
        )
        port_args = PortArgs.init_new(server_args)

        # Configure logging
        configure_logger(server_args, prefix=f" TP{tp_rank}")
        suppress_other_loggers()

        # Initialize model config with all necessary parameters
        self.model_config = ModelConfig(
            model_path,
            trust_remote_code=trust_remote_code,
            revision=None,
            context_length=None,
            model_override_args="{}",
            is_embedding=False,
            dtype=dtype,
            quantization=None,
        )

        # Initialize model runner with all required parameters
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=mem_fraction_static,
            gpu_id=tp_rank,
            tp_rank=tp_rank,
            tp_size=tp_size,
            nccl_port=port_args.nccl_port,
            server_args=server_args,
            is_draft_worker=False,
        )

        # Initialize tokenizer
        self.tokenizer = get_tokenizer(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        print(f"Initialized with max_total_tokens={self.model_runner.max_total_num_tokens}")

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        """Generate completions for the given prompts.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            List of generated texts
        """
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )

        # Prepare requests
        reqs = []
        for i, prompt in enumerate(prompts):
            # Encode prompt
            input_ids = self.tokenizer.encode(prompt)
            
            # Create request
            req = Req(
                rid=i,
                origin_input_text=prompt,
                origin_input_ids=input_ids,
                sampling_params=sampling_params,
            )
            req.prefix_indices = []
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids)
            reqs.append(req)

        # Create batch
        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            tree_cache=None,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            enable_custom_logit_processor=False,
        )

        # Prefill phase
        batch.prepare_for_extend()
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        logits_output = self.model_runner.forward(forward_batch)
        next_token_ids = self.model_runner.sample(logits_output, forward_batch)

        # Decode phase
        output_ids = [reqs[i].origin_input_ids + [next_token_ids[i]] for i in range(len(reqs))]
        for _ in range(max_new_tokens - 1):
            batch.output_ids = next_token_ids
            batch.prepare_for_decode()
            model_worker_batch = batch.get_model_worker_batch()
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
            logits_output = self.model_runner.forward(forward_batch)
            next_token_ids = self.model_runner.sample(logits_output, forward_batch)
            
            # Append new tokens
            next_token_ids_list = next_token_ids.tolist()
            for i in range(len(reqs)):
                output_ids[i].append(next_token_ids_list[i])

        # Decode output tokens
        outputs = []
        for ids in output_ids:
            text = self.tokenizer.decode(ids)
            outputs.append(text)

        return outputs

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, 'model_runner'):
            # Clean up distributed resources
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

def main():
    try:
        # Initialize inference
        model_path = "/ai_sds_yumc/models/Qwen1.5-MoE-A2.7B"
        batch_size = 2
        inferencer = CoreInference(model_path=model_path)

        # Prepare prompts
        prompts = ["Hello, my name is " for _ in range(batch_size)]
        
        # Generate
        print("Starting generation...")
        start_time = time.time()
        outputs = inferencer.generate(
            prompts=prompts,
            max_new_tokens=32,
            temperature=0.8,
            top_p=0.95
        )
        end_time = time.time()

        # Print results
        print(f"\nGeneration completed in {end_time - start_time:.2f}s")
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(f"\n=== Sample {i} ===")
            print(f"Prompt: {prompt}")
            print(f"Output: {output}")
    finally:
        # Clean up distributed resources
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main() 