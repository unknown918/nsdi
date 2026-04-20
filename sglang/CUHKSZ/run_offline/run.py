import sys
sys.path.insert(0, "/home/moe/amdissagCore")

import os
import time
import asyncio
import sglang as sgl
from sglang.test.test_utils import is_in_ci
from sglang.utils import stream_and_merge, async_stream_and_merge

os.environ["NUM_EXPERTS"] = "6"
os.environ["NUM_HIDDEN_LAYERS"] = "4"
os.environ["NUM_EXPERTS_PER_TOK"] = "6"
os.environ["N_GROUP"] = "1" # Only for DeepSeek-V3
os.environ["TOPK_GROUP"] = "1" # Only for DeepSeek-V3

if is_in_ci():
    import patch

def main():
    
    print("=========================== User Init Engine ============================")
    model_path = "/home/moe/.cache/huggingface/hub/models--Qwen--Qwen1.5-MoE-A2.7B/snapshots/1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
    # llm = sgl.Engine(model_path="/ai_sds_yumc/models/Qwen1.5-MoE-A2.7B")
    llm = sgl.Engine(model_path=model_path,
                    disable_cuda_graph=True,
                    tp_size=4,
                    enable_ep_moe=True
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95}

    print("=========================== User Start Inference ============================")

    # outputs = llm.generate(prompts, sampling_params)
    outputs = llm.generate(prompts, sampling_params) 

    print("=========================== User End Inference ============================")
    # for prompt, output in zip(prompts, outputs):
    #     print("===============================")
    #     print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

if __name__ == '__main__':
    main()
