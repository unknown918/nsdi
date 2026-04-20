import sys

# os.environ[" NCCL_IBEXT_DISABLE"] = '1'

from dotenv import load_dotenv
load_dotenv("/root/sglang/CUHKSZ/userConfig/.env")  # 路径按实际情况
load_dotenv("/root/sglang/CUHKSZ/userConfig/att.env")  

REPO_PATH = os.getenv("REPO_PATH")
sys.path.insert(0, REPO_PATH)

import os
import sglang as sgl
from sglang.CUHKSZ.disaggregate.dist_patch import apply_world_size_patch
# apply the patch
apply_world_size_patch()

model_used = os.getenv("MODEL_USED")

if model_used == "Qwen":
    model_path =  os.getenv("MODEL_PATH_QWEN")
elif model_used == "DS2":
    model_path =  os.getenv("MODEL_PATH_DS2")
elif model_used == "DS3":
    model_path =  os.getenv("MODEL_PATH_DS3")
else:
    raise ValueError(f"Invalid model used: {model_used}")

def main():
    
    print("=========================== User Init Engine ============================")
    
    llm = sgl.Engine(
                    model_path=model_path, 
                    disable_cuda_graph=True,
                    disable_overlap_schedule=True,
                    tp_size=1,
                    dp_size=1,
                    enable_p2p_check=True,
                    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",        
    ]

    sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_new_tokens": 6, "min_new_tokens": 6}

    print("=========================== User Start Inference ============================")

    outputs = llm.generate(prompts, sampling_params) 

    print("=========================== User End Inference ============================")
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")

if __name__ == '__main__':
    main()
