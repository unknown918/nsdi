import sys
import os

from dotenv import load_dotenv
import torch
current_dir = os.path.dirname(__file__)
env_path = os.path.join(current_dir, "../userConfig/.env")
moe_env_path = os.path.join(current_dir, "../userConfig/moe.env")
load_dotenv(env_path)  
load_dotenv(moe_env_path)  

REPO_PATH = os.getenv("REPO_PATH")
print(f"REPO_PATH: {REPO_PATH}")
sys.path.insert(0, REPO_PATH)

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["UCX_NET_DEVICES"] = "all"
os.environ["UCX_TLS"] = "all"

 
from sglang.CUHKSZ.disaggregate.managers.moe_controller import run_controller
# os.environ["NCCL_IBEXT_DISABLE"] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def main():
    # print("visible_devices = ", os.getenv("CUDA_VISIBLE_DEVICES"))
    controller = run_controller()

if __name__ == "__main__":
    main() 