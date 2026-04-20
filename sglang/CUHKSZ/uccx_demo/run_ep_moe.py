import sys
import os

from dotenv import load_dotenv

current_dir = os.path.dirname(__file__)
env_path = os.path.join(current_dir, "../userConfig/.env")
moe_env_path = os.path.join(current_dir, "../userConfig/moe.env")
# load_dotenv(env_path)  
# load_dotenv(moe_env_path)  

REPO_PATH = os.getenv("REPO_PATH")
sys.path.insert(0, REPO_PATH)

from sglang.CUHKSZ.uccx_demo.moe_controller import run_controller
# os.environ["NCCL_IBEXT_DISABLE"] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"

def main():
    controller = run_controller()

if __name__ == "__main__":
    main() 