import json
import re, os, shutil
import requests
import argparse
import subprocess
from transformers import AutoTokenizer
from sglang.utils import print_highlight

from dotenv import load_dotenv

current_dir = os.path.dirname(__file__)
env_path = os.path.join(current_dir, "../userConfig/.env")
data_env_path = os.path.join(current_dir, "../userConfig/data.env")
load_dotenv(env_path)  
load_dotenv(data_env_path)  

# os.environ["LOG_FILE_PATH_MOE"] = "log_moe.txt"Add commentMore actions
# os.environ["LOG_FILE_PATH_E2E"] = "log_e2e.txt"

def load_prompts_from_dataset(dataset_path, batch_size=None, max_length=None, tokenizer=None):
    prompts = [] 
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
                
    for value in list(data.values()):
        if isinstance(value, str) and value.startswith("User:"):                
            prompt = re.sub(r'^User:\s*', '', value)
            prompts.append(prompt)
            
    print(f"Successfully loaded {len(prompts)} prompts from dataset")
    
    if batch_size is not None:
        if len(prompts) < batch_size:
            prompts = prompts * (batch_size // len(prompts) + 1)
        prompts = prompts[:batch_size]
        
    # 如果设置了max_length，对每个prompt进行tokenize和截断
    if max_length is not None and tokenizer is not None:
        tokenized_prompts = []
        for p in prompts:
            tokens = tokenizer.encode(p)
            if len(tokens) > max_length:
                # 截断到指定长度
                tokens = tokens[:max_length]
                # 转回文本
                p = tokenizer.decode(tokens)
            tokenized_prompts.append(p)
        prompts = tokenized_prompts
        
    print(f"Final number of prompts: {len(prompts)}")
    if max_length is not None:
        print(f"Max token length per prompt: {max_length}")

    return prompts

def main(): 
    parser = argparse.ArgumentParser()
    
    dataset_path = os.getenv("DATASET_PATH")
    log_path = os.getenv("LOG_PATH")
    # LOG_PATH= "/root/tmp/breakdown"
        
    model_used = os.getenv("MODEL_USED")

    if model_used == "Qwen":
        model_path =  os.getenv("MODEL_PATH_QWEN")
    elif model_used == "DS2":
        model_path =  os.getenv("MODEL_PATH_DS2")
    elif model_used == "DS3":
        model_path =  os.getenv("MODEL_PATH_DS3")
    else:
        raise ValueError(f"Invalid model used: {model_used}")    
    parser.add_argument('--dataset-path', type=str, default=dataset_path)
    parser.add_argument('--batch-size', type=int, default=256)                                
    parser.add_argument('--input-len', type=int, default=16)                      
    parser.add_argument('--model-path', type=str, default=model_path)
    args = parser.parse_args()

    # 在 /home/moe/amdissagCore/janus/CUHKSZ/run_online/tmp 目录下创建日志文件，breakdown_[batch_size].txt
    # log_file_path = f"/root/tmp/breakdown_{args.batch_size}.txt"
    log_file_path = f"{log_path}_{args.batch_size}.txt" 
    # 创建文件但不写入内容
    open(log_file_path, "w").close()
    # with open(log_file_path, "w") as f:
    #     f.write(f"batch_size: {args.batch_size}\n")
    #     f.write(f"input_len: {args.input_len}\n")
    #     f.write(f"model_path: {args.model_path}\n")
    #     f.write(f"dataset_path: {args.dataset_path}\n")
    # 初始化tokenizer
    print(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, 
        trust_remote_code=True,
        verbosity="error"
    )

    # Load prompts from dataset with length control
    prompts = load_prompts_from_dataset(
        args.dataset_path, 
        args.batch_size,
        args.input_len,
        tokenizer
    )
    
    if not prompts:
        print("No prompts loaded. Exiting.")
        return

    # Prepare request data
    url = "http://localhost:30010/generate"

    sampling_params = {
        "temperature": 1.6,             
        "max_new_tokens": 8,  # 最多生成100个token
        "min_new_tokens": 8,   
    }

    data = {
        "text": prompts,
        "sampling_params": sampling_params
    }

    # Send request
    print("=" * 60)
    print(f"🚀 Sending request with {len(prompts)} prompt(s)...")
    print("=" * 60)
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        print("✅ Request successful!")
        print("-" * 40)
        response_data = response.json()
        
        for i, result in enumerate(response_data):
            print(f"📄 Response {i+1}:")
            print(f"   Generated Text: {result.get('text', 'N/A')}")
            
            meta_info = result.get('meta_info', {})
            finish_reason = meta_info.get('finish_reason', {})
            
            print(f"   Finish Reason: {finish_reason.get('type', 'N/A')}")
            print(f"   Prompt Tokens: {meta_info.get('prompt_tokens', 'N/A')}")
            print(f"   Completion Tokens: {meta_info.get('completion_tokens', 'N/A')}")
            print(f"   Cached Tokens: {meta_info.get('cached_tokens', 'N/A')}")
            
            if i < len(response_data) - 1:
                print("-" * 40)     
    else:
        print(f"❌ Request failed with status code: {response.status_code}")
        print("Response:", response.text)
            

    print("=" * 60)
    print("🗑️  Flushing cache...")
        
    flush_response = requests.post("http://localhost:30010/flush_cache")
    if flush_response.status_code == 200:
        print("✅ Cache flushed successfully")

    print("=" * 60)
    # 将日志文件移动到 /home/moe/amdissagCore/janus/CUHKSZ/run_online/res目录下
    # shutil.move(log_file_path, f"/root/tmp/breakdown_{args.batch_size}.txt")
    shutil.move(log_file_path, f"{log_path}_{args.batch_size}.txt")

    # print("🧹 Executing cleanup script...")
    #     # 执行kill_processes.sh脚本
    # script_path = "/janus/CUHKSZ/run_online/kill_processes.sh"
    
    # # 确保脚本有执行权限
    # os.chmod(script_path, 0o755)
    
    # # 执行脚本并捕获输出
    # result = subprocess.run(
    #     ["bash", script_path], 
    #     capture_output=True, 
    #     text=True,
    #     timeout=10  # 10秒超时
    # )
    
    # if result.returncode == 0:
    #     print("✅ Cleanup script executed successfully")
    #     if result.stdout:
    #         print("Script output:")
    #         print(result.stdout)
    # else:
    #     print(f"⚠️  Cleanup script returned code {result.returncode}")
    #     if result.stderr:
    #         print("Script errors:")
    #         print(result.stderr)
            

print("=" * 60)

if __name__ == "__main__":
    main()