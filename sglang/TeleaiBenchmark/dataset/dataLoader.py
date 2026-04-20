"""
Extract prompts from ShareGPT dataset that meet the requirements

EG:
python dataLoader.py --dataset-path dataset.json --tokenizer-path Qwen1.5-MoE-A2.7B --context-len 256 --num-prompts 10240 --output-file prompts.json

--dataset-path : the path of the ShareGPT dataset
--tokenizer-path : the path of the tokenizer
--context-len : the context length of all prompt
--num-prompts : the number of prompts to extract
--output-file : the path of the output file
"""

import argparse
import json
import logging
import os
import random
from typing import List, Dict, Optional

from transformers import AutoTokenizer, PreTrainedTokenizer

def get_tokenizer(tokenizer_path: str) -> PreTrainedTokenizer:    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def analyze_length_distribution(prompts: List[str], tokenizer: PreTrainedTokenizer) -> None:    
    length_distribution = {}
    
    for prompt in prompts:
        # only consider the user input part
        user_parts = []
        for part in prompt.split("\n\n"):
            if part.startswith("User:"):
                user_parts.append(part)
        
        if user_parts:
            user_text = "\n\n".join(user_parts)
            token_count = len(tokenizer.encode(user_text))
            
            # calculate the bucket (every 100 tokens as a group)
            bucket = (token_count // 100) * 100
            length_distribution[bucket] = length_distribution.get(bucket, 0) + 1
    
    # output the statistics
    print("User input token length distribution (every 100 tokens as a group):")
    for bucket in sorted(length_distribution.keys()):
        print(f"  {bucket}-{bucket+99} tokens: {length_distribution[bucket]}个prompt")


def process_dataset(
    dataset_path: str,
    num_prompts: int,
    context_len: int,
    tokenizer: PreTrainedTokenizer,
) -> List[str]:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset file does not exist: {dataset_path}")
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded the dataset: {dataset_path}, {len(data)} records")
        
    random.shuffle(data)
    
    all_prompts = []
    
    filtered_prompts = []
    token_count_stats = {"total": 0, "equal": 0, "greater": 0, "less": 0}
    filter_stats = {"assistant_start": 0}
    
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
        prompts = [item for item in data if item.strip()]        
    else:
        # ShareGPT format processing        
        prompts = []
        for item in data:
            if "conversations" in item:
                # build the conversation text
                text = ""
                for msg in item["conversations"]:
                    role = msg.get("role", msg.get("from", "unknown"))
                    content = msg.get("content", msg.get("value", ""))
                    
                    if role in ["user", "human"]:
                        text += f"User: {content}\n\n"
                    elif role in ["assistant", "gpt"]:
                        text += f"Assistant: {content}\n\n"
                    elif role in ["system"]:
                        text += f"System: {content}\n\n"
                
                if text:
                    prompts.append(text)
    
    # collect all prompts for length analysis
    all_prompts.extend(prompts)
    
    # process each prompt
    for prompt in prompts:
        # filter out prompts starting with Assistant:
        if prompt.lstrip().startswith("Assistant:"):
            filter_stats["assistant_start"] += 1
            continue
            
        tokens = tokenizer.encode(prompt)
        token_count = len(tokens)
        token_count_stats["total"] += 1
        
        if token_count < context_len:
            # skip prompts with length less than context_len
            continue
        elif token_count == context_len:
            filtered_prompts.append(prompt)
            token_count_stats["equal"] += 1
        else:  # token_count > context_len
            # truncate the prompt to context_len
            truncated_tokens = tokens[:context_len]
            truncated_prompt = tokenizer.decode(truncated_tokens)
            filtered_prompts.append(truncated_prompt)
            token_count_stats["greater"] += 1
            
        # if enough prompts are collected, stop processing
        if len(filtered_prompts) >= num_prompts:
            break
        
    analyze_length_distribution(all_prompts, tokenizer)
    
    # prompts are not enough
    if len(filtered_prompts) < num_prompts:
        print(f"Warning: Only {len(filtered_prompts)} prompts were found < {num_prompts}")
    else:
        print(f"Successfully extracted {len(filtered_prompts)} prompts")
        
    return filtered_prompts[:num_prompts]


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--tokenizer-path", type=str, default="/ai_sds_yumc/models/Qwen1.5-MoE-A2.7B")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
        
    random.seed(args.seed)
        
    tokenizer = get_tokenizer(args.tokenizer_path)
        
    prompts = process_dataset(
        args.dataset_path,
        args.num_prompts,
        args.context_len,
        tokenizer
    )
        
    # save to file 
    output_dict = {str(i): prompt for i, prompt in enumerate(prompts)}
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_dict, f, ensure_ascii=False, indent=2)
        
if __name__ == "__main__":
    main()