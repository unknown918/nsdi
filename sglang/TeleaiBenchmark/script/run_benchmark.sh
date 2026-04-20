#!/bin/bash
set -a

# change the following parameters based on your need
# ------------------------------------------------------------
max_batch_size_power=12 # max batch size (2^12 = 4096), depends on your GPU, memory, etc.
parallel_sizes=(2 4 6 8) # parallel size (=GPU Number)

model_path="/ai_sds_yumc/models/Qwen1.5-MoE-A2.7B" # model path
dataset_path="/zhangzhexiang/sglang/CUHKSZ/run_offline/custom_prompts.json" # dataset path

# can be any folder/ file
result_filename="performance_results.jsonl" # benchmark performance results
# ------------------------------------------------------------

input_len=256 # fixed
output_len=32 # fixed


for parallel_size in "${parallel_sizes[@]}"
do
    echo "=== Testing with parallel size: $parallel_size ==="
        
    for j in $(seq 0 $max_batch_size_power)
    do
        batch_size=$((2**$j))
        echo "batch_size: $batch_size"
    
        python3 sglang_latency_test.py \
            --batch-size $batch_size \
            --input-len $input_len \
            --output-len $output_len \
            --model-path $model_path \
            --dataset-path $dataset_path \
            --result-filename $result_filename \
            --tp-size $parallel_size \
            --ep-size $parallel_size \
            --dp-size $parallel_size \
            --enable-dp-attention \
            --enable-ep-moe
                
        sleep 1
    done
done
set +a

