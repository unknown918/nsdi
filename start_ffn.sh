#!/bin/bash

LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path Qwen3-235B \
    --port 30020 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-dp-attention \
    --enable-disaggregation \
    --tp-size 1 \
    --ep-size 7 \
    --nnodes 2 \
    --node-rank 1 \
    --attention-node -1 \
    --expert-node 0 \
    --num-expert-per-gpu 21 \
    --dist-init-addr localhost:60000 \
    --sampling-backend pytorch \
    --mem-fraction-static 0.95
