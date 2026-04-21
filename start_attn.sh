#!/bin/bash

LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0,1,2

python -m sglang.launch_server \
    --model-path ./Qwen3-235B \
    --port 30010 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-dp-attention \
    --enable-disaggregation \
    --tp-size 3 \
    --ep-size 5 \
    --nnodes 2 \
    --node-rank 0 \
    --attention-node 0 \
    --expert-node -1 \
    --dist-init-addr localhost:60000 \
    --sampling-backend pytorch \
    --mem-fraction-static 0.95
