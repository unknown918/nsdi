#!/bin/bash

python -m sglang.launch_server \
    --model-path ../Qwen3-235B \
    --port 30010 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --enable-dp-attention \
    --tp-size 8 \
    --ep-size 8 \
    --sampling-backend pytorch \
    --mem-fraction-static 0.95