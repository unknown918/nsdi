cd /data/250010042/yqj/nsdi
source .venv/bin/activate

python bench.py \
  --model-path ./Qwen3-235B \
  --model-name Qwen3-235B \
  --data-path benchmark/ShareGPT_V3.json \
  --log-root benchmark/Qwen3 \
  --experiments-json benchmark/config.json