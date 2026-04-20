# README

## Overview
To enable the profiler to work correctly:
1. Replace the existing `scheduler.py` in the `sglang` package with the modified `scheduler.py` provided.
2. Dataset : `dataset/custom_prompts.json`, containing **10,239 prompts** of length 256.

Run the `script` to obtain the profiling results.
1. `script/run_benchmark.sh` : to obtain the latency / throughput results.
2. `script/run_benchmark_profiler.sh` : to obtain the cpu / gpu kernel profiling results.

## Compatibility
The `script` currently supports only the **deepseek-moe** series models. If you need to support other MoE (Mixture of Experts) models, comment out the `--enable-dp-attention` option. If you need to support non-MoE models, comment out both `--enable-ep-moe` and `ep-size`.

## Obtaining Other Dataset Formats
1. Download the ShareGPT dataset from [this link](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json).
2. Run the `dataset/dataLoader` script to generate the required format. Example:
   ```bash
   python dataLoader.py --dataset-path dataset.json --tokenizer-path Qwen1.5-MoE-A2.7B --context-len 256 --num-prompts 10240 --output-file prompts.json
   ```

## Configuration Details
- When `--enable-dp-attention` is enabled:
  - `gpu = dp = ep = tp`
- When `--enable-dp-attention` is disabled:
  - `gpu = dp * tp`
