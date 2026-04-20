from huggingface_hub import snapshot_download

# 下载整个模型仓库到本地
snapshot_download(repo_id="Qwen/Qwen3-30B-A3B-Instruct-2507")
