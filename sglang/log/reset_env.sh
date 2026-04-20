#!/bin/bash
# 文件名: reset_env.sh

# 1. 清空 model/xxx.txt 文件内容
> /gemini/space/wxy/janus/log/port_log.txt
echo "已清空 model/xxx.txt 内容"

# 2. 杀掉相关进程
pkill -9 python3
pkill -9 python
pkill -9 janus
echo "已杀掉 python3 / python / sglang 进程"

# 3. 检查 3033 端口并 kill 占用的进程
pid=$(lsof -t -i:3033)
if [ -n "$pid" ]; then
    kill -9 $pid
    echo "已杀掉占用 3033 端口的进程 (PID: $pid)"
else
    echo "3033 端口未被占用"
fi