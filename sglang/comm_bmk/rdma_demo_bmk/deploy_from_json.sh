#!/bin/bash

# Multi-node file deployment script (using Python to parse JSON, SSH keys should be pre-configured)
# Usage: ./deploy_from_json.sh [config_file] [remote_target_dir]

set -e  

CONFIG_FILE=${1:-"./config.json"}
REMOTE_DIR=${2:-"/tmp/deployed"}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' does not exist"
    exit 1
fi

if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python is required to parse JSON"
    exit 1
fi

PYTHON_CMD="python"
if ! command -v python &> /dev/null; then
    PYTHON_CMD="python3"
fi

echo "Reading config file: $CONFIG_FILE"
NODES_INFO=$($PYTHON_CMD -c "
import json
import sys

with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)

for node in config.get('attn_nodes', []):
    print(f\"attn {node['node_rank']} {node['ip']} {node['user']}\")

for node in config.get('moe_nodes', []):
    print(f\"moe {node['node_rank']} {node['ip']} {node['user']}\")
")

if [ -z "$NODES_INFO" ]; then
    echo "Error: Could not parse config file or config file is empty"
    exit 1
fi

TOTAL_NODES=$(echo "$NODES_INFO" | wc -l)
ATTN_NODES=$(echo "$NODES_INFO" | grep "^attn" | wc -l)
MOE_NODES=$(echo "$NODES_INFO" | grep "^moe" | wc -l)

echo "Found $ATTN_NODES attention (attn) nodes"
echo "Found $MOE_NODES mixture of experts (moe) nodes"
echo "Total $TOTAL_NODES nodes"

echo "==== File Distribution Configuration ===="
echo "Source directory: $(pwd)"
echo "Target directory: $REMOTE_DIR"
echo "======================="

read -p "Confirm the above configuration and start SSH key setup and file distribution? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Operation cancelled"
    exit 0
fi

if [ ! -f ~/.ssh/id_rsa.pub ]; then
    echo "SSH public key not found. Need to generate SSH key pair"
    read -p "Generate SSH key pair? (y/n): " gen_key
    if [[ $gen_key == "y" || $gen_key == "Y" ]]; then
        ssh-keygen -t rsa -b 2048
    else
        echo "Operation cancelled"
        exit 0
    fi
fi

echo "==== Configuring SSH Keys ===="
echo "Will request password for each node to set up SSH key authentication"

while read node_info; do
    read -r node_type node_rank node_ip node_user <<< "$node_info"
    ssh_target="${node_user}@${node_ip}"
    
    echo "[${node_type} node[${node_rank}]: ${ssh_target}] Configuring SSH key..."
    
    # Use ssh-copy-id to copy SSH key
    if ssh-copy-id -o StrictHostKeyChecking=no $ssh_target; then
        echo "[${ssh_target}] SSH key configuration successful"
    else
        echo "[${ssh_target}] SSH key configuration failed"
        read -p "Continue anyway? (y/n): " continue_anyway
        if [[ $continue_anyway != "y" && $continue_anyway != "Y" ]]; then
            echo "Operation cancelled"
            exit 1
        fi
    fi
done <<< "$NODES_INFO"

echo "SSH key configuration completed"

TEMP_DIR=$(mktemp -d)
echo "Log directory: $TEMP_DIR"

STATUS_FILE="$TEMP_DIR/status.txt"
echo "Node status:" > $STATUS_FILE

process_node() {
    local node_info="$1"
    read -r node_type node_rank node_ip node_user <<< "$node_info"
    
    local ssh_target="${node_user}@${node_ip}"
    local node_log="$TEMP_DIR/${node_type}_${node_rank}_${node_ip}.log"
    local node_id="${node_type} node[${node_rank}]: ${ssh_target}"
    
    echo "[${node_id}] Starting process..." | tee -a $node_log
    
    if ! ssh -o ConnectTimeout=5 $ssh_target "echo Connection test successful" >> $node_log 2>&1; then
        echo "[${node_id}] Cannot connect to node!" | tee -a $node_log
        echo "${node_id}: Failed - Connection failed" >> $STATUS_FILE
        return 1
    fi
    
    echo "[${node_id}] Checking/creating target directory..." | tee -a $node_log
    if ! ssh $ssh_target "mkdir -p $REMOTE_DIR" >> $node_log 2>&1; then
        echo "[${node_id}] Cannot create directory $REMOTE_DIR" | tee -a $node_log
        echo "${node_id}: Failed - Cannot create directory" >> $STATUS_FILE
        return 1
    fi
    
    echo "[${node_id}] Copying files..." | tee -a $node_log
    if rsync -az --progress ./ $ssh_target:$REMOTE_DIR/ >> $node_log 2>&1; then
        echo "[${node_id}] File copy successful!" | tee -a $node_log
        echo "${node_id}: Success" >> $STATUS_FILE
        return 0
    else
        echo "[${node_id}] File copy failed!" | tee -a $node_log
        echo "${node_id}: Failed - File copy error" >> $STATUS_FILE
        return 1
    fi
}

echo "Starting parallel file distribution..."

pids=()

while read node_info; do
    process_node "$node_info" &
    pids+=($!)
done <<< "$NODES_INFO"

for pid in "${pids[@]}"; do
    wait $pid
done

echo "All tasks completed!"

echo -e "\n==== Distribution Result Summary ===="
cat $STATUS_FILE

success_count=$(grep -c ": Success" $STATUS_FILE)
failure_count=$(grep -c ": Failed" $STATUS_FILE)

echo -e "\nSummary: $success_count/$TOTAL_NODES successful, $failure_count/$TOTAL_NODES failed"

if [ $failure_count -gt 0 ]; then
    echo -e "\nThere are failed tasks! Detailed logs saved in: $TEMP_DIR"
    echo "You can use the following command to view all logs:"
    echo "ls -l $TEMP_DIR/*.log"
else
    echo -e "\nAll nodes deployed successfully! Logs saved in: $TEMP_DIR"
    read -p "Delete log files? (y/n): " clean_logs
    if [[ $clean_logs == "y" || $clean_logs == "Y" ]]; then
        rm -rf $TEMP_DIR
        echo "Logs cleaned"
    fi
fi