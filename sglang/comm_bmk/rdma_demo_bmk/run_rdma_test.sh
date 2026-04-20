#!/bin/bash

set -e

REMOTE_DIR="/tmp/deployed"
CONFIG_FILE="${REMOTE_DIR}/config.json"

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

master_addr = None
for node in config.get('attn_nodes', []):
    if node['node_rank'] == 0:
        master_addr = node['ip']
        break

if not master_addr:
    print('Error: No attn_node with node_rank=0 found')
    sys.exit(1)

print(f'MASTER_ADDR={master_addr}')

for node in config.get('attn_nodes', []):
    print(f\"attn {node['node_rank']} {node['ip']} {node['user']}\")

for node in config.get('moe_nodes', []):
    print(f\"moe {node['node_rank']} {node['ip']} {node['user']}\")
")

MASTER_ADDR=$(echo "$NODES_INFO" | grep "^MASTER_ADDR=" | cut -d= -f2)
if [ -z "$MASTER_ADDR" ]; then
    echo "Error: Cannot get MASTER_ADDR"
    exit 1
fi

NODES_INFO=$(echo "$NODES_INFO" | grep -v "^MASTER_ADDR=")

TOTAL_NODES=$(echo "$NODES_INFO" | wc -l)
echo "Found $TOTAL_NODES nodes, MASTER_ADDR=$MASTER_ADDR"

ATTN_NODES=$(echo "$NODES_INFO" | grep "^attn" | wc -l)
MOE_NODES=$(echo "$NODES_INFO" | grep "^moe" | wc -l)
echo "Including $ATTN_NODES attn nodes, $MOE_NODES moe nodes"

read -p "Confirm starting RDMA test? (y/n): " confirm
if [[ $confirm != "y" && $confirm != "Y" ]]; then
    echo "Operation cancelled"
    exit 0
fi

TEMP_DIR=$(mktemp -d)
echo "Log directory: $TEMP_DIR"

STATUS_FILE="$TEMP_DIR/status.txt"
echo "Node status:" > $STATUS_FILE

TOTAL_WORLD_SIZE=$TOTAL_NODES

echo "Starting RDMA test on all nodes..."

launch_test() {
    local node_info="$1"
    read -r node_type node_rank node_ip node_user <<< "$node_info"
    
    local ssh_target="${node_user}@${node_ip}"
    local node_log="$TEMP_DIR/${node_type}_${node_rank}_${node_ip}.log"
    local node_id="${node_type} node[${node_rank}]: ${ssh_target}"
    
    echo "[${node_id}] Starting test..." | tee -a $node_log
    
    if ! ssh -o ConnectTimeout=5 $ssh_target "echo Connection test successful" >> $node_log 2>&1; then
        echo "[${node_id}] Cannot connect to node!" | tee -a $node_log
        echo "${node_id}: Failed - Connection failed" >> $STATUS_FILE
        return 1
    fi
    
    local NODE=${node_rank}
    local NODE_TYPE=${node_type}
    
    echo "[${node_id}] Starting RDMA test, NODE=${NODE} NODE_TYPE=${NODE_TYPE} MASTER_ADDR=${MASTER_ADDR}" | tee -a $node_log
    
    ssh $ssh_target "cd ${REMOTE_DIR} && NODE=${NODE} NODE_TYPE=${NODE_TYPE} MASTER_ADDR=${MASTER_ADDR} WORLD_SIZE=${TOTAL_WORLD_SIZE} nohup python rdma_test.py > rdma_test_${node_type}_${NODE}.log 2>&1 &" >> $node_log 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[${node_id}] Test started successfully!" | tee -a $node_log
        echo "${node_id}: Started, running" >> $STATUS_FILE
        return 0
    else
        echo "[${node_id}] Test failed to start!" | tee -a $node_log
        echo "${node_id}: Failed - Start error" >> $STATUS_FILE
        return 1
    fi
}

pids=()
active_nodes=()

while read node_info; do
    launch_test "$node_info" &
    pids+=($!)
    active_nodes+=("$node_info")
done <<< "$NODES_INFO"

for pid in "${pids[@]}"; do
    wait $pid
done

echo "All launch tasks completed!"

echo "All nodes have started testing, waiting for tests to complete..."

all_completed=false
start_time=$(date +%s)
timeout_seconds=1800

while [ "$all_completed" = false ]; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt $timeout_seconds ]; then
        echo "Wait timeout! Waited ${elapsed} seconds, exceeding the ${timeout_seconds} second limit."
        echo "Some nodes may still be running or have failed."
        break
    fi
    
    all_completed=true
    
    for node_info in "${active_nodes[@]}"; do
        read -r node_type node_rank node_ip node_user <<< "$node_info"
        node_id="${node_type} node[${node_rank}]: ${node_user}@${node_ip}"
        
        if ssh ${node_user}@${node_ip} "ps aux | grep rdma_test.py | grep -v grep" &> /dev/null; then
            all_completed=false
        else
            if ! grep -q "${node_id}: Completed" $STATUS_FILE && ! grep -q "${node_id}: Failed" $STATUS_FILE; then
                echo "[${node_id}] Test completed!"
                sed -i "s/${node_id}: Started, running/${node_id}: Completed/g" $STATUS_FILE
            fi
        fi
    done
    
    if [ "$all_completed" = false ]; then
        echo -n "."
        sleep 10
    fi
done

echo -e "\nAll tests completed or timed out!"

echo -e "\nStarting to collect test results from all nodes..."

MASTER_INFO=$(echo "$NODES_INFO" | grep "^attn 0 ")
read -r master_type master_rank master_ip master_user <<< "$MASTER_INFO"
master_ssh="${master_user}@${master_ip}"

for node_info in "${active_nodes[@]}"; do
    read -r node_type node_rank node_ip node_user <<< "$node_info"
    node_id="${node_type} node[${node_rank}]: ${node_user}@${node_ip}"
    node_ssh="${node_user}@${node_ip}"
    
    if [ "$node_ip" = "$master_ip" ] && [ "$node_rank" = "$master_rank" ]; then
        echo "[${node_id}] This is the Master node, no transfer needed"
        continue
    fi
    
    echo -n "[${node_id}] Looking for latest results folder..."
    
    result_dir=$(ssh $node_ssh "cd ${REMOTE_DIR} && ls -td communication_plots_* 2>/dev/null | head -1 || echo ''")
    
    if [ -z "$result_dir" ]; then
        echo "No results folder found!"
        continue
    fi
    
    echo "Found: $result_dir"
    
    echo -n "[${node_id}] Ensuring the same directory exists on Master node..."
    ssh $master_ssh "cd ${REMOTE_DIR} && mkdir -p $result_dir"
    echo "Done"
    
    echo "[${node_id}] Transferring result files to Master node..."
    
    tmp_transfer_dir=$(mktemp -d)
    
    scp -r $node_ssh:${REMOTE_DIR}/$result_dir/* $tmp_transfer_dir/
    
    scp -r $tmp_transfer_dir/* $master_ssh:${REMOTE_DIR}/$result_dir/
    
    rm -rf $tmp_transfer_dir
    
    echo "[${node_id}] Results transfer completed!"
done

echo -e "\nAll node results have been collected to Master node: ${master_ssh}:${REMOTE_DIR}/$result_dir/"

echo -e "\n==== Test Results Summary ===="
cat $STATUS_FILE

success_count=$(grep -c ": Completed" $STATUS_FILE)
failure_count=$(grep -c ": Failed" $STATUS_FILE)
running_count=$(grep -c ": Started, running" $STATUS_FILE)

echo -e "\nSummary: $success_count/$TOTAL_NODES successfully completed, $failure_count/$TOTAL_NODES failed, $running_count/$TOTAL_NODES may still be running"

if [ $success_count -eq $TOTAL_NODES ]; then
    echo -e "\nCongratulations! RDMA tests on all nodes have completed successfully!"
    echo "Test logs are saved on each node at ${REMOTE_DIR}/rdma_test_<node_type>_<node_rank>.log"
    echo "All test result charts have been collected to Master node: ${master_ssh}:${REMOTE_DIR}/$result_dir/"
else
    if [ $failure_count -gt 0 ]; then
        echo -e "\nWarning: $failure_count nodes failed testing."
    fi
    if [ $running_count -gt 0 ]; then
        echo -e "\nWarning: $running_count nodes may still be running or have unknown status."
    fi
    echo "Detailed launch logs are saved at: $TEMP_DIR"
fi

echo -e "\nTo view complete test results, connect to the Master node to see the merged charts:"
echo "ssh ${master_user}@${master_ip} 'ls -l ${REMOTE_DIR}/$result_dir/'"
echo -e "\nTo view original logs on each node, use:"
echo "ssh <user>@<node_ip> 'cat ${REMOTE_DIR}/rdma_test_<node_type>_<node_rank>.log'"