#!/bin/bash

echo "=== Killing Python and sglang processes ==="

# Kill python processes
echo "Killing python processes..."
pkill -9 python 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Successfully killed python processes"
else
    echo "No python processes found or already killed"
fi

# Kill python3 processes
echo "Killing python3 processes..."
pkill -9 python3 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Successfully killed python3 processes"
else
    echo "No python3 processes found or already killed"
fi

# Kill janus processes
echo "Killing sglang processes..."
pkill -9 janus 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✓ Successfully killed sglang processes"
else
    echo "No sglang processes found or already killed"
fi

echo ""
echo "=== Cleanup completed ==="

# Verify port 30010 is free
echo "Checking if port 30010 is now free:"
PORT_CHECK=$(lsof -i:30010 2>/dev/null)
if [ -z "$PORT_CHECK" ]; then
    echo "✓ Port 30010 is now free"
else
    echo "⚠ Port 30010 is still in use:"
    lsof -i:30010
fi

# Show remaining GPU processes
echo ""
echo "Remaining GPU processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || echo "No GPU processes running"