#!/bin/bash

echo "🔍 GPU Memory Analysis"
echo "======================"

# Get GPU memory info
nvidia-smi --query-gpu=index,memory.free,memory.used,memory.total --format=csv,noheader,nounits | while IFS=', ' read -r gpu free used total; do
    free_gb=$((free / 1024))
    used_gb=$((used / 1024))
    total_gb=$((total / 1024))
    usage_percent=$((used * 100 / total))
    
    echo "GPU $gpu: ${free_gb}GB free, ${used_gb}GB used (${usage_percent}% usage)"
done

echo ""
echo "🎯 Finding GPU with most free memory..."

# Find GPU with maximum free memory
best_gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1)
best_free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f2)
best_free_gb=$((best_free / 1024))

echo "✅ BEST GPU: $best_gpu (${best_free_gb}GB free)"
echo ""
echo "📝 Recommended command:"
echo "   export CUDA_VISIBLE_DEVICES=$best_gpu"
echo "   python main.py"
echo ""
echo "Or update main.sh to use GPU $best_gpu"

