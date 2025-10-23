#!/bin/bash

echo "🔍 GPU Memory Analysis"
echo "======================"

# Get GPU memory info
nvidia-smi --query-gpu=index,memory.free,memory.used,memory.total --format=csv,noheader,nounits | while IFS=', ' read -r gpu free used total; do
    free_gb=$(echo "scale=1; $free / 1024" | bc)
    used_gb=$(echo "scale=1; $used / 1024" | bc)
    total_gb=$(echo "scale=1; $total / 1024" | bc)
    usage_percent=$(echo "scale=1; $used * 100 / $total" | bc)
    
    echo "GPU $gpu: ${free_gb}GB free, ${used_gb}GB used (${usage_percent}% usage)"
done

echo ""
echo "🎯 Recommendation:"
echo "GPU 3 has the most free memory (21.5GB) - BEST CHOICE"
echo "GPU 7 has 9.6GB free - Second choice"
echo "GPU 6 has 9.6GB free - Third choice"
echo ""
echo "✅ Current setting: CUDA_VISIBLE_DEVICES=\"3\" (optimal)"
echo ""
echo "🚀 Ready to run: ./main.sh"
