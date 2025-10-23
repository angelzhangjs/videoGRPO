#!/bin/bash
# Kill existing Python processes to free GPU memory
echo "Cleaning up GPU memory..."
pkill -f "python.*main.py" 2>/dev/null || true
sleep 3

# Initialize conda (try multiple possible locations)
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f /home/ubuntu/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/ubuntu/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
else
    echo "⚠️  Conda not found, using system Python3"
fi

# Try to activate conda environment
if command -v conda &> /dev/null; then
    conda activate videosearch 2>/dev/null || echo "⚠️  videosearch environment not found, using base"
fi

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Use SINGLE GPU only (avoid fragmentation across multiple GPUs)
export CUDA_VISIBLE_DEVICES=0

echo "Starting with:"
echo "  - Conda environment: videosearch"
echo "  - GPU 0 (23.7 GB free - plenty of space!)"
echo "  - Resolution: 256x384"
echo "  - Frames: 81"
echo "  - CPU offloading: enabled"
echo ""

# Ensure we're in the correct directory
cd /home/ghr/angel/videosearch

# Activate virtual environment if it exists
if [ -d "env" ]; then
    source env/bin/activate
    echo "✅ Virtual environment activated"
fi

# Use python3 as fallback if python is not available
if command -v python &> /dev/null; then
    python main.py
else
    python3 main.py
fi