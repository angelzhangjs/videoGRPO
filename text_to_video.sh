#!/bin/bash

# Navigate to ltx_video_source directory where inference.py is located
cd ltx_video_source
# Set GPU
export CUDA_VISIBLE_DEVICES=5
echo "Generating video..."
python inference.py \
    --prompt "A ball bouncing down a staircase, hitting each step sequentially as it falls" \
    --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
    --height 320 \
    --width 512 \
    --num_frames 160 \
    --frame_rate 16 \
    --seed 2025