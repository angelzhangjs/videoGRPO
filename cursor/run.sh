#!/bin/bash

# Navigate to ltx_video_source directory where inference.py is located
cd ltx_video_source

# Set GPU
export CUDA_VISIBLE_DEVICES=3

echo "Generating video..."
python inference.py \
    --prompt "Hands carefully carving a large orange pumpkin with a knife, creating intricate triangular eyes and a jagged smile, pumpkin seeds scattered on a wooden table." \
    --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
    --height 320 \
    --width 512 \
    --num_frames 160 \
    --frame_rate 16 \
    --seed 2025
