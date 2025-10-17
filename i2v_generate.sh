#!/bin/bash

cd ltx_video_source

# Set GPU
export CUDA_VISIBLE_DEVICES=3

echo "Generating image-to-video..."
python inference.py \
    --prompt "erase the red dots with a white eraser" \
    --conditioning_media_paths "images/dot.png" \
    --conditioning_start_frames 0 \
    --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
    --height 320 \
    --width 512 \
    --num_frames 300 \
    --seed 2
