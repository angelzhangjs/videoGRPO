#!/bin/bash

cd ltx_video_source

# Set GPU
export CUDA_VISIBLE_DEVICES=3

# IMPROVED PROMPT: More specific and descriptive
# Instead of: "erase the red dots with a white eraser"
# Use: Detailed description of action, object, and motion

echo "Generating image-to-video with IMPROVED prompt..."
python inference.py \
    --prompt "a white eraser slowly moves across the surface, erasing small red circular dots one by one with smooth back-and-forth strokes, leaving white eraser marks" \
    --conditioning_media_paths "images/dot.png" \
    --conditioning_start_frames 0 \
    --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
    --height 320 \
    --width 512 \
    --num_frames 300 \
    --seed 2 \

echo "Done! Compare this with the basic prompt version."

