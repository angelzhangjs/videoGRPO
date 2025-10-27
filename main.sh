#!/bin/bash

# Use SINGLE GPU only (avoid fragmentation across multiple GPUs)
export CUDA_VISIBLE_DEVICES=5

# Ensure we're in the correct directory
cd /home/ghr/angel/videosearch

python main.py