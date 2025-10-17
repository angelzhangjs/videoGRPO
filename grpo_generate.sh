#!/bin/bash

# GRPO-style Video Generation Script
cd ltx_video_source
export CUDA_VISIBLE_DEVICES=3

PROMPT="A glowing jack-o'-lantern with a carved smiling face sits on a wooden porch, candlelight flickering inside"
OUTPUT_DIR="../outputs/grpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Starting GRPO-style generation..."
echo "Prompt: $PROMPT"
echo "Output directory: $OUTPUT_DIR"

# Stage 1: Exploration (low guidance, multiple seeds)
echo "=== Stage 1: Exploration ==="
for seed in 2025 2026 2027 2028; do
    echo "Generating exploration candidate with seed $seed..."
    python inference.py \
        --prompt "$PROMPT" \
        --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
        --height 512 \
        --width 768 \
        --num_frames 160 \
        --frame_rate 16 \
        --seed $seed \
        --guidance_scale 5.0 \
        --num_inference_steps 30 \
        --output_path "$OUTPUT_DIR/stage1/"
done

# Stage 2: Refinement (medium guidance, best seeds)
echo "=== Stage 2: Refinement ==="
for seed in 2025 2027; do  # Assume these were best from stage 1
    echo "Refining candidate with seed $seed..."
    python inference.py \
        --prompt "$PROMPT" \
        --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
        --height 512 \
        --width 768 \
        --num_frames 160 \
        --frame_rate 16 \
        --seed $seed \
        --guidance_scale 8.0 \
        --num_inference_steps 40 \
        --output_path "$OUTPUT_DIR/stage2/"
done

# Stage 3: Final polish (high guidance, best seed)
echo "=== Stage 3: Final Polish ==="
BEST_SEED=2025  # You would determine this from manual review or reward model
echo "Final generation with best seed $BEST_SEED..."
python inference.py \
    --prompt "$PROMPT" \
    --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
    --height 512 \
    --width 768 \
    --num_frames 160 \
    --frame_rate 16 \
    --seed $BEST_SEED \
    --guidance_scale 12.0 \
    --num_inference_steps 50 \
    --output_path "$OUTPUT_DIR/final/"

echo "GRPO generation complete!"
echo "Check results in: $OUTPUT_DIR"
echo "Compare videos and select the best one manually or implement automatic ranking."


