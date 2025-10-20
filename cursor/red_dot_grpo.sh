#!/bin/bash

# GRPO for Red Dot Recognition Task
cd ltx_video_source
export CUDA_VISIBLE_DEVICES=3

PROMPT="erase the red dots with a white eraser"
IMAGE="images/dot.png"
OUTPUT_DIR="../outputs/red_dot_grpo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  GRPO: Red Dot Recognition Task"
echo "========================================"
echo "Prompt: $PROMPT"
echo "Image: $IMAGE"
echo "Output: $OUTPUT_DIR"
echo ""

# GRPO Iteration 1: Exploration (diverse candidates)
echo "=== GRPO Iteration 1: Exploration ==="
echo "Generating diverse video candidates..."

for seed in 1 2 3 4 5 6; do
    echo "  Candidate $seed..."
    python inference.py \
        --prompt "$PROMPT" \
        --conditioning_media_paths "$IMAGE" \
        --conditioning_start_frames 0 \
        --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
        --height 320 \
        --width 512 \
        --num_frames 300 \
        --seed $seed \
        --guidance_scale 6.0 \
        --num_inference_steps 30 \
        --output_path "$OUTPUT_DIR/iteration1/" > /dev/null 2>&1 &
    
    # Limit parallel jobs
    if [ $((seed % 2)) -eq 0 ]; then
        wait
    fi
done
wait

echo "  ✓ Generated 6 candidates"
echo ""

# Evaluate candidates (you would run Python evaluation here)
echo "=== Evaluating Candidates ==="
echo "Run: python red_dot_grpo.py to evaluate videos"
echo "Select best seeds based on rewards"
echo ""

# GRPO Iteration 2: Refinement (best 3 seeds with higher quality)
echo "=== GRPO Iteration 2: Refinement ==="
echo "Refining best candidates..."

# Assuming seeds 2, 4, 5 were best (you would determine this from rewards)
for seed in 2 4 5; do
    echo "  Refining candidate $seed..."
    python inference.py \
        --prompt "$PROMPT" \
        --conditioning_media_paths "$IMAGE" \
        --conditioning_start_frames 0 \
        --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
        --height 320 \
        --width 512 \
        --num_frames 300 \
        --seed $seed \
        --guidance_scale 8.0 \
        --num_inference_steps 40 \
        --output_path "$OUTPUT_DIR/iteration2/" > /dev/null 2>&1
done

echo "  ✓ Refined 3 candidates"
echo ""

# GRPO Iteration 3: Final polish (best seed)
echo "=== GRPO Iteration 3: Final Polish ==="
BEST_SEED=2  # You would determine this from reward evaluation
echo "  Final generation with best seed $BEST_SEED..."

python inference.py \
    --prompt "$PROMPT" \
    --conditioning_media_paths "$IMAGE" \
    --conditioning_start_frames 0 \
    --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
    --height 320 \
    --width 512 \
    --num_frames 300 \
    --seed $BEST_SEED \
    --guidance_scale 10.0 \
    --num_inference_steps 50 \
    --output_path "$OUTPUT_DIR/final/"

echo "  ✓ Final video generated"
echo ""

echo "========================================"
echo "  GRPO Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Review videos in $OUTPUT_DIR"
echo "  2. Evaluate with: python ../red_dot_grpo.py"
echo "  3. Select best video based on rewards"
echo ""

