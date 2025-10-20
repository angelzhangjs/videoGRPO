#!/bin/bash

# Image-to-Video Generation with Visual Recognition
cd ltx_video_source
export CUDA_VISIBLE_DEVICES=3

IMAGE="images/dot.png"
OUTPUT_DIR="../outputs/recognition_$(date +%Y%m%d_%H%M%S)"

echo "========================================"
echo "  Visual Recognition + Video Generation"
echo "========================================"
echo ""

# Step 1: Analyze image to understand red dots
echo "Step 1: Analyzing image..."
cd ..
python visual_recognition_pipeline.py > recognition_output.txt 2>&1
cd ltx_video_source

echo "✓ Image analyzed"
echo ""

# Step 2: Generate with enhanced understanding
echo "Step 2: Generating videos with enhanced prompts..."
echo ""

# Try multiple enhanced prompts based on visual understanding
declare -a PROMPTS=(
    "erase the red dots with a white eraser, starting from the top and moving systematically through each dot"
    "slowly erase each of the red circular dots with a white eraser, showing clear erasing motion"
    "use a white eraser to remove the red dots one by one, making smooth strokes across each dot"
    "white eraser moves across the image, erasing the red dots completely, leaving white marks"
)

mkdir -p "$OUTPUT_DIR"

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    echo "Variant $((i+1)): ${PROMPT:0:50}..."
    
    python inference.py \
        --prompt "$PROMPT" \
        --conditioning_media_paths "$IMAGE" \
        --conditioning_start_frames 0 \
        --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
        --height 320 \
        --width 512 \
        --num_frames 300 \
        --seed $((i+1)) \
        --guidance_scale 7.5 \
        --num_inference_steps 40 \
        --output_path "$OUTPUT_DIR/variant_$((i+1))/"
    
    echo "  ✓ Generated variant $((i+1))"
done

echo ""
echo "========================================"
echo "  Generation Complete!"
echo "========================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Compare the variants to see which prompt"
echo "helps the model better recognize and act"
echo "on the red dots."
echo ""

