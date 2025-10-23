#!/bin/bash

# Batch Video Generation from test_prompts_quick.txt
# Reads prompts from test_prompts_quick.txt and generates videos one by one

cd ltx_video_source

# Set GPU
export CUDA_VISIBLE_DEVICES=6
# Configuration
PROMPT_FILE="../final_prompts.txt"
CONFIG="configs/ltxv-2b-0.9.8-distilled.yaml"
OUTPUT_BASE="../outputs/batch_$(date +%Y%m%d_%H%M%S)"
HEIGHT=320
WIDTH=512
NUM_FRAMES=160
SEED=2025
FRAME_RATE=16

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Check if test_prompts_quick.txt exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "❌ Error: $PROMPT_FILE not found!"
    echo "Please create test_prompts_quick.txt with one prompt per line"
    exit 1
fi

echo "========================================"
echo "  Batch Video Generation"
echo "========================================"
echo "Prompts file: $PROMPT_FILE"
echo "Output directory: $OUTPUT_BASE"
echo "Configuration: $CONFIG"
echo "Resolution: ${HEIGHT}x${WIDTH}"
echo "Frames: $NUM_FRAMES"
echo "Frame Rate: $FRAME_RATE fps"
echo "Seed: $SEED"
echo ""

# Count total prompts (excluding comments and empty lines)
TOTAL_PROMPTS=$(grep -v '^#' "$PROMPT_FILE" | grep -v '^[[:space:]]*$' | wc -l)
echo "Total prompts to process: $TOTAL_PROMPTS"
echo ""

# Process each prompt
COUNTER=0
SUCCESS_COUNT=0
FAIL_COUNT=0

while IFS= read -r line || [ -n "$line" ]; do
    # Skip comments and empty lines
    if [[ "$line" =~ ^#.*$ ]] || [[ -z "${line// }" ]]; then
        continue
    fi
    
    COUNTER=$((COUNTER + 1))
    
    echo "========================================="
    echo "[$COUNTER/$TOTAL_PROMPTS] Processing prompt:"
    echo "\"$line\""
    echo "========================================="
    
    # Create subdirectory for this prompt
    PROMPT_DIR="$OUTPUT_BASE/prompt_$(printf "%03d" $COUNTER)"
    
    # Run inference
    python inference.py \
        --prompt "$line" \
        --pipeline_config "$CONFIG" \
        --height $HEIGHT \
        --width $WIDTH \
        --num_frames $NUM_FRAMES \
        --frame_rate $FRAME_RATE \
        --seed $SEED \
        --output_path "$PROMPT_DIR"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✅ Success: Video generated for prompt $COUNTER"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "❌ Failed: Error generating video for prompt $COUNTER"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    
    echo ""
    
done < "$PROMPT_FILE"

# Summary
echo "========================================"
echo "  Batch Generation Complete!"
echo "========================================"
echo "Total prompts: $TOTAL_PROMPTS"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"
echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""

# List generated videos
echo "Generated videos:"
find "$OUTPUT_BASE" -name "*.mp4" -type f | sort

echo ""
echo "Done!"

