# Solution: How to Make the Model Recognize Red Dots

## The Problem

The current video generation model **cannot "see" or recognize** specific objects like red dots in the conditioning image. It only uses the image as a general visual reference.

**Why it fails:**
- Text prompt: "erase the red dots"
- Model receives: text + image pixels
- Model **doesn't know WHERE** the red dots are
- Model **doesn't know WHAT** red dots look like specifically
- Result: Generic video that doesn't actually interact with the dots

## The Solution: 3-Tier Approach

### Tier 1: Enhanced Prompts (Easiest - Start Here!)

Instead of vague prompts, use **spatially-aware, action-specific prompts**:

```bash
# ❌ BAD (vague)
"erase the red dots"

# ✅ GOOD (specific)
"slowly erase each of the small red circular dots with a white eraser, showing clear erasing motion across each dot"

# ✅ BETTER (very specific)
"white eraser moves systematically from left to right, erasing each small red dot completely with smooth strokes, leaving white marks where the dots were"
```

**Try this now:**
```bash
cd ltx_video_source
python inference.py \
    --prompt "white eraser moves across the image, erasing the small red circular dots one by one with smooth strokes" \
    --conditioning_media_paths "images/dot.png" \
    --conditioning_start_frames 0 \
    --pipeline_config "configs/ltxv-2b-0.9.8-distilled.yaml" \
    --height 320 \
    --width 512 \
    --num_frames 300 \
    --seed 5 \
    --guidance_scale 8.0 \
    --num_inference_steps 40 \
    --output_path "../outputs/enhanced_prompt/"
```

### Tier 2: Visual Recognition Pipeline (Better)

Use computer vision to **analyze the image first**, then generate enhanced prompts:

**What it does:**
1. Detects red dots in the image (location, size, count)
2. Describes their spatial distribution
3. Generates spatially-aware prompts
4. Optionally uses Gemini VLM for semantic understanding

**Files created:**
- `visual_recognition_pipeline.py` - Analyzes images and detects red dots
- `generate_with_recognition.sh` - Generates videos with enhanced prompts

**To install dependencies:**
```bash
pip install opencv-python google-generativeai pillow
```

**To use:**
```bash
# Analyze the image
python3 visual_recognition_pipeline.py

# Generate videos with enhanced prompts
bash generate_with_recognition.sh
```

**Example output:**
```
🔴 Red Dots Found: 6
📊 Distribution: 6 red dots scattered across the image (2 on the left, 3 in the center, 1 on the right)

✨ ENHANCED PROMPT:
"erase the red dots with a white eraser. There are 6 red dots scattered across the image. 
Focus on each dot systematically. The eraser should clearly show white strokes on the 
light background. Show smooth, natural erasing motion for each dot."
```

### Tier 3: GRPO with Recognition Rewards (Best)

Use GRPO to learn which generation strategies work best for the red dot task:

**How it works:**
1. Generate multiple video candidates (different seeds/prompts)
2. Evaluate each video with **red dot recognition reward**:
   - ✓ Are dots tracked correctly?
   - ✓ Do they disappear progressively?
   - ✓ Is the erasing motion plausible?
   - ✓ Is it temporally consistent?
3. Select best candidates
4. Refine with higher quality settings
5. Iterate

**Files created:**
- `red_dot_grpo.py` - Reward function for red dot recognition
- `red_dot_grpo.sh` - GRPO training script

**Reward function features:**
```python
- tracking_score: How well dots are tracked across frames (25%)
- completion_score: Task-specific (erasing/connecting) completion (35%)
- disappearance_rate: For erasing - do dots disappear? (20%)
- consistency_score: Smooth transitions, no flickering (10%)
- plausibility_score: Does the action look natural? (10%)
```

**To use:**
```bash
# Run GRPO iterations
bash red_dot_grpo.sh

# This will:
# - Iteration 1: Generate 6 diverse candidates (seeds 1-6)
# - Iteration 2: Refine best 3 candidates
# - Iteration 3: Polish the best one

# Then evaluate:
python3 red_dot_grpo.py
```

## Quick Start: What to Try Right Now

### Option 1: Better Prompts (No code needed)

Edit `i2v_generate.sh` and try these prompts:

```bash
# Prompt 1: Slow and systematic
"slowly erase each small red circular dot with a white eraser, showing clear strokes"

# Prompt 2: Action-focused  
"white eraser moving across the image, leaving white marks as it removes each red dot"

# Prompt 3: Very detailed
"a white eraser appears and methodically erases the red dots one by one, with smooth back-and-forth motions, leaving white eraser marks on the surface"
```

### Option 2: Multiple Variants (GRPO-style exploration)

Run this to test 4 different prompt variations:

```bash
bash generate_with_recognition.sh
```

Compare the results to see which prompt formulation works best!

## Why This Helps

**Before (fails):**
- Model: "I should make a video with erasing motion... somewhere... I think?"
- Result: Generic motion, doesn't target actual dots

**After (works better):**
- Model: "I need to show white eraser strokes on small red circular objects, making smooth motions that remove them"
- Result: More targeted action on dot-like features

## Key Insights

1. **The model doesn't "see" like we do** - it needs textual guidance
2. **Spatial description matters** - "small red circular dots" > "red dots"
3. **Action description matters** - "smooth erasing strokes" > "erase"
4. **GRPO finds what works** - test multiple variations, keep the best

## Next Steps

1. ✅ **Start simple**: Try better prompts in `i2v_generate.sh`
2. ✅ **Install CV tools**: `pip install opencv-python` for visual recognition
3. ✅ **Run analysis**: `python3 visual_recognition_pipeline.py`
4. ✅ **Try GRPO**: `bash red_dot_grpo.sh` for systematic exploration

## Files Summary

| File | Purpose |
|------|---------|
| `visual_recognition_pipeline.py` | Detects red dots and generates enhanced prompts |
| `red_dot_grpo.py` | Reward function for evaluating red dot task performance |
| `red_dot_grpo.sh` | GRPO training script with 3 iterations |
| `generate_with_recognition.sh` | Generate videos with 4 enhanced prompt variants |
| `i2v_generate.sh` | Simple generation (edit prompts here) |

---

**Bottom line**: The model can't magically recognize red dots, but with better prompts and GRPO exploration, you can find generation strategies that work!

