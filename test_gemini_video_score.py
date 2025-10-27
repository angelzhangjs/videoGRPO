#!/usr/bin/env python3
"""
Gemini VLM Video Reasoning Score Test
Evaluates ball bouncing video and outputs reasoning scores
"""
import os
import numpy as np
from PIL import Image
import imageio
try:
    import google.generativeai as genai
    print("✓ Libraries imported")
except ImportError as e:
    print(f"❌ Missing: {e}")
    print("   Install: pip install google-generativeai imageio imageio-ffmpeg")
    exit(1)

# Configuration
GEMINI_API_KEY = "AIzaSyCeZPolffTxxQQDVKbrb3U7M1MM-oLo_YU"
#Ball_bouncing_VIDEO_PATH = "outputs/physics_outputs/prompt_004/video_output_0_a-ball-bouncing-down-a-staircase_2025_320x512x160_0.mp4"
#VIDEO_PATH = "outputs/physics_outputs/prompt_004/video_output_0_corgi-walking-down-the-street_2025_320x512x160_0.mp4"
#VIDEO_PATH = "frame_by_frame_video/beam_Flocks_of_birds_spiral_upwards_in_synchronized_arcs_weaving_around_the_rooftops_before_scattering_into_the_open_sky._0.mp4"
#VIDEO_PATH = "frame_by_frame_video/Flocks_of_birds_spiral_upwards_in_synchronized_arcs_weaving_around_the_rooftops_before_scattering_into_the_open_sky_0.mp4"
VIDEO_PATH = "outputs/video_output_0_a-ball-bouncing-up-a-staircase_2025_320x512x64_0.mp4"
print("="*70)
print("🎬 Gemini VLM Video Reasoning Evaluation")
print("="*70)
print(f"Video: {VIDEO_PATH}")

# Check video exists
if not os.path.exists(VIDEO_PATH):
    print(f"❌ Video not found: {VIDEO_PATH}")
    exit(1)

# Load video and extract frames
print("\n📹 Loading video...")
reader = imageio.get_reader(VIDEO_PATH)
all_frames = []
for frame in reader:
    all_frames.append(frame)
reader.close()

print(f"   ✓ Loaded {len(all_frames)} frames")

# Sample 4 key frames (to save quota)
frame_indices = [0, len(all_frames)//3, 2*len(all_frames)//3, len(all_frames)-1]
sampled_frames = [all_frames[i] for i in frame_indices]

# Convert to PIL Images
pil_frames = [Image.fromarray(frame) for frame in sampled_frames]
print(f"   ✓ Sampled {len(pil_frames)} key frames for analysis")

# Configure Gemini
print("\n🤖 Configuring Gemini...")
genai.configure(api_key=GEMINI_API_KEY)

# Find available models
print("📋 Finding available models...")
available_models = []
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        model_name = m.name.replace('models/', '')
        print(f"   - {model_name}")
        available_models.append(model_name)

if not available_models:
    print("\n❌ No models found!")
    exit(1)

# Try models in order of preference
model_preferences = [
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-flash',
    'gemini-pro',
]

model_to_use = None
for preferred in model_preferences:
    for available in available_models:
        if preferred in available:
            model_to_use = available
            break
    if model_to_use:
        break

# Fallback to first available
if not model_to_use:
    model_to_use = available_models[0]

print(f"\n✓ Using model: {model_to_use}")

model = genai.GenerativeModel(model_to_use)

# Create physics reasoning and motion consistency evaluation prompt
reasoning_prompt = """
Analyze this video of A ball bouncing up a staircase, hitting each step sequentially as it arrives.

Rate the following aspects on a scale of 0-10:

**PHYSICS REASONING:**
- Does the ball accelerate downward due to gravity between bounces?
- Are the bounce heights decreasing over time due to energy loss?
- Does the ball follow realistic parabolic trajectories between step contacts?
- Are the bounce angles physically plausible (angle of incidence ≈ angle of reflection)?
- Does the ball lose energy with each bounce (reduced bounce height)?
- Is the ball's speed increasing as it progresses down the staircase?

**MOTION CONSISTENCY:**
- Are the motion patterns smooth and continuous between frames?
- Does the ball maintain forward momentum while bouncing down?
- Is the overall motion sequence temporally coherent and believable?
- Does the ball interact correctly with each step of the staircase?

Respond in this format:

Physics Reasoning: [score]/10
Motion Consistency: [score]/10
Overall Score: [average]/10

Brief Explanation: [Why these scores? What works well and what doesn't?]
Improvement Suggestions: [How to enhance the physics and motion realism?]
"""

# Send to Gemini
print("\n🔍 Sending to Gemini for reasoning evaluation...")
print("   (This may take 10-20 seconds)")

try:
    # Prepare content: prompt + frames
    content = [reasoning_prompt]
    for i, frame in enumerate(pil_frames):
        content.append(f"Frame {i+1}/{len(pil_frames)}:")
        content.append(frame)
    
    # Query Gemini
    response = model.generate_content(content)
    
    # Display results
    print("\n" + "="*70)
    print("📊 Gemini VLM Reasoning Scores")
    print("="*70)
    print(response.text)
    print("="*70)
    
    print("\n✅ SUCCESS! Gemini evaluated your video!")
    print("\n💡 These scores can be used for:")
    print("  - GRPO candidate selection")
    print("  - Training reward models")
    print("  - Validating physics quality")
    print("  - Understanding video strengths/weaknesses")

except Exception as e:
    error_msg = str(e)
    
    print(f"\n❌ Error: {error_msg[:300]}")
    
    if "429" in error_msg or "quota" in error_msg.lower():
        print("\n⏰ Quota Exceeded")
        print("   Daily free tier limit reached")
        print("   Solutions:")
        print("     1. Wait until tomorrow")
        print("     2. Use CLIP+DINOv2+Physics rewards (handcraft_reward_funcion.py)")
        print("     3. Check quota: https://aistudio.google.com/app/apikey")
    elif "404" in error_msg:
        print(f"\n❌ Model not available")
        print("   Try: gemini-1.5-flash or gemini-1.5-pro")
    else:
        print(f"\n❓ Unexpected error: {type(e).__name__}")
        print("   Full error saved for debugging")

if __name__ == "__main__":
    pass  # Main code runs at module level

