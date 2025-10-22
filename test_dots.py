#!/usr/bin/env python3
"""
Simple Gemini Vision Test: Count red dots in image
"""
import os
from PIL import Image
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyA66HCDZA9HERtGLs9HAc9FA_VMRg7aPtc"
image_path = "ltx_video_source/images/dot.png"

print("="*70)
print("🔴 Gemini Vision Test: Count Pairs of Dots")
print("="*70)

# Check image exists
if not os.path.exists(image_path):
    print(f"❌ Image not found: {image_path}")
    exit(1)

# Load image
print(f"\n📷 Loading: {image_path}")
image = Image.open(image_path)
print(f"   ✓ Size: {image.size}, Mode: {image.mode}")

# Configure Gemini
print(f"\n🤖 Configuring Gemini...")
genai.configure(api_key=GEMINI_API_KEY)

# List models
print("\n📋 Finding vision-capable model...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        model_name = m.name.replace('models/', '')
        print(f"   Found: {model_name}")

# Try gemini-pro first (text+image capable)
try:
    print(f"\n🔍 Testing with gemini-pro...")
    model = genai.GenerativeModel('gemini-pro')
    
    # Ask the question
    question = "What kind of tool should I use to draw the line to connect the dots?"
    print(f"   Question: '{question}'")
    
    response = model.generate_content([question, image])
    
    print("\n" + "="*70)
    print("✅ Gemini Answer:")
    print("="*70)
    print(response.text)
    print("="*70)
    
    print("\n✓ SUCCESS! Gemini vision works!")

except Exception as e:
    error_msg = str(e)
    
    print(f"\n❌ Error: {error_msg[:200]}")
    
    if "429" in error_msg:
        print("\n⏰ Quota exceeded. Your options:")
        print("  1. Wait until tomorrow (daily quota resets)")
        print("  2. Check usage: https://aistudio.google.com/app/apikey")
        print("  3. For now, use CLIP+DINO (no API needed)")
    elif "404" in error_msg or "not found" in error_msg.lower():
        print("\n💡 Try different model. Available with vision:")
        print("   - gemini-1.5-flash-latest")
        print("   - gemini-1.5-pro-latest")
    else:
        print(f"\n❓ Error type: {type(e).__name__}")

