#!/usr/bin/env python3
"""
Simple Gemini Vision Test - Image with Red Dots
"""
import os
from PIL import Image

try:
    import google.generativeai as genai
    print("✓ Gemini SDK imported")
except ImportError:
    print("❌ Install: pip install google-generativeai")
    exit(1)

# Your Gemini API Key
GEMINI_API_KEY = "AIzaSyCeZPolffTxxQQDVKbrb3U7M1MM-oLo_YU"

def test_gemini_with_image():
    """Test Gemini vision with single image"""
    
    print("="*70)
    print("🎯 Gemini Vision Test: Count Red Dots")
    print("="*70)
    
    # Configure API
    genai.configure(api_key=GEMINI_API_KEY)
    
    # List available models
    print("\n📋 Available models:")
    vision_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            model_name = m.name.replace('models/', '')
            print(f"   - {model_name}")
            if any(keyword in model_name.lower() for keyword in ['flash', 'pro', 'vision']):
                vision_models.append(model_name)
    
    if not vision_models:
        print("\n❌ No vision models found")
        return
    
    # Use first available vision model
    model_name = vision_models[0]
    print(f"\n✓ Using model: {model_name}")
    
    # Load image
    #dot_image_path = "ltx_video_source/images/dot.png"
    image_path = "ltx_video_source/images/corgi.png"
    
    if not os.path.exists(image_path):
        print(f"\n❌ Image not found: {image_path}")
        print("Available images:")
        images_dir = "ltx_video_source/images"
        if os.path.exists(images_dir):
            for f in os.listdir(images_dir):
                print(f"   - {f}")
        return
    
    print(f"\n📷 Loading image: {image_path}")
    image = Image.open(image_path)
    print(f"   ✓ Image loaded: {image.size} pixels, mode: {image.mode}")
    
    # Create model
    model = genai.GenerativeModel(model_name)
    
    # Query Gemini
    question = "What breed of dog is this?"
    
    print(f"\n🤖 Asking Gemini: '{question}'")
    print("   Sending request...")
    
    try:
        # Send image + question to Gemini
        response = model.generate_content([question, image])
        
        print("\n" + "="*70)
        print("✅ Gemini Response:")
        print("="*70)
        print(response.text)
        print("="*70)
        
        print("\n✓ Gemini vision API works!")
        print("\n💡 Next steps:")
        print("  - Gemini can analyze images ✓")
        print("  - Can use for video frame analysis")
        print("  - Ready for GRPO reward model training")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
        if "quota" in str(e).lower() or "429" in str(e):
            print("\n⏰ Rate limit hit. Wait 1 minute and try again.")
            print("   Free tier: 15 requests/minute")
        elif "404" in str(e):
            print(f"\n❌ Model '{model_name}' not found or doesn't support images")
            print("   Try a different model from the list above")
        else:
            print(f"\n❌ Unexpected error: {type(e).__name__}")


if __name__ == "__main__":
    test_gemini_with_image()

