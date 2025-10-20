#!/usr/bin/env python3
"""
Simple script to run inference search
"""

import sys
sys.path.append('ltx_video_source')

from inference_search import LTXVideoSearchGenerator, SearchConfig
import torch

def main():
    # Set GPU
    device = "cuda:3"
    
    # Initialize generator
    print("Initializing LTX-Video with search capabilities...")
    generator = LTXVideoSearchGenerator(
        config_path="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
        device=device
    )
    
    # Configure search
    search_config = SearchConfig(
        search_type="beam_search",  # or "guided_search"
        beam_width=3,
        search_steps=5,
        quality_threshold=0.7
    )
    
    # Configure generation
    generation_config = {
        'height': 512,
        'width': 768,
        'num_frames': 160,
        'num_inference_steps': 40,
        'guidance_scale': 7.5,
        'seed': 2025
    }
    
    # Halloween pumpkin prompt
    prompt = "A glowing jack-o'-lantern with a carved smiling face sits on a wooden porch, candlelight flickering inside, autumn leaves gently falling around it"
    
    print(f"Prompt: {prompt}")
    print(f"Search type: {search_config.search_type}")
    
    # Generate with search
    try:
        video = generator.generate_with_search(
            prompt=prompt,
            search_config=search_config,
            generation_config=generation_config
        )
        
        print(f"✅ Success! Generated video shape: {video.shape}")
        
        # Save video (you can implement saving logic here)
        # save_video(video, "output_with_search.mp4")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


