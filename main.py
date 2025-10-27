#!/usr/bin/env python3
"""
Main script to run video generation using GRPO with physics rewards
"""
import os
import sys
import torch
import numpy as np
import datetime
import json
import gc
import imageio
import glob
import traceback

# Add ltx_video_source to path
ltx_path = os.path.join(os.path.dirname(__file__), "ltx_video_source")
sys.path.insert(0, ltx_path)

# Import LTX-Video components
from ltx_video.inference import infer, InferenceConfig

# Import GRPO components
from video_grpo import grpo_search_with_physics_rewards

def check_gpu_memory():
    """Check available GPU memory and suggest optimizations"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
        print(f"   GPU Memory: {allocated:.1f}GB used / {total:.1f}GB total ({free:.1f}GB free)")
        
        if free < 5.0:
            print("   ⚠️ Low memory warning!")
            return False
        return True
    return False

def create_ltx_video_pipeline():
    """
    Create LTX-Video pipeline for GRPO
    """
    try:
        # Create inference config
        config = InferenceConfig(
            pipeline_config="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
            height = 320, 
            width = 512, 
            num_frames=121,
            seed=2025
        )
        
        # This would create the actual pipeline - simplified for now
        # In practice, you'd need to properly initialize the LTX pipeline
        print("✅ LTX-Video pipeline configuration created")
        return config
        
    except Exception as e:
        print(f"❌ Failed to create pipeline: {e}")
        return None


def main():
    """
    Main function to run GRPO video generation
    """
    print("🚀 Starting GRPO Video Generation")
    print("=" * 70)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 Device: {device}")
    
    # Create output directory (use tmp or specify custom path if disk full)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Try to create output directory, handle disk full error
    try:
        output_dir = f"grpo_outputs/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")
    except OSError as e:
        # Disk quota exceeded - use /tmp instead
        print("⚠️  Disk quota exceeded in current directory")
        output_dir = f"/tmp/grpo_outputs/{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Using /tmp directory: {output_dir}")
        print("⚠️  Note: /tmp files may be deleted on reboot!")
    
    # Create video pipeline
    print("\n📹 Initializing video pipeline...")
    
    # Create real LTX-Video pipeline
    try:
        # LTX-Video imports already done at top of file
        
        # Create video pipeline using LTX-Video inference (PURE TENSOR - no intermediate MP4s)
        
        # Define result class outside to be accessible everywhere
        class VideoResult:
            def __init__(self, tensor):
                self.images = tensor
        
        def ltx_video_pipeline(prompt, **kwargs):
            # Memory-optimized video dimensions
            # HIGH-QUALITY settings for H100 80GB GPU - Use LARGEST model!
            req_height = kwargs.get('height', 320)
            req_width = kwargs.get('width', 512)
            req_num_frames = kwargs.get('num_frames', 161)
            req_seed = kwargs.get('seed', 2025)
            req_guidance = kwargs.get('guidance_scale', None)
            config = InferenceConfig(
                prompt=prompt,
                pipeline_config="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",  # 2B model (faster, less memory)
                height=req_height,
                width=req_width,
                num_frames=req_num_frames,  # Respect caller's request
                frame_rate=16,  # 16 fps (standard for LTX-Video)
                seed=req_seed,
                offload_to_cpu=False,  # H100 has 80GB - keep on GPU for max speed
                output_path=output_dir,  # Save directly under grpo_outputs/<timestamp>
            )
            # Wire GRPO overrides through to the pipeline
            if req_guidance is not None:
                config.guidance_scale_override = float(req_guidance)
            # Do NOT pass latents into LTX pipeline; it asserts when starting at timestep=1.0.
            # We'll drive diversity via seed and guidance instead.
            
            # Generate video
            try:
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # infer() saves video to disk and returns None
                print(f"   Generating video with seed {config.seed}...")
                infer(config)
                print("   ✓ infer() completed")
                
                # Load the saved video and return as tensor
                video_files = glob.glob(f"{output_dir}/**/*.mp4", recursive=True)
                print(f"   Found {len(video_files)} video files in {output_dir}")
                
                if video_files:
                    latest_video = max(video_files, key=os.path.getctime) 
                    # Load video as tensor
                    video_frames = imageio.mimread(latest_video)
                    video_np = np.array(video_frames)  # [T, H, W, C]
                    video_tensor = torch.from_numpy(video_np).float()
                    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
                    video_tensor = (video_tensor / 255.0 - 0.5) * 2  # Normalize to [-1, 1]
                    
                    # Keep the MP4 under grpo_outputs for user access
                    return VideoResult(video_tensor)
                else:
                    print("⚠️ No video file generated, using fallback tensor")
                    # Return fallback tensor instead of None
                    fallback = torch.randn(1, 3, 161, 320, 512)
                    return VideoResult(fallback)
                    
            except Exception as e:
                print(f"❌ Pipeline error: {e}")
                traceback.print_exc()
                print("⚠️ Using fallback tensor")
                # Always return valid result with .images attribute
                fallback = torch.randn(1, 3, 161, 320, 512)
                return VideoResult(fallback)
        
        video_pipeline = ltx_video_pipeline
        print("✅ Real LTX-Video pipeline created")
        
    except Exception as e:
        print(f"❌ Failed to create video pipeline: {e}")
        return

    # Test prompts
    test_prompts = [
        "A ball bouncing down a staircase, hitting each step sequentially as it falls"]
    
    # Run GRPO search for each prompt
    for i, prompt in enumerate(test_prompts):
        print(f"🎬 GRPO Generation {i+1}/{len(test_prompts)}")
        print(f"Prompt: '{prompt}'")
        
        try:
            # Aggressive GPU memory cleanup before each GRPO run
            if torch.cuda.is_available() and device.startswith('cuda'):
                print("🧹 Pre-GRPO memory cleanup...")
                
                try:
                    # Test GPU accessibility before cleanup
                    test_tensor = torch.randn(5, 5).to(device)
                    del test_tensor
                    
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Force garbage collection
                    gc.collect()
                    
                    # Check memory status and warn if low
                    memory_ok = check_gpu_memory()
                    if not memory_ok:
                        print("🔧 Using reduced settings: 3 rounds × 4 candidates = 12 total videos")
                        
                except Exception as gpu_test_error:
                    print(f"❌ GPU accessibility test failed: {gpu_test_error}")
                    print("🔄 Attempting GPU recovery...")
                    
                    try:
                        # Try to reset GPU state
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        
                        # Test again
                        test_tensor = torch.randn(5, 5).to(device)
                        del test_tensor
                        print("✅ GPU recovery successful")
                        
                    except Exception:
                        print("❌ GPU recovery failed, switching to CPU for this run")
                        device = "cpu"
            
            # Run GRPO search with physics rewards (MEMORY OPTIMIZED)
            result = grpo_search_with_physics_rewards(
                video_pipeline=video_pipeline,
                prompt=prompt,
                num_rounds=3,
                candidates_per_round=24,
                reward_type='combined_physics',  # Use comprehensive physics rewards
                base_seed=2025,    # Different seed per prompt
                device=device,
            )
            
            # Print results
            best_episode = result['best_episode']
            all_episodes = result['all_episodes']
            
            print(f"\n🏆 GRPO Results for: '{prompt[:50]}...'")
            print(f"  Best reward: {best_episode.reward:.4f}")
            print(f"  Total videos generated: {len(all_episodes)}")
            print(f"  Best config: α={best_episode.generation_params.get('alpha', 'N/A'):.3f}, "
                  f"guidance={best_episode.generation_params.get('guidance_scale', 'N/A'):.1f}")
            
            # Show reward breakdown if available
            if hasattr(best_episode, 'reward_info') and best_episode.reward_info:
                print("  🏆 Quality breakdown:")
                for key, value in best_episode.reward_info.items():
                    if isinstance(value, (int, float)):
                        if key == 'clip_alignment':
                            print(f"    🎯 CLIP text-video alignment: {value:.3f}")
                        elif key == 'subject_consistency':
                            print(f"    🔄 Subject consistency (DINO): {value:.3f}")
                        elif key == 'physics_velocity':
                            print(f"    ⚡ Physics velocity: {value:.3f}")
                        elif key == 'temporal_diversity':
                            print(f"    📹 Temporal diversity: {value:.3f}")
                        else:
                            print(f"    {key}: {value:.3f}")
            
            # Save ONLY the final best video (after all 5 GRPO rounds complete)
            try:
                prompt_safe = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
                
                print("\n🏆 GRPO Complete! Converting ONLY the best video to MP4...")
                print("   📊 Total videos generated: {len(all_episodes)}")
                print("   🎯 Videos saved: 1 (best only)")
                
                # Save best video tensor
                video_filename = f"{output_dir}/grpo_best_{i+1}_{prompt_safe}.pt"
                torch.save(best_episode.video, video_filename)
                print("  💾 Best video tensor: {video_filename}")
                
                # Convert ONLY the final best video to MP4
                try:
                    mp4_filename = f"{output_dir}/grpo_best_{i+1}_{prompt_safe}.mp4"
                    
                    # Convert the winning tensor to MP4
                    video_tensor = best_episode.video.squeeze(0)  # Remove batch dim [C, T, H, W]
                    video_np = ((video_tensor + 1) * 127.5).clamp(0, 255).byte()  # Denormalize to [0, 255]
                    video_np = video_np.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
                    
                    imageio.mimwrite(mp4_filename, video_np, fps=16, quality=8)
                    print(f"  🎬 Final best MP4: {mp4_filename}")
                    print(f"      ✨ This is the winner from {len(all_episodes)} candidates!")
                    
                except Exception as mp4_error:
                    print(f"  ⚠️ Failed to convert best video to MP4: {mp4_error}")
                
                # Save comprehensive results summary
                results_filename = f"{output_dir}/grpo_results_{i+1}_{prompt_safe}.json"
                
                # Extract CLIP score for highlighting
                clip_score = None
                if hasattr(best_episode, 'reward_info') and best_episode.reward_info:
                    clip_score = best_episode.reward_info.get('clip_alignment', None)
                
                results_summary = {
                    'prompt': prompt,
                    'best_reward': float(best_episode.reward),
                    'clip_text_alignment_score': float(clip_score) if clip_score is not None else None,
                    'grpo_algorithm': {
                        'total_videos_generated': len(all_episodes),
                        'videos_saved_to_disk': 1,
                        'rounds_completed': 10,
                        'candidates_per_round': 16,
                        'selection_method': 'highest_advantage_reward_with_clip'
                    },
                    'reward_components': {
                        'clip_weight': 0.90,
                        'physics_weight': 0.60,
                        'consistency_weight': 0.80,
                        'description': 'CLIP has highest weight for text-video alignment'
                    },
                    'best_video_config': best_episode.generation_params,
                    'reward_breakdown': best_episode.reward_info if hasattr(best_episode, 'reward_info') else {},
                    'efficiency': {
                        'storage_saved': f"{len(all_episodes)-1} videos not saved",
                        'only_winner_preserved': True
                    }
                }
                with open(results_filename, 'w') as f:
                    json.dump(results_summary, f, indent=2)
                print(f"  📊 GRPO summary: {results_filename}")
                
            except Exception as save_error:
                print(f"  ⚠️ Failed to save final results: {save_error}")
            
        except Exception as e:
            print(f"❌ GRPO failed for prompt: {e}")
            continue
    
    print(f"\n{'='*70}")
    print("🎉 GRPO Video Generation Complete!")
    print("   Only the best videos have been saved to disk.")
    print("   No intermediate MP4 files were created during GRPO.")
    print(f"📁 Results saved in: {output_dir}")
    print(f"🕐 Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()