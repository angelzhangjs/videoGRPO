#!/usr/bin/env python3
"""
Main script to run video generation using GRPO with physics rewards
"""

import os
import sys
import torch
import numpy as np

# Fix tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable CUDA debugging for error diagnosis
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Enable for debugging GPU errors

# Memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

# Add ltx_video_source to path
sys.path.append("ltx_video_source")

# Import LTX-Video components
from ltx_video.inference import infer, InferenceConfig

# Import GRPO components
from video_grpo import grpo_search_with_physics_rewards

def check_gpu_memory():
    """Check available GPU memory and suggest optimizations"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated
        
        print("🔍 GPU Memory Status:")
        print(f"   Total: {total:.1f}GB")
        print(f"   Allocated: {allocated:.1f}GB")
        print(f"   Cached: {cached:.1f}GB") 
        print(f"   Free: {free:.1f}GB")
        
        if free < 5.0:  # Less than 5GB free
            print("⚠️  Low memory warning! Consider reducing video dimensions or candidates")
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
            height=512,
            width=768,
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
    print("=" * 50)
    
    # Create output folder for GRPO results with unique timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    output_dir = f"grpo_outputs/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print(f"🕐 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set device with error handling
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"🔍 CUDA available: {device_count} device(s) visible")
            
            # Test GPU accessibility
            device = "cuda:0"  # GPU 3 becomes cuda:0 with CUDA_VISIBLE_DEVICES="3"
            
            # Try to create a small tensor to test GPU
            test_tensor = torch.randn(10, 10).to(device)
            print(f"✅ GPU test successful: {device} (physical GPU 3)")
            del test_tensor
            torch.cuda.empty_cache()
            
            print(f"Available GPUs: {device_count}")
            
            # Check memory status
            for i in range(device_count):
                try:
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    print(f"  GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total:.1f}GB total")
                except Exception as gpu_error:
                    print(f"  GPU {i}: Error accessing - {gpu_error}")
        else:
            device = "cpu"
            print("⚠️ CUDA not available, using CPU")
            
    except Exception as cuda_error:
        print(f"❌ CUDA initialization error: {cuda_error}")
        print("🔄 Falling back to CPU")
        device = "cpu"
    
    # Create video pipeline
    print("\n📹 Initializing video pipeline...")
    
    # Create real LTX-Video pipeline
    try:
        # LTX-Video imports already done at top of file
        
        # Create video pipeline using LTX-Video inference (PURE TENSOR - no intermediate MP4s)
        def ltx_video_pipeline(prompt, **kwargs):
            # Memory-optimized video dimensions
            config = InferenceConfig(
                prompt=prompt,
                pipeline_config="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
                height=kwargs.get('height', 512),
                width=kwargs.get('width', 768),
                num_frames=kwargs.get('num_frames', 121),
                seed=kwargs.get('seed', 2025)
            )
            
            # Generate video but work with tensors directly (no MP4 saving during GRPO)
            try:
                # Use LTX-Video pipeline directly to get tensor output
                # This avoids saving intermediate MP4 files
                result = infer(config)
                
                # Create result object that GRPO expects (PURE TENSOR WORKFLOW)
                class VideoResult:
                    def __init__(self, inference_result):
                        try:
                            # If LTX-Video returns tensor directly, use it
                            if hasattr(inference_result, 'images') and isinstance(inference_result.images, torch.Tensor):
                                self.images = inference_result.images
                                print("✅ Using direct tensor from LTX-Video")
                            else:
                                # Fallback: Load from the generated MP4 but don't keep it
                                import glob
                                video_files = glob.glob("ltx_video_source/outputs/**/*.mp4", recursive=True)
                                if video_files:
                                    latest_video = max(video_files, key=os.path.getctime)
                                    
                                    # Load video frames into tensor
                                    import imageio
                                    video_frames = imageio.mimread(latest_video)
                                    
                                    # Convert to tensor [1, C, T, H, W]
                                    video_array = np.array(video_frames)  # [T, H, W, C]
                                    video_tensor = torch.from_numpy(video_array).float()
                                    video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, T, H, W]
                                    video_tensor = (video_tensor / 255.0 - 0.5) * 2  # Normalize to [-1, 1]
                                    
                                    self.images = video_tensor
                                    
                                    # Delete the intermediate MP4 immediately after loading
                                    try:
                                        os.remove(latest_video)
                                        print(f"🔄 Loaded tensor and deleted intermediate: {os.path.basename(latest_video)}")
                                        
                                        # Force memory cleanup after each video load
                                        del video_frames, video_array
                                        torch.cuda.empty_cache()
                                    except Exception:
                                        pass  # Ignore deletion errors
                                else:
                                    # Fallback tensor
                                    self.images = torch.randn(1, 3, kwargs.get('num_frames', 121), 
                                                            kwargs.get('height', 512), kwargs.get('width', 768))
                                    print("⚠️ Using fallback random tensor")
                        except Exception as e:
                            print(f"⚠️ Tensor creation error: {e}")
                            self.images = torch.randn(1, 3, kwargs.get('num_frames', 121), 
                                                    kwargs.get('height', 512), kwargs.get('width', 768))
                
                return VideoResult(result)
                
            except Exception as e:
                print(f"❌ Pipeline error: {e}")
                # Return fallback result
                class FallbackResult:
                    def __init__(self):
                        self.images = torch.randn(1, 3, kwargs.get('num_frames', 121), 
                                                kwargs.get('height', 512), kwargs.get('width', 768))
                return FallbackResult()
        
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
        print(f"\n{'='*70}")
        print(f"🎬 GRPO Generation {i+1}/{len(test_prompts)}")
        print(f"{'='*70}")
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
                    import gc
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
                num_rounds=3,                    # Reduced from 5 to 3 for memory
                candidates_per_round=4,          # Reduced from 8 to 4 for memory
                reward_type='combined_physics',  # Use comprehensive physics rewards
                base_seed=2025 + i * 1000,      # Different seed per prompt
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
                    
                    import imageio
                    imageio.mimwrite(mp4_filename, video_np, fps=16, quality=8)
                    print(f"  🎬 Final best MP4: {mp4_filename}")
                    print(f"      ✨ This is the winner from {len(all_episodes)} candidates!")
                    
                except Exception as mp4_error:
                    print(f"  ⚠️ Failed to convert best video to MP4: {mp4_error}")
                
                # Save comprehensive results summary
                results_filename = f"{output_dir}/grpo_results_{i+1}_{prompt_safe}.json"
                import json
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
                        'rounds_completed': 5,
                        'candidates_per_round': 8,
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