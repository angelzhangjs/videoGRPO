# How to Call GRPO Search with LTX-Video Model

## Overview

You have **3 existing GRPO implementations** ready to use with the LTX-Video model. No code modifications needed - just import and call!

---

## Implementation 1: GRPOSearchWithLTXVideo (Recommended)

**Location:** `cursor/ltx_grpo_integration.py`

This is the most complete integration - it hooks directly into LTX-Video's inference pipeline.

### Full Example

```python
#!/usr/bin/env python3
"""
Example: GRPO Search with LTX-Video
"""

import sys
sys.path.append('cursor')
sys.path.append('ltx_video_source')

import torch
from ltx_grpo_integration import LTXVideoGRPOGenerator, GRPOSearchWithLTXVideo

# ============================================================
# Step 1: Define Reward Function
# ============================================================

def simple_reward_function(video: torch.Tensor, prompt: str) -> dict:
    """
    Simple reward function that evaluates video quality
    
    Args:
        video: Generated video tensor [C, T, H, W]
        prompt: Text prompt used for generation
    
    Returns:
        dict with 'reward' key and optional detailed metrics
    """
    C, T, H, W = video.shape
    
    # Example metrics
    motion_score = torch.diff(video, dim=1).abs().mean().item()
    variance_score = video.var().item()
    temporal_consistency = 1.0 / (1.0 + torch.diff(video, dim=1).var().item())
    
    # Combined reward
    reward = (
        0.4 * min(motion_score * 10, 1.0) +
        0.3 * min(variance_score * 5, 1.0) +
        0.3 * temporal_consistency
    )
    
    return {
        'reward': reward,
        'reward_info': {
            'motion': motion_score,
            'variance': variance_score,
            'temporal_consistency': temporal_consistency,
        }
    }


# ============================================================
# Step 2: Initialize GRPO System
# ============================================================

# Initialize LTX-Video generator
ltx_generator = LTXVideoGRPOGenerator(
    pipeline_config_path="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
    device="cuda",
    precision="bfloat16"
)

# Initialize GRPO search
grpo_search = GRPOSearchWithLTXVideo(
    ltx_generator=ltx_generator,
    reward_calculator=simple_reward_function
)


# ============================================================
# Step 3: Run GRPO Search
# ============================================================

prompt = "A red ornament swings slowly from a tree branch"

# Base configuration
base_config = {
    'height': 320,
    'width': 512,
    'num_frames': 160,
    'frame_rate': 30,
    'num_inference_steps': 40,
    'guidance_scale': 7.5,
    'seed': 2025,
}

# Run GRPO search
candidates = grpo_search.grpo_search(
    prompt=prompt,
    num_candidates_per_round=4,  # Generate 4 videos per round
    num_rounds=3,                 # 3 rounds of refinement
    base_config=base_config
)

# ============================================================
# Step 4: Get Results
# ============================================================

# Best video (highest reward)
best_candidate = candidates[0]
best_video = best_candidate['video']
best_reward = best_candidate['total_reward']
best_config = best_candidate['config']

print(f"\n{'='*70}")
print(f"BEST VIDEO FOUND:")
print(f"Reward: {best_reward:.4f}")
print(f"Config: {best_config}")
print(f"{'='*70}")

# Save best video
import imageio
video_np = best_video.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
video_np = (video_np * 255).astype('uint8')

with imageio.get_writer('grpo_best_video.mp4', fps=30) as writer:
    for frame in video_np:
        writer.append_data(frame)

print("✅ Best video saved to grpo_best_video.mp4")

# Print all candidates ranked by reward
print(f"\n{'='*70}")
print("ALL CANDIDATES (ranked by reward):")
print(f"{'='*70}")
for i, cand in enumerate(candidates[:10]):  # Top 10
    print(f"{i+1}. Round {cand['round']+1}, "
          f"Reward: {cand['total_reward']:.4f}, "
          f"Guidance: {cand['config']['guidance_scale']:.1f}, "
          f"Steps: {cand['config']['num_inference_steps']}")
```

### Output Example

```
======================================================================
GRPO Round 1/3
======================================================================

--- Candidate 1/4 ---
Generating video...
Reward: 0.4523

--- Candidate 2/4 ---
Generating video...
Reward: 0.5234

--- Candidate 3/4 ---
Generating video...
Reward: 0.4891

--- Candidate 4/4 ---
Generating video...
Reward: 0.3987

✅ Round 1 Best Reward: 0.5234
Updated config: guidance=9.0, steps=40

======================================================================
GRPO Round 2/3
======================================================================
...

======================================================================
GRPO Search Complete!
Best Overall Reward: 0.6789
======================================================================

BEST VIDEO FOUND:
Reward: 0.6789
Config: {'height': 320, 'width': 512, 'num_frames': 160, ...}
======================================================================

✅ Best video saved to grpo_best_video.mp4

======================================================================
ALL CANDIDATES (ranked by reward):
======================================================================
1. Round 3, Reward: 0.6789, Guidance: 10.5, Steps: 50
2. Round 2, Reward: 0.6234, Guidance: 9.0, Steps: 40
3. Round 3, Reward: 0.5987, Guidance: 8.5, Steps: 45
...
```

---

## Implementation 2: VideoGRPOSearcher (Framework Style)

**Location:** `cursor/video_grpo_framework.py`

More flexible framework-style implementation.

### Example

```python
#!/usr/bin/env python3
import sys
sys.path.append('cursor')
sys.path.append('ltx_video_source')

import torch
from video_grpo_framework import VideoGRPOSearcher
from ltx_grpo_integration import LTXVideoGRPOGenerator

# Define reward function
def reward_function(video: torch.Tensor, prompt: str) -> float:
    """Returns single reward score"""
    motion = torch.diff(video, dim=1).abs().mean().item()
    return min(motion * 10, 1.0)

# Initialize
generator = LTXVideoGRPOGenerator(
    pipeline_config_path="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
    device="cuda"
)

grpo_searcher = VideoGRPOSearcher(
    video_generator=generator,
    reward_function=reward_function
)

# Run GRPO search
results = grpo_searcher.grpo_search(
    prompts=["A pumpkin rolls down a path"],
    num_videos_per_prompt=4,
    num_iterations=3,
    generation_config={
        'height': 320,
        'width': 512,
        'num_frames': 160,
        'frame_rate': 30,
        'base_seed': 2025
    },
    device=torch.device("cuda")
)

# Get best from final iteration
best_episodes = results['iteration_3']['best_episodes']
best_video = best_episodes[0].video_frames
best_reward = best_episodes[0].reward

print(f"Best reward: {best_reward:.4f}")
```

---

## Implementation 3: Inference Search (Beam Search Style)

**Location:** `cursor/inference_search.py`

Beam search and guided search implementations.

### Example

```python
#!/usr/bin/env python3
import sys
sys.path.append('cursor')

from inference_search import LTXVideoSearchGenerator, SearchConfig

# Initialize
generator = LTXVideoSearchGenerator(
    model_name_or_path="Lightricks/LTX-Video",
    config_path="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
    device="cuda"
)

# Search config
search_config = SearchConfig(
    beam_width=4,
    search_steps=10,
    temperature=0.8
)

# Generation config
gen_config = {
    'height': 320,
    'width': 512,
    'num_frames': 160,
    'num_inference_steps': 40,
    'guidance_scale': 7.5,
    'seed': 2025
}

# Run guided search
best_video = generator.guided_search_generation(
    prompt="A silver bell spins slowly above the fireplace",
    search_config=search_config,
    generation_config=gen_config
)

print(f"Best video shape: {best_video.shape}")
```

---

## Advanced: Custom Reward Functions

### 1. Multi-Component Reward

```python
def advanced_reward_function(video: torch.Tensor, prompt: str) -> dict:
    """
    Advanced reward with multiple components
    """
    C, T, H, W = video.shape
    
    # Component 1: Motion Quality
    motion = torch.diff(video, dim=1).abs().mean().item()
    motion_score = min(motion * 10, 1.0)
    
    # Component 2: Visual Complexity
    variance = video.var().item()
    complexity_score = min(variance * 5, 1.0)
    
    # Component 3: Temporal Consistency
    temporal_diff = torch.diff(video, dim=1)
    consistency = 1.0 / (1.0 + temporal_diff.var().item() * 100)
    
    # Component 4: Color Diversity
    color_std = video.std(dim=[-2, -1]).mean().item()
    color_score = min(color_std * 2, 1.0)
    
    # Component 5: Edge Sharpness
    edges_h = torch.diff(video, dim=-2).abs().mean().item()
    edges_v = torch.diff(video, dim=-1).abs().mean().item()
    sharpness = min((edges_h + edges_v) * 10, 1.0)
    
    # Weighted combination
    weights = {
        'motion': 0.25,
        'complexity': 0.20,
        'consistency': 0.20,
        'color': 0.15,
        'sharpness': 0.20
    }
    
    scores = {
        'motion': motion_score,
        'complexity': complexity_score,
        'consistency': consistency,
        'color': color_score,
        'sharpness': sharpness
    }
    
    total_reward = sum(scores[k] * weights[k] for k in scores)
    
    return {
        'reward': total_reward,
        'reward_info': scores
    }
```

### 2. Cross-Attention Fusion Reward (Your System!)

```python
#!/usr/bin/env python3
"""
Use your cross-attention fusion system as reward
"""

import sys
sys.path.append('cursor')
sys.path.append('ltx_video_source')

import torch
from ltx_grpo_integration import LTXVideoGRPOGenerator, GRPOSearchWithLTXVideo

# Import your fusion models (if available)
# from unified_spatial_dino_encoder import DINOIntegratedSpatialEncoder
# from cross_attention_fusion_detailed import AdvancedCrossAttentionFusion

def fusion_based_reward(video: torch.Tensor, prompt: str) -> dict:
    """
    Reward based on cross-attention fusion quality
    
    This would use your DINO + Spatial + Fusion system
    """
    # Pseudocode (replace with actual implementation):
    
    # 1. Extract DINO features
    # dino_features = dino_model(video)  # [T, 768]
    
    # 2. Extract spatial features
    # spatial_features = spatial_encoder(video)  # [T, 512]
    
    # 3. Cross-attention fusion
    # fusion_result = fusion_model(dino_features, spatial_features)
    
    # 4. Compute quality scores
    # object_quality = fusion_result['object_quality'].mean()
    # spatial_quality = fusion_result['spatial_quality'].mean()
    # fusion_quality = fusion_result['fusion_quality'].mean()
    
    # 5. Combined reward
    # reward = (
    #     0.35 * object_quality +
    #     0.35 * spatial_quality +
    #     0.30 * fusion_quality
    # )
    
    # For now, simple implementation
    motion = torch.diff(video, dim=1).abs().mean().item()
    reward = min(motion * 10, 1.0)
    
    return {
        'reward': reward,
        'reward_info': {
            # 'object_quality': object_quality,
            # 'spatial_quality': spatial_quality,
            # 'fusion_quality': fusion_quality,
            'motion': motion
        }
    }

# Use with GRPO
generator = LTXVideoGRPOGenerator(
    pipeline_config_path="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
    device="cuda"
)

grpo = GRPOSearchWithLTXVideo(
    ltx_generator=generator,
    reward_calculator=fusion_based_reward  # Use your fusion reward!
)

candidates = grpo.grpo_search(
    prompt="A wrapped gift slides across the floor",
    num_candidates_per_round=4,
    num_rounds=3
)
```

### 3. Gemini VLM Reward (High Quality!)

```python
def gemini_vlm_reward(video: torch.Tensor, prompt: str, api_key: str) -> dict:
    """
    Use Gemini VLM to evaluate video quality
    """
    import google.generativeai as genai
    from PIL import Image
    import io
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Convert video to frames
    C, T, H, W = video.shape
    video_np = video.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
    video_np = (video_np * 255).astype('uint8')
    
    # Sample frames (Gemini has token limits)
    sample_indices = [0, T//4, T//2, 3*T//4, T-1]
    frames = [Image.fromarray(video_np[i]) for i in sample_indices]
    
    # Ask Gemini to evaluate
    evaluation_prompt = f"""
    Evaluate this video sequence (5 frames shown) based on the prompt: "{prompt}"
    
    Rate on a scale of 0-10 for:
    1. Motion quality (smooth, natural movement)
    2. Visual coherence (consistent appearance)
    3. Prompt alignment (matches the text description)
    4. Technical quality (sharpness, colors, artifacts)
    
    Provide scores as: motion=X, coherence=X, alignment=X, technical=X
    """
    
    # Call Gemini
    response = model.generate_content([evaluation_prompt] + frames)
    
    # Parse response (simplified - would need robust parsing)
    text = response.text
    
    # Extract scores (simplified)
    try:
        scores = {}
        for line in text.split('\n'):
            if '=' in line:
                key, val = line.split('=')
                scores[key.strip()] = float(val.strip()) / 10.0
        
        reward = sum(scores.values()) / len(scores)
    except:
        reward = 0.5  # Default if parsing fails
    
    return {
        'reward': reward,
        'reward_info': scores,
        'gemini_response': text
    }
```

---

## Configuration Options

### Generation Parameters

```python
config = {
    # Video dimensions
    'height': 320,              # Frame height (multiple of 32)
    'width': 512,               # Frame width (multiple of 32)
    'num_frames': 160,          # Number of frames
    'frame_rate': 30,           # FPS
    
    # Generation quality
    'num_inference_steps': 40,  # More steps = higher quality, slower
    'guidance_scale': 7.5,      # Higher = closer to prompt (3-15)
    
    # Randomness
    'seed': 2025,               # For reproducibility
    
    # Prompts
    'negative_prompt': "worst quality, inconsistent motion, blurry",
}
```

### GRPO Search Parameters

```python
grpo_params = {
    'num_candidates_per_round': 4,  # Videos generated each round
    'num_rounds': 3,                 # Number of refinement rounds
    'base_config': config,           # Starting configuration
}

# Total videos generated: num_candidates_per_round * num_rounds
# For example: 4 * 3 = 12 videos total
```

---

## Full Working Script

Save this as `run_grpo.py`:

```python
#!/usr/bin/env python3
"""
Complete GRPO Search Example with LTX-Video
"""

import sys
sys.path.append('cursor')
sys.path.append('ltx_video_source')

import torch
import imageio
from pathlib import Path
from ltx_grpo_integration import LTXVideoGRPOGenerator, GRPOSearchWithLTXVideo


def reward_function(video: torch.Tensor, prompt: str) -> dict:
    """Simple but effective reward function"""
    C, T, H, W = video.shape
    
    # Motion score
    motion = torch.diff(video, dim=1).abs().mean().item()
    motion_score = min(motion * 10, 1.0)
    
    # Variance score
    variance = video.var().item()
    variance_score = min(variance * 5, 1.0)
    
    # Temporal consistency
    temporal_diff = torch.diff(video, dim=1)
    consistency = 1.0 / (1.0 + temporal_diff.var().item() * 100)
    
    # Combined
    reward = 0.4 * motion_score + 0.3 * variance_score + 0.3 * consistency
    
    return {
        'reward': reward,
        'reward_info': {
            'motion': motion_score,
            'variance': variance_score,
            'consistency': consistency
        }
    }


def main():
    print("="*70)
    print("GRPO Search with LTX-Video")
    print("="*70)
    
    # Initialize
    print("\n1. Initializing LTX-Video generator...")
    generator = LTXVideoGRPOGenerator(
        pipeline_config_path="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
        device="cuda",
        precision="bfloat16"
    )
    
    print("2. Initializing GRPO search...")
    grpo = GRPOSearchWithLTXVideo(
        ltx_generator=generator,
        reward_calculator=reward_function
    )
    
    # Configuration
    prompt = "A pumpkin rolls gently down a sloped path"
    base_config = {
        'height': 320,
        'width': 512,
        'num_frames': 160,
        'frame_rate': 30,
        'num_inference_steps': 40,
        'guidance_scale': 7.5,
        'seed': 2025,
    }
    
    print(f"\n3. Running GRPO search for: '{prompt}'")
    print(f"   Config: {base_config}")
    
    # Run GRPO
    candidates = grpo.grpo_search(
        prompt=prompt,
        num_candidates_per_round=4,
        num_rounds=3,
        base_config=base_config
    )
    
    # Save best video
    print("\n4. Saving best video...")
    best = candidates[0]
    best_video = best['video']
    
    output_path = Path("grpo_outputs")
    output_path.mkdir(exist_ok=True)
    
    video_np = best_video.permute(1, 2, 3, 0).cpu().numpy()
    video_np = (video_np * 255).astype('uint8')
    
    output_file = output_path / "best_video.mp4"
    with imageio.get_writer(output_file, fps=30) as writer:
        for frame in video_np:
            writer.append_data(frame)
    
    print(f"✅ Best video saved to: {output_file}")
    print(f"   Reward: {best['total_reward']:.4f}")
    print(f"   Config: {best['config']}")
    
    # Save top 3
    print("\n5. Saving top 3 candidates...")
    for i, cand in enumerate(candidates[:3]):
        video_np = cand['video'].permute(1, 2, 3, 0).cpu().numpy()
        video_np = (video_np * 255).astype('uint8')
        
        output_file = output_path / f"candidate_{i+1}_reward_{cand['total_reward']:.4f}.mp4"
        with imageio.get_writer(output_file, fps=30) as writer:
            for frame in video_np:
                writer.append_data(frame)
        
        print(f"   Saved: {output_file}")
    
    print("\n" + "="*70)
    print("GRPO Search Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
```

Run it:
```bash
python run_grpo.py
```

---

## Summary

### Available Implementations

| Implementation | Location | Best For |
|----------------|----------|----------|
| `GRPOSearchWithLTXVideo` | `cursor/ltx_grpo_integration.py` | **Recommended** - Full LTX integration |
| `VideoGRPOSearcher` | `cursor/video_grpo_framework.py` | Framework-style, flexible |
| `LTXVideoSearchGenerator` | `cursor/inference_search.py` | Beam search style |

### Quick Start

```python
# 1. Import
from ltx_grpo_integration import LTXVideoGRPOGenerator, GRPOSearchWithLTXVideo

# 2. Initialize
generator = LTXVideoGRPOGenerator("ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml")
grpo = GRPOSearchWithLTXVideo(generator, reward_function)

# 3. Search
candidates = grpo.grpo_search(prompt="...", num_candidates_per_round=4, num_rounds=3)

# 4. Get best
best_video = candidates[0]['video']
```

**No code modifications needed - just import and use!** 🚀

