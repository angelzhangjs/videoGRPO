# Searching Over Latent Space in LTX-Video

## Overview

In LTX-Video, the **latent space** is the compressed representation where the diffusion process operates. Instead of generating directly in pixel space, the model works in this more efficient latent space.

---

## Understanding the Latent Space

### What is the Latent Space?

```
Pixel Space (Video):      [B, C=3, T=160, H=512, W=768]
                                   ↓ VAE Encode (Compress)
Latent Space (Compressed): [B, C=128, T=41, H=32, W=48]
                                   ↓ Diffusion Process
Denoised Latents:         [B, C=128, T=41, H=32, W=48]
                                   ↓ VAE Decode (Decompress)
Output Video:             [B, C=3, T=160, H=512, W=768]
```

### Compression Ratios

From the VAE configuration:

```python
# Spatial compression
spatial_factor = 16          # Height and width compressed 16x
patch_size = 4               # Additional patchification
spatial_downscale = 16       # Final spatial compression

# Temporal compression
temporal_factor = 4          # Time compressed ~4x
temporal_downscale = 4

# Channel expansion
pixel_channels = 3           # RGB
latent_channels = 128        # Expanded to 128 channels
```

### Latent Shape Calculation

```python
def calculate_latent_shape(video_shape):
    """
    Calculate latent shape from video shape
    
    Args:
        video_shape: (batch, channels=3, frames, height, width)
    
    Returns:
        latent_shape: (batch, latent_channels, latent_frames, latent_height, latent_width)
    """
    B, C, T, H, W = video_shape
    
    # From LTX-Video VAE config
    latent_channels = 128
    spatial_compression = 16
    temporal_compression = 4
    
    latent_shape = (
        B,                           # Batch size stays same
        latent_channels,             # 3 → 128 channels
        (T + 3) // temporal_compression,  # ~T/4 frames
        H // spatial_compression,    # H/16 height
        W // spatial_compression     # W/16 width
    )
    
    return latent_shape

# Example
video_shape = (1, 3, 160, 512, 768)
latent_shape = calculate_latent_shape(video_shape)
print(latent_shape)  # (1, 128, 41, 32, 48)
```

---

## Accessing Latent Space in LTX-Video

### Method 1: Get Latents from Pipeline

```python
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline

# Initialize pipeline
pipeline = ...  # Your LTX-Video pipeline

# Generation with latents exposed
output = pipeline(
    prompt="Your prompt",
    height=512,
    width=768,
    num_frames=160,
    output_type="latent",  # ← KEY: Return latents instead of decoded video
    return_dict=True
)

latents = output.images  # Shape: [1, 128, 41, 32, 48]
```

### Method 2: Encode Video to Latents

```python
import torch
from ltx_video.models.autoencoders.vae_encode import vae_encode

# You have a video tensor
video = torch.randn(1, 3, 160, 512, 768)  # [B, C, T, H, W]

# Encode to latents
latents = vae_encode(
    video,
    vae=pipeline.vae,
    vae_per_channel_normalize=True
)

print(latents.shape)  # [1, 128, 41, 32, 48]
```

### Method 3: Initialize Random Latents

```python
def initialize_latents(pipeline, generation_config):
    """
    Initialize random latent noise for generation
    """
    batch_size = 1
    latent_channels = pipeline.vae.config.latent_channels  # 128
    
    # Calculate latent dimensions
    temporal_compression = 4  # From VAE config
    spatial_compression = 16
    
    latent_shape = (
        batch_size,
        latent_channels,
        (generation_config['num_frames'] + 3) // temporal_compression,
        generation_config['height'] // spatial_compression,
        generation_config['width'] // spatial_compression
    )
    
    # Create random latents
    latents = torch.randn(
        latent_shape,
        device=pipeline.device,
        dtype=torch.float32
    )
    
    # Scale by scheduler's initial noise sigma
    latents = latents * pipeline.scheduler.init_noise_sigma
    
    return latents

# Usage
latents = initialize_latents(pipeline, {
    'num_frames': 160,
    'height': 512,
    'width': 768
})
```

---

## Latent Space Search Methods

You have **3 existing implementations** for searching over latent space!

### Implementation 1: LatentPolicyOptimizer (Best for GRPO!)

**Location:** `cursor/latent_policy_update.py`

This optimizes latent noise directly using gradient descent to maximize reward.

```python
#!/usr/bin/env python3
"""
Example: Latent Space Optimization for Video Generation
"""

import sys
sys.path.append('cursor')
sys.path.append('ltx_video_source')

import torch
from latent_policy_update import LatentPolicyOptimizer

# Your LTX-Video pipeline
from ltx_video.inference import create_ltx_video_pipeline
pipeline = create_ltx_video_pipeline(...)

# Define reward function
def reward_function(video: torch.Tensor, prompt: str) -> dict:
    """Evaluate video quality"""
    motion = torch.diff(video, dim=1).abs().mean().item()
    reward = min(motion * 10, 1.0)
    return {'reward': reward}

# Initialize latent optimizer
latent_optimizer = LatentPolicyOptimizer(
    video_pipeline=pipeline,
    reward_function=reward_function
)

# Optimize latents to maximize reward
optimized_video = latent_optimizer.optimize_latent_policy(
    prompt="A pumpkin rolls down a path",
    num_optimization_steps=10,   # Gradient descent steps
    learning_rate=0.01,           # Step size
    num_candidates=4,             # Population size
    generation_config={
        'height': 512,
        'width': 768,
        'num_frames': 160,
        'guidance_scale': 7.5,
        'num_inference_steps': 40
    }
)

print(f"Optimized video shape: {optimized_video.shape}")
```

#### How It Works

```python
# From latent_policy_update.py lines 37-99

# 1. Initialize random latents (this is the "policy")
latent_noise = torch.randn(latent_shape, requires_grad=True)  # ← Optimizable!

# 2. Create optimizer
optimizer = torch.optim.Adam([latent_noise], lr=0.01)

# 3. Optimization loop
for step in range(num_optimization_steps):
    # Generate video from current latents
    videos = pipeline(
        latents=latent_noise,  # ← Use our optimizable latents
        ...
    )
    
    # Compute rewards
    rewards = [reward_function(video, prompt) for video in videos]
    
    # Compute policy gradient (REINFORCE)
    advantages = rewards - rewards.mean()
    policy_loss = -advantages.mean()  # Maximize reward
    
    # Backpropagate through latents
    policy_loss.backward()
    
    # Update latents (policy update!)
    optimizer.step()
    
    # Latents are now optimized to produce higher-reward videos!
```

**Key Insight:** The latent noise IS the policy! We optimize it directly using gradient descent.

### Implementation 2: VideoPathSearcher (Path-Based Search)

**Location:** `cursor/video_path_search.py`

This optimizes entire video sequences (paths) in latent space.

```python
#!/usr/bin/env python3
import sys
sys.path.append('cursor')

from video_path_search import VideoPathSearcher

# Initialize
searcher = VideoPathSearcher(
    pipeline=pipeline,
    reward_function=reward_function
)

# Generate initial path candidates
paths = searcher.generate_path_candidates(
    prompt="A wrapped gift slides across the floor",
    num_paths=8,           # Generate 8 different paths
    path_length=160,       # 160 frames
    generation_config={...}
)

# Optimize paths in latent space
optimized_paths = searcher.optimize_path_sequence(
    initial_paths=paths,
    num_optimization_steps=15,
    learning_rate=0.01
)

# Get best path
best_path = max(optimized_paths, key=lambda p: p.path_reward)
best_video = best_path.frames
print(f"Best path reward: {best_path.path_reward:.4f}")
```

#### How It Works

```python
# From video_path_search.py lines 189-255

for path in initial_paths:
    # 1. Extract latent representation
    optimizable_latents = path.latent_path.clone()
    optimizable_latents.requires_grad = True  # Make optimizable
    
    # 2. Create optimizer for this path's latents
    optimizer = torch.optim.Adam([optimizable_latents], lr=0.01)
    
    # 3. Optimize
    for step in range(num_optimization_steps):
        # Decode latents → video frames
        decoded_frames = pipeline.vae.decode(optimizable_latents).sample
        
        # Evaluate full path (not just frames!)
        path_reward, frame_rewards = evaluate_path(decoded_frames, prompt)
        
        # Maximize path reward
        loss = -torch.tensor(path_reward, requires_grad=True)
        loss.backward()
        
        # Update latent path
        optimizer.step()
        
    # Path latents are now optimized!
```

**Key Insight:** Optimizes complete sequences, considering temporal coherence.

### Implementation 3: Direct Latent Manipulation

You can also directly manipulate latents for controlled generation.

```python
#!/usr/bin/env python3
"""
Example: Direct Latent Space Manipulation
"""

import torch
import numpy as np

def interpolate_latents(latent1, latent2, alpha=0.5):
    """
    Interpolate between two latent representations
    
    Args:
        latent1: First latent [1, 128, T, H, W]
        latent2: Second latent [1, 128, T, H, W]
        alpha: Interpolation factor (0=latent1, 1=latent2)
    
    Returns:
        Interpolated latent
    """
    return (1 - alpha) * latent1 + alpha * latent2


def latent_space_walk(pipeline, start_latent, num_steps=10, step_size=0.1):
    """
    Perform random walk in latent space
    
    Generates sequence of videos by gradually changing latents
    """
    current_latent = start_latent.clone()
    videos = []
    
    for step in range(num_steps):
        # Generate video from current latent
        video = pipeline(
            latents=current_latent,
            output_type="pt"
        ).frames
        
        videos.append(video)
        
        # Random step in latent space
        noise = torch.randn_like(current_latent) * step_size
        current_latent = current_latent + noise
        
        print(f"Step {step+1}/{num_steps} - Latent norm: {current_latent.norm():.2f}")
    
    return videos


def search_latent_directions(pipeline, base_latent, num_directions=8):
    """
    Search in multiple directions from a base latent
    """
    results = []
    
    for direction_idx in range(num_directions):
        # Create direction vector
        direction = torch.randn_like(base_latent)
        direction = direction / direction.norm()  # Normalize
        
        # Move along direction
        for magnitude in [0.0, 0.5, 1.0, 2.0]:
            modified_latent = base_latent + magnitude * direction
            
            # Generate video
            video = pipeline(
                latents=modified_latent,
                output_type="pt"
            ).frames
            
            # Evaluate
            reward = reward_function(video, prompt)['reward']
            
            results.append({
                'direction': direction_idx,
                'magnitude': magnitude,
                'video': video,
                'reward': reward
            })
    
    # Find best
    best = max(results, key=lambda x: x['reward'])
    return best, results


# Usage Example
pipeline = ...  # Your LTX-Video pipeline

# Initialize two random latents
latent1 = initialize_latents(pipeline, config)
latent2 = initialize_latents(pipeline, config)

# Interpolate
interpolated = interpolate_latents(latent1, latent2, alpha=0.3)

# Generate from interpolated latent
video = pipeline(
    latents=interpolated,
    prompt="Your prompt",
    output_type="pt"
).frames

# Or do a random walk
videos = latent_space_walk(pipeline, latent1, num_steps=10)

# Or search directions
best_result, all_results = search_latent_directions(
    pipeline, latent1, num_directions=8
)
```

---

## Full Example: GRPO with Latent Search

Combining GRPO with latent space optimization:

```python
#!/usr/bin/env python3
"""
Complete Example: GRPO Search Over Latent Space
"""

import sys
sys.path.append('cursor')
sys.path.append('ltx_video_source')

import torch
import imageio
from pathlib import Path

# Import components
from ltx_grpo_integration import LTXVideoGRPOGenerator, GRPOSearchWithLTXVideo
from latent_policy_update import LatentPolicyOptimizer


# ============================================================
# Step 1: Define Advanced Reward Function
# ============================================================

def advanced_reward_function(video: torch.Tensor, prompt: str) -> dict:
    """
    Multi-component reward for comprehensive evaluation
    """
    C, T, H, W = video.shape
    
    # Motion quality
    motion = torch.diff(video, dim=1).abs().mean().item()
    motion_score = min(motion * 10, 1.0)
    
    # Temporal consistency
    temporal_var = torch.diff(video, dim=1).var().item()
    consistency_score = 1.0 / (1.0 + temporal_var * 100)
    
    # Spatial quality
    spatial_var = video.var(dim=[-2, -1]).mean().item()
    spatial_score = min(spatial_var * 5, 1.0)
    
    # Overall reward
    reward = (
        0.4 * motion_score +
        0.3 * consistency_score +
        0.3 * spatial_score
    )
    
    return {
        'reward': reward,
        'reward_info': {
            'motion': motion_score,
            'consistency': consistency_score,
            'spatial': spatial_score
        }
    }


# ============================================================
# Step 2: Initialize System
# ============================================================

print("Initializing LTX-Video with GRPO + Latent Optimization...")

# Generator
generator = LTXVideoGRPOGenerator(
    pipeline_config_path="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
    device="cuda",
    precision="bfloat16"
)

# Latent optimizer
latent_optimizer = LatentPolicyOptimizer(
    video_pipeline=generator.pipeline,
    reward_function=advanced_reward_function
)


# ============================================================
# Step 3: Multi-Stage GRPO with Latent Refinement
# ============================================================

def grpo_with_latent_refinement(
    prompt: str,
    num_grpo_rounds: int = 3,
    candidates_per_round: int = 4,
    latent_optimization_steps: int = 5
):
    """
    GRPO search with latent space refinement
    
    Stage 1: GRPO search (parameter space)
    Stage 2: Latent optimization (latent space)
    Stage 3: Combined refinement
    """
    
    print(f"\n{'='*70}")
    print(f"Multi-Stage GRPO + Latent Search")
    print(f"{'='*70}")
    print(f"Prompt: {prompt}")
    
    all_results = []
    
    # Stage 1: GRPO Search in Parameter Space
    print(f"\n🔍 Stage 1: GRPO Parameter Search")
    print(f"{'='*70}")
    
    grpo = GRPOSearchWithLTXVideo(
        ltx_generator=generator,
        reward_calculator=advanced_reward_function
    )
    
    grpo_candidates = grpo.grpo_search(
        prompt=prompt,
        num_candidates_per_round=candidates_per_round,
        num_rounds=num_grpo_rounds,
        base_config={
            'height': 320,
            'width': 512,
            'num_frames': 160,
            'frame_rate': 30,
            'seed': 2025
        }
    )
    
    # Get top 3 from GRPO
    top_grpo = grpo_candidates[:3]
    
    # Stage 2: Latent Space Optimization
    print(f"\n🎯 Stage 2: Latent Space Optimization")
    print(f"{'='*70}")
    
    for i, candidate in enumerate(top_grpo):
        print(f"\nOptimizing latents for candidate {i+1}/3...")
        print(f"Initial reward: {candidate['total_reward']:.4f}")
        
        # Optimize in latent space
        optimized_video = latent_optimizer.optimize_latent_policy(
            prompt=prompt,
            num_optimization_steps=latent_optimization_steps,
            learning_rate=0.01,
            num_candidates=2,  # Small population
            generation_config=candidate['config']
        )
        
        # Evaluate optimized
        optimized_reward = advanced_reward_function(optimized_video, prompt)
        
        result = {
            'video': optimized_video,
            'config': candidate['config'],
            'grpo_reward': candidate['total_reward'],
            'optimized_reward': optimized_reward['reward'],
            'improvement': optimized_reward['reward'] - candidate['total_reward'],
            'stage': 'latent_optimized'
        }
        
        all_results.append(result)
        
        print(f"After latent optimization: {optimized_reward['reward']:.4f}")
        print(f"Improvement: {result['improvement']:+.4f}")
    
    # Sort by final reward
    all_results.sort(key=lambda x: x['optimized_reward'], reverse=True)
    
    print(f"\n{'='*70}")
    print(f"Search Complete!")
    print(f"{'='*70}")
    print(f"Best reward: {all_results[0]['optimized_reward']:.4f}")
    print(f"Total improvement from GRPO: {all_results[0]['improvement']:+.4f}")
    
    return all_results


# ============================================================
# Step 4: Run Search
# ============================================================

results = grpo_with_latent_refinement(
    prompt="A silver bell spins slowly above the fireplace",
    num_grpo_rounds=2,
    candidates_per_round=4,
    latent_optimization_steps=5
)


# ============================================================
# Step 5: Save Results
# ============================================================

output_dir = Path("latent_search_outputs")
output_dir.mkdir(exist_ok=True)

for i, result in enumerate(results[:3]):
    video_np = result['video'].permute(1, 2, 3, 0).cpu().numpy()
    video_np = (video_np * 255).astype('uint8')
    
    filename = f"rank_{i+1}_reward_{result['optimized_reward']:.4f}.mp4"
    filepath = output_dir / filename
    
    with imageio.get_writer(filepath, fps=30) as writer:
        for frame in video_np:
            writer.append_data(frame)
    
    print(f"Saved: {filepath}")

print(f"\n✅ All videos saved to {output_dir}/")
```

---

## Advanced Techniques

### 1. Latent Space Arithmetic

```python
def latent_arithmetic(pipeline, prompt1, prompt2, operation='add', alpha=0.5):
    """
    Perform arithmetic operations in latent space
    """
    # Generate latents for two prompts
    latent1 = pipeline(prompt=prompt1, output_type="latent").images
    latent2 = pipeline(prompt=prompt2, output_type="latent").images
    
    # Operations
    if operation == 'add':
        result_latent = latent1 + alpha * latent2
    elif operation == 'subtract':
        result_latent = latent1 - alpha * latent2
    elif operation == 'interpolate':
        result_latent = (1 - alpha) * latent1 + alpha * latent2
    
    # Generate from combined latent
    video = pipeline(latents=result_latent, output_type="pt").frames
    return video

# Example: "pumpkin rolling" + "leaves falling"
video = latent_arithmetic(
    pipeline,
    "A pumpkin rolling down a path",
    "Leaves falling from trees",
    operation='add',
    alpha=0.3
)
```

### 2. Latent Space Clustering

```python
def cluster_latent_space(pipeline, prompts, num_samples=5):
    """
    Generate samples and cluster in latent space
    """
    from sklearn.cluster import KMeans
    
    all_latents = []
    all_prompts = []
    
    for prompt in prompts:
        for seed in range(num_samples):
            latent = pipeline(
                prompt=prompt,
                seed=seed,
                output_type="latent"
            ).images
            
            all_latents.append(latent.flatten().cpu().numpy())
            all_prompts.append(prompt)
    
    # Cluster
    kmeans = KMeans(n_clusters=len(prompts))
    clusters = kmeans.fit_predict(all_latents)
    
    return clusters, kmeans.cluster_centers_
```

### 3. Gradient-Based Latent Search

```python
def gradient_based_search(pipeline, prompt, target_property, num_steps=50):
    """
    Search latents using gradient descent toward target property
    """
    # Initialize
    latent = initialize_latents(pipeline, config)
    latent.requires_grad = True
    
    optimizer = torch.optim.Adam([latent], lr=0.05)
    
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Generate
        video = pipeline(latents=latent, output_type="pt").frames
        
        # Compute target property
        # (e.g., motion amount, color distribution, etc.)
        current_property = compute_property(video)
        
        # Loss = distance to target
        loss = (current_property - target_property) ** 2
        loss.backward()
        
        optimizer.step()
        
        print(f"Step {step}: property={current_property:.3f}, target={target_property:.3f}")
    
    return latent.detach()
```

---

## Summary

### Latent Space in LTX-Video

| Aspect | Details |
|--------|---------|
| **Shape** | `[batch, 128, T/4, H/16, W/16]` |
| **Channels** | 128 (from 3 RGB) |
| **Temporal Compression** | ~4x |
| **Spatial Compression** | 16x |
| **Access Method** | `output_type="latent"` in pipeline |
| **Function** | `prepare_latents()` in `pipeline_ltx_video.py` |

### Available Search Methods

| Method | Location | Best For |
|--------|----------|----------|
| `LatentPolicyOptimizer` | `cursor/latent_policy_update.py` | **GRPO-style optimization** |
| `VideoPathSearcher` | `cursor/video_path_search.py` | Path-based search |
| Direct manipulation | Custom code | Interpolation, arithmetic |

### Key Functions

```python
# Initialize latents
latents = pipeline.prepare_latents(...)

# Encode video → latents
latents = vae_encode(video, vae)

# Decode latents → video
video = vae.decode(latents).sample

# Optimize latents
optimizer = LatentPolicyOptimizer(pipeline, reward_fn)
best_video = optimizer.optimize_latent_policy(...)
```

**The latent space is where all the magic happens** - it's the compressed representation that the diffusion model learns to denoise, and searching over it gives you direct control over video generation! 🎬

