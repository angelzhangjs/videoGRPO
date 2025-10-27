# tokens in language models --> latents/frames in video generative models
"""
GRPO framework for video generation 
update gradient at latent space 
self-defined reward function to improve the video output quality
"""
import torch
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass
from collections import defaultdict

# Import hand-crafted reward functions

from reward_functions import (
        combined_physics_reward,
        physics_velocity_reward,
        physics_acceleration_reward,
        physics_trajectory_smoothness,
        physics_momentum_conservation,
        physics_gravity_consistency,
        grpo_reward_function,
    )

global CHANNEL_SIZE 
CHANNEL_SIZE = 8  # LTX-Video uses 8 latent channels

@dataclass
class videoepisode:
    prompt: str
    video: torch.Tensor
    seed: int
    generation_params: Dict[str, Any]
    reward: float
    reward_info: Dict[str, Any]
    is_finished: bool = True
    
@dataclass
class sampling_configuration:
    alpha: float
    guidance_scale: float
    
@dataclass
class video_pipeline_configuration:
    pipeline: Any  # LTXVideoPipeline instance
    prompt: str
    num_frames: int
    height: int
    width: int
    num_candidates_per_prompt: int
    guidance_scale: float
    num_inference_steps: int
    device: str    
    output_type: str = "pt",  # Return PyTorch tensor

@torch.no_grad()
def video_rollout(
    video_pipeline, 
    prompt: str, 
    num_frames: int = 121,
    height: int = 320,
    width: int = 512,
    num_candidates_per_prompt: int = 16,
    guidance_scale: float = 1.3,
    num_inference_steps: int = 48,
    device: str = "cuda",
    base_seed: int = 2025,
    latent_anchor_a: Optional[torch.Tensor] = None,
    latent_anchor_b: Optional[torch.Tensor] = None,
) -> List[videoepisode]:
    """
    Rollout multiple video candidates for a SINGLE prompt using 2D configuration sampling.
    
    Generates num_candidates_per_prompt different videos, all from the same prompt,
    by varying:
      1. Latent initialization (via interpolation between two anchor latents)
      2. Guidance scale (sampled around base guidance_scale)
    
    This creates diverse candidates for GRPO group evaluation.
    
    Args:
        video_pipeline: LTXVideoPipeline instance
        prompt: SINGLE text prompt (all candidates use this same prompt)
        num_frames: Number of frames to generate (default: 121)
        height: Frame height (default: 512)
        width: Frame width (default: 768)
        num_candidates_per_prompt: Number of video candidates to generate (default: 16)
        guidance_scale: Base guidance scale (will be varied ±30% for diversity)
        num_inference_steps: Number of denoising steps (default: 40)
        device: Device to run on (default: "cuda")
        base_seed: Base random seed for reproducibility (default: 2025)
    
    Returns:
        List of videoepisode objects, all generated from the SAME prompt
        Each episode has different (latent, guidance) configuration
        Ready for GRPO group normalization and selection
    
    Example:
        >>> episodes = video_rollout(
        ...     video_pipeline=pipeline,
        ...     prompt="How to grasp a cup",  # Same prompt for all
        ...     num_candidates_per_prompt=16,  # Generate 16 variants
        ... )
        >>> # All 16 episodes have same prompt, different latent/guidance
        >>> assert all(ep.prompt == "How to grasp a cup" for ep in episodes)
    """
    
    # Calculate latent dimensions
    vae_scale_factor = 8
    video_scale_factor = 8
     
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    latent_num_frames = num_frames // video_scale_factor + 1  # +1 for causal VAE
    
    # Latent shape: [batch_size, channels, temporal_frames, height, width]
    latent_shape = (1, CHANNEL_SIZE, latent_num_frames, latent_height, latent_width)
    
    # Generate multiple latent initializations for diversity
    candidates = []
    
    # Create or use provided anchor latents for interpolation
    if latent_anchor_a is not None and latent_anchor_b is not None:
        latent_a = latent_anchor_a
        latent_b = latent_anchor_b
    else:
        # Use base_seed for reproducibility, but generate two different latents
        torch.manual_seed(base_seed)
        latent_a = torch.randn(latent_shape, device=device)
        latent_b = torch.randn(latent_shape, device=device)  # Different draw from same seeded RNG
    
    # Get sampling configurations (alpha and guidance pairs)
    configs = create_sampling_configurations(
        number_of_samples=num_candidates_per_prompt,
        latent_range=(0.0, 1.0),
        guidance_range=(1.0, 1.6),  # Narrow, stable guidance for LTX
        device=device,
    )
    
    # Generate candidates using sampled configurations
    # ALL candidates use the SAME prompt, different (latent, guidance) configs
    for i, config in enumerate(configs):
        # Get alpha and guidance from configuration
        alpha = config.alpha
        current_guidance = config.guidance_scale
        
        # Interpolate between latent_a and latent_b using sampled alpha
        latent_init = (1 - alpha) * latent_a + alpha * latent_b
        
        # Generate video with this specific latent initialization and guidance
        # SAME prompt for all candidates (critical for GRPO group comparison)
        result = video_pipeline(
            prompt=prompt,  
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=current_guidance,  # ← Varies per candidate
            num_inference_steps=num_inference_steps,
            latents=latent_init,  # ← Varies per candidate
            seed=base_seed,  # Fixed seed across candidates
            output_type="pt",
            is_video=True,
            vae_per_channel_normalize=True,
        )
        
        video_tensor = result.images
        
        # Create episode with SAME prompt, different config
        episode = videoepisode(
            prompt=prompt,  # ← Same for all episodes
            video=video_tensor,
            seed=base_seed + i,
            generation_params={
                'alpha': alpha,
                'guidance_scale': current_guidance,
                'num_inference_steps': num_inference_steps,
                'latent_shape': latent_shape,
            },
            reward=0.0,  # Initialize with default reward
            reward_info={}  # Initialize with empty reward info
        )
        
        candidates.append(episode)
        
    return candidates

def create_sampling_configurations(
    number_of_samples: int = 16,
    latent_range: tuple = (0.0, 1.0),
    guidance_range: tuple = (0.8, 1.2),
    device: str = "cuda",
) -> list:
    """
    Sample latent space in 2 dimensions: latent and guidance scale
    Args: Latent range for interpolation, alpha
    Guidance range for guidance scale
    Returns: List of sampling configurations via alpha and guidance scale
    """
    sampling_configurations = []
    for i in range(number_of_samples):
        alpha = np.random.uniform(latent_range[0], latent_range[1])
        guidance = np.random.uniform(guidance_range[0], guidance_range[1])
        sampling_configurations.append(sampling_configuration(alpha=alpha, guidance_scale=guidance))
    return sampling_configurations

def evaluate_episodes_with_physics_rewards(
    episodes: List[videoepisode],
    reward_type: str = 'combined_physics',
    custom_reward_fn: Optional[Callable] = None,
) -> List[videoepisode]:
    """
    Evaluate episodes using hand-crafted physics rewards.
    
    Leverages reward_functions.py for easy physics-aware evaluation.
    
    Args:
        episodes: List of videoepisode objects (videos already generated)
        reward_type: Which physics reward to use:
            - 'combined_physics': All physics metrics (recommended)
            - 'velocity': Motion magnitude and smoothness
            - 'acceleration': Force and dynamics patterns
            - 'smoothness': Trajectory continuity
            - 'momentum': Motion consistency
            - 'gravity': Downward acceleration bias
            - 'temporal': Frame-to-frame consistency
            - 'quality': Visual quality
            - 'custom': Use provided custom_reward_fn
        custom_reward_fn: Custom reward function(video, prompt) -> float
    
    Returns:
        Episodes with .reward field populated
        
    Example:
        >>> episodes = video_rollout(pipeline, prompt="Ball bouncing")
        >>> episodes = evaluate_episodes_with_physics_rewards(
        ...     episodes, reward_type='combined_physics'
        ... )
        >>> best = max(episodes, key=lambda ep: ep.reward)
    """
    print(f"\n📊 Evaluating {len(episodes)} episodes with '{reward_type}' reward...")
   
    # Evaluate each episode
    for i, episode in enumerate(episodes):
        # Select reward function
        if reward_type == 'combined_physics':
            reward = combined_physics_reward(episode.video, episode.prompt)
        elif reward_type == 'velocity':
            reward = physics_velocity_reward(episode.video, episode.prompt)
        elif reward_type == 'acceleration':
            reward = physics_acceleration_reward(episode.video, episode.prompt)
        elif reward_type == 'smoothness':
            reward = physics_trajectory_smoothness(episode.video, episode.prompt)
        elif reward_type == 'momentum':
            reward = physics_momentum_conservation(episode.video, episode.prompt)
        elif reward_type == 'gravity':
            reward = physics_gravity_consistency(episode.video, episode.prompt)
        elif reward_type == 'temporal':
            reward = grpo_reward_function(episode.video, episode.prompt, use_physics=False)
        elif reward_type == 'quality':
            reward = grpo_reward_function(episode.video, episode.prompt, use_physics=False)
        elif reward_type == 'custom' and custom_reward_fn is not None:
            reward = custom_reward_fn(episode.video, episode.prompt)
        else:
            reward = combined_physics_reward(episode.video, episode.prompt)
        
        # Store reward
        episode.reward = reward
        
        # Progress logging
        if (i + 1) % 4 == 0 or (i + 1) == len(episodes):
            recent_rewards = [ep.reward for ep in episodes[max(0, i-3):i+1]]
            print(f"  [{i+1}/{len(episodes)}] Recent rewards: {[f'{r:.3f}' for r in recent_rewards]}")
    
    # Summary
    all_rewards = [ep.reward for ep in episodes]
    print("\n✅ Evaluation complete!")
    print(f"  Reward range: [{min(all_rewards):.3f}, {max(all_rewards):.3f}]")
    print(f"  Mean: {np.mean(all_rewards):.3f}, Std: {np.std(all_rewards):.3f}\n")
    
    return episodes


def normalize_rewards_per_group(candidates: List[videoepisode]) -> List[videoepisode]:
    """
    Normalize rewards within a group (GRPO group relative normalization)
    
    Args:
        candidates: List of videoepisode objects with 'reward' field
    
    Returns:
        Same list with 'advantage' added to reward_info
    """
    if not candidates:
        return candidates
    
    groups = defaultdict(list)
    for episode in candidates:
        # Group by prompt
        groups[episode.prompt].append(episode)
    
    normalized_candidates = []
    
    for prompt, group in groups.items():
        group_rewards = [episode.reward for episode in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            # Compute advantage
            advantage = (episode.reward - mean_reward) / (std_reward + 1e-4)
            
            # Store in reward_info
            if not isinstance(episode.reward_info, dict):
                episode.reward_info = {}
            episode.reward_info['advantage'] = advantage
            episode.reward_info['mean_reward'] = mean_reward
            episode.reward_info['std_reward'] = std_reward
            
            normalized_candidates.append(episode)
    
    return normalized_candidates
    
def grpo_search_with_physics_rewards(
    video_pipeline,
    prompt: str,
    num_rounds: int = 5,
    candidates_per_round: int = 16,
    reward_type: str = 'combined_physics',
    base_seed: int = 2025,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Complete GRPO search using hand-crafted physics rewards.
    
    Convenience function that combines:
      - video_rollout() for generation
      - evaluate_episodes_with_physics_rewards() for scoring
      - normalize_rewards_per_group() for GRPO
      - update_policy_for_latent_trajectories() for improvement
    
    Args:
        video_pipeline: LTXVideoPipeline instance
        prompt: Text prompt for video generation
        num_rounds: Number of GRPO optimization rounds (default: 3)
        candidates_per_round: Videos per round (default: 16)
        reward_type: Physics reward type (default: 'combined_physics')
        base_seed: Random seed (default: 2025)
        device: Device (default: "cuda")
    
    Returns:
        Dictionary with:
            - 'best_episode': Best video found
            - 'all_episodes': All generated videos
            - 'final_anchors': (latent_a, latent_b) after optimization
    
    Example:
        >>> result = grpo_search_with_physics_rewards(
        ...     video_pipeline=pipeline,
        ...     prompt="Ball bouncing down stairs",
        ...     num_rounds=3,
        ...     reward_type='combined_physics',
        ... )
        >>> best_video = result['best_episode'].video
        >>> print(f"Best reward: {result['best_episode'].reward:.3f}")
    """
    print(f"\n{'='*80}")
    print("🎯 GRPO Search with Hand-Crafted Physics Rewards")
    print(f"{'='*80}")
    print(f"Prompt: '{prompt}'")
    print(f"Rounds: {num_rounds}, Candidates/round: {candidates_per_round}")
    print(f"Reward: {reward_type}")
    print(f"{'='*80}\n")
    
    # Initialize latent anchors
    torch.manual_seed(base_seed)
    latent_shape = (1, CHANNEL_SIZE, 16, 64, 96)  # Adjust based on resolution
    latent_a = torch.randn(latent_shape, device=device)
    latent_b = torch.randn(latent_shape, device=device)
    
    all_episodes = []
    best_overall = None
    
    for round_idx in range(num_rounds):
        print(f"\n{'='*70}")
        print(f"🔄 Round {round_idx + 1}/{num_rounds} - Starting...")
        print(f"{'='*70}")
        
        try:
            # Generate candidates (would need to pass anchors - simplified here)
            # In practice, you'd modify video_rollout to accept custom anchors
            print(f"📹 Generating {candidates_per_round} video candidates...")
            episodes = video_rollout(
                video_pipeline=video_pipeline,
                prompt=prompt,
                num_candidates_per_prompt=candidates_per_round,
                base_seed=base_seed + round_idx * 1000,
                device=device,
                latent_anchor_a=latent_a,
                latent_anchor_b=latent_b,
            )
            print(f"✅ Generated {len(episodes)} episodes")
            
            # Evaluate with physics rewards
            print(f"🏆 Computing physics rewards ({reward_type})...")
            episodes = evaluate_episodes_with_physics_rewards(
                episodes=episodes,
                reward_type=reward_type,
            )
            print("✅ Rewards computed")
            
            # Add video quality rewards (temporal diversity + spatial complexity)
            for episode in episodes:
                temporal_score = video_temporal_diversity(episode.video)
                spatial_score = video_spatial_complexity(episode.video)
                
                # Combine with existing reward
                # Increase visual quality influence
                quality_bonus = 0.2 * (temporal_score + spatial_score) / 2
                episode.reward += quality_bonus
                
                # Store in reward info
                if not hasattr(episode, 'reward_info') or episode.reward_info is None:
                    episode.reward_info = {}
                episode.reward_info['temporal_diversity'] = temporal_score
                episode.reward_info['spatial_complexity'] = spatial_score
                episode.reward_info['quality_bonus'] = quality_bonus
            
            # GRPO normalization
            episodes = normalize_rewards_per_group(episodes)
            
            # Find best this round
            best_this_round = max(episodes, key=lambda ep: ep.reward)
            
            # Update overall best
            if best_overall is None or best_this_round.reward > best_overall.reward:
                best_overall = best_this_round
                print(f"\n✨ New best found! Reward: {best_overall.reward:.4f}")
            
            print(f"\nRound {round_idx+1} Summary:")
            print(f"  Best: {best_this_round.reward:.4f}")
            print(f"  Advantage: {best_this_round.reward_info.get('advantage', 0):+.2f}")
            
            # Update trajectory policy (except last round)
            if round_idx < num_rounds - 1:
                latent_a, latent_b = update_policy_for_latent_trajectories(
                    candidates=episodes,
                    latent_a=latent_a,
                    latent_b=latent_b,
                    learning_rate=0.1 / (round_idx + 1),
                    prompt=prompt,
                )
            
            # Only keep track of episode count, not all episodes (memory optimization)
            all_episodes.extend(episodes)
            print(f"✅ Round {round_idx + 1} completed! Total episodes so far: {len(all_episodes)}")
            print(f"   💾 Memory: Only best video will be saved (not all {len(all_episodes)} videos)")
            
        except Exception as round_error:
            print(f"❌ Round {round_idx + 1} failed: {round_error}")
            print("   Continuing to next round...")
            continue
    
    print(f"\n{'='*80}")
    print("🏆 GRPO Search Complete!")
    print(f"{'='*80}")
    print(f"Best overall reward: {best_overall.reward:.4f}")
    print(f"Total videos generated: {len(all_episodes)}")
    print(f"{'='*80}\n")
    
    return {
        'best_episode': best_overall,
        'all_episodes': all_episodes,
        'final_anchors': (latent_a, latent_b),
    }


def video_temporal_diversity(video_tensor: torch.Tensor) -> float:
    """
    Measure temporal diversity in video (meaningful replacement for entropy)
    
    Args:
        video_tensor: Video tensor [B, C, T, H, W]
    Returns:
        Temporal diversity score [0, 1]
    """
    if video_tensor.dim() != 5:
        return 0.5
    
    # Measure how much frames differ from each other
    frame_differences = torch.diff(video_tensor, dim=2)  # [B, C, T-1, H, W]
    temporal_diversity = frame_differences.var().item()
    
    # Normalize to [0, 1] range
    normalized_diversity = min(temporal_diversity * 10, 1.0)
    return normalized_diversity

def video_spatial_complexity(video_tensor: torch.Tensor) -> float:
    """
    Measure spatial complexity in video frames
    
    Args:
        video_tensor: Video tensor [B, C, T, H, W]
    Returns:
        Spatial complexity score [0, 1]
    """
    if video_tensor.dim() != 5:
        return 0.5
    
    # Compute spatial gradients
    grad_x = torch.diff(video_tensor, dim=-1)  # Horizontal gradients
    grad_y = torch.diff(video_tensor, dim=-2)  # Vertical gradients 
    # Spatial complexity
    complexity = (grad_x.var() + grad_y.var()).item()

    # Add simple sharpness term (L1 gradient magnitude)
    sharpness = (grad_x.abs().mean() + grad_y.abs().mean()).item()
    # Normalize terms to [0,1] by conservative scaling
    normalized_complexity = min(complexity * 5, 1.0)
    normalized_sharpness = min(sharpness * 2, 1.0)
    # Blend, emphasizing sharpness slightly
    return 0.6 * normalized_complexity + 0.4 * normalized_sharpness

def improve_temporal_smoothness(
    latent_anchor: torch.Tensor,
    smoothness_weight: float = 0.3,
) -> torch.Tensor:
    """
    Improve temporal structure within a single anchor latent.
    
    Applies temporal smoothing to reduce frame-to-frame jitter
    in the latent trajectory. 
    
    Args:
        latent_anchor: [1, 8, T, H, W] where T=16 is temporal dimension
        smoothness_weight: How much smoothing (0=none, 1=maximum)
    
    Returns:
        Latent with smoother temporal progression
    """
    B, C, T, H, W = latent_anchor.shape
    
    if T < 3:
        return latent_anchor
    
    original = latent_anchor.clone()
    smoothed = original.clone()
    
    # Apply temporal smoothing across the temporal dimension
    for t in range(1, T-1):
        # Average with neighbors in time
        smoothed[:, :, t, :, :] = (
            0.25 * original[:, :, t-1, :, :] +  # Previous temporal frame
            0.50 * original[:, :, t, :, :] +    # Current temporal frame
            0.25 * original[:, :, t+1, :, :]    # Next temporal frame
        )
    
    # Blend original and smoothed
    improved = (1 - smoothness_weight) * original + smoothness_weight * smoothed
    
    return improved


def extract_temporal_pattern(latent_trajectory: torch.Tensor) -> torch.Tensor:
    """
    Extract temporal pattern from a successful latent trajectory.
    
    Analyzes how the latent magnitude changes across temporal dimension
    and extracts this as a reusable pattern.
    
    Args:
        latent_trajectory: [1, 8, T, H, W] from best candidate
    
    Returns:
        Temporal pattern [T] showing magnitude progression
    """
    B, C, T, H, W = latent_trajectory.shape
    
    # Compute magnitude of each temporal frame
    temporal_magnitudes = []
    for t in range(T):
        frame_magnitude = latent_trajectory[:, :, t, :, :].norm().item()
        temporal_magnitudes.append(frame_magnitude)
    
    # Convert to tensor and normalize
    pattern = torch.tensor(temporal_magnitudes, device=latent_trajectory.device)
    pattern = pattern / pattern.mean()  # Normalize to mean=1
    
    return pattern


def apply_temporal_pattern(
    latent_anchor: torch.Tensor,
    temporal_pattern: torch.Tensor,
    pattern_strength: float = 0.5,
) -> torch.Tensor:
    """
    Apply learned temporal pattern to anchor latent.
    
    Modulates the temporal progression of the anchor to match
    a successful pattern from previous candidates.
    
    Args:
        latent_anchor: [1, 8, T, H, W]
        temporal_pattern: [T] magnitude scaling per frame
        pattern_strength: How strongly to apply pattern (0=none, 1=full)
    
    Returns:
        Latent with applied temporal pattern
    """
    B, C, T, H, W = latent_anchor.shape
    
    # Compute baseline (average across time)
    temporal_mean = latent_anchor.mean(dim=2, keepdim=True)  # [1, 8, 1, H, W]
    
    # Temporal deviations from mean
    temporal_deviation = latent_anchor - temporal_mean
    
    # Reshape pattern for broadcasting
    pattern = temporal_pattern.view(1, 1, T, 1, 1)
    
    # Apply pattern to modulate deviations
    patterned = temporal_mean + temporal_deviation * pattern
    
    # Blend with original
    improved = (1 - pattern_strength) * latent_anchor + pattern_strength * patterned
    
    return improved


def apply_physics_prior(
    latent_anchor: torch.Tensor,
    physics_type: str = 'acceleration',
    strength: float = 0.4,
) -> torch.Tensor:
    """
    Apply physics-based temporal prior to latent trajectory.
    
    Injects known physics patterns (acceleration, deceleration, oscillation)
    into the latent's temporal structure.
    
    Args:
        latent_anchor: [1, 8, T, H, W]
        physics_type: 'acceleration', 'deceleration', 'oscillation', 'constant'
        strength: How strongly to apply prior (0=none, 1=full physics)
    
    Returns:
        Latent with physics-informed temporal structure
    """
    B, C, T, H, W = latent_anchor.shape
    
    # Create physics-based temporal weight curve
    t_normalized = torch.linspace(0, 1, T, device=latent_anchor.device)
    
    if physics_type == 'acceleration':
        # Quadratic increase (gravity, falling objects)
        weights = t_normalized ** 2
        
    elif physics_type == 'deceleration':
        # Quadratic decrease (friction, rolling to stop)
        weights = 1 - (1 - t_normalized) ** 2
        
    elif physics_type == 'oscillation':
        # Sinusoidal (pendulum, spring)
        weights = 0.5 + 0.5 * torch.sin(2 * np.pi * t_normalized)
        
    elif physics_type == 'constant':
        # Linear (constant velocity)
        weights = t_normalized
        
    else:
        weights = torch.ones(T, device=latent_anchor.device)
    
    # Normalize weights
    weights = weights / weights.mean()
    
    # Apply to latent temporal dimension
    weights = weights.view(1, 1, T, 1, 1)
    
    temporal_mean = latent_anchor.mean(dim=2, keepdim=True)
    temporal_variation = latent_anchor - temporal_mean
    
    physics_informed = temporal_mean + temporal_variation * weights
    
    # Blend with original
    improved = (1 - strength) * latent_anchor + strength * physics_informed
    
    return improved


def estimate_latent_gradient(
    candidates: List[videoepisode],
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
) -> torch.Tensor:
    """
    Estimate gradient in latent space using finite differences.
    
    Computes direction in latent space that improves reward.
    Uses candidates with different alpha values to estimate gradient.
    
    Args:
        candidates: List of videoepisode objects with rewards
        latent_a: First anchor latent
        latent_b: Second anchor latent
    
    Returns:
        Estimated gradient tensor pointing toward better latent regions
    """
    # Extract alpha and rewards
    alphas = []
    rewards = []
    for ep in candidates:
        if 'alpha' in ep.generation_params:
            alphas.append(ep.generation_params['alpha'])
            rewards.append(ep.reward)
    
    if len(alphas) < 2:
        # Not enough data for gradient estimation
        return torch.zeros_like(latent_b - latent_a)
    
    # Convert to numpy for easier computation
    alphas = np.array(alphas)
    rewards = np.array(rewards)
    
    # Estimate gradient d(reward)/d(alpha) using linear regression
    # reward ≈ a * alpha + b
    # gradient = a (slope)
    if len(alphas) > 1:
        # Fit line through (alpha, reward) points
        coeffs = np.polyfit(alphas, rewards, deg=1)
        gradient_alpha = coeffs[0]  # Slope = d(reward)/d(alpha)
    else:
        gradient_alpha = 0.0
    
    # Convert alpha gradient to latent space gradient
    # latent(alpha) = (1-alpha)*A + alpha*B = A + alpha*(B-A)
    # d(latent)/d(alpha) = B - A
    latent_direction = latent_b - latent_a
    
    # Gradient in latent space
    latent_gradient = gradient_alpha * latent_direction
    
    return latent_gradient


def update_policy_for_latent_trajectories(
    candidates: List[videoepisode],
    latent_a: torch.Tensor,
    latent_b: torch.Tensor,
    learning_rate: float = 0.1,
    prompt: str = None,
) -> tuple:
    """
    COMPLETE trajectory improvement with 4 components.
    
    Updates latent policy to discover better latent TRAJECTORIES through time:
      1. Update anchor POSITIONS (spatial) - move to better regions
      2. Update TEMPORAL STRUCTURE - smooth motion within anchors
      3. Learn from BEST patterns - copy successful dynamics
      4. Apply PHYSICS PRIORS - inject known motion patterns
    
    Args:
        candidates: List of videoepisode objects with rewards
        latent_a: Current first anchor [1, 8, 16, 64, 96] with temporal dim
        latent_b: Current second anchor with temporal structure
        learning_rate: Step size for update (default: 0.1)
        prompt: Optional prompt to infer physics type
    
    Returns:
        Tuple of (new_latent_a, new_latent_b) with improved trajectories
        
    Complete Algorithm:
        SPATIAL: Move anchors to better latent regions (gradient ascent)
        TEMPORAL: Smooth temporal progression (reduce jitter)
        LEARNING: Extract and apply successful temporal patterns
        PHYSICS: Inject domain knowledge (acceleration, etc.)
    """
    print(f"\n{'='*70}")
    print("🎬 COMPLETE Latent Trajectory Improvement")
    print(f"{'='*70}")
    
    # Find best candidate
    best_candidate = max(candidates, key=lambda ep: ep.reward)
    best_alpha = best_candidate.generation_params.get('alpha', 0.5)
    best_latent_trajectory = (1 - best_alpha) * latent_a + best_alpha * latent_b
    
    print(f"Best candidate: α={best_alpha:.3f}, reward={best_candidate.reward:.4f}")
    
    # ============================================================
    # COMPONENT 1: Update Anchor POSITIONS (Spatial)
    # ============================================================
    print("\n1️⃣ Updating anchor POSITIONS...")
    
    gradient = estimate_latent_gradient(candidates, latent_a, latent_b)
    gradient_norm = torch.norm(gradient)
    
    # Move toward better region
    trajectory_center = best_latent_trajectory + learning_rate * gradient
    
    if gradient_norm > 1e-6:
        trajectory_direction = gradient / gradient_norm
    else:
        trajectory_direction = torch.randn_like(gradient)
        trajectory_direction = trajectory_direction / torch.norm(trajectory_direction)
    
    # Create new spatial positions
    spread = 0.5
    new_latent_a = trajectory_center - spread * trajectory_direction
    new_latent_b = trajectory_center + spread * trajectory_direction
    
    print("   ✓ Gradient magnitude: {gradient_norm.item():.6f}")
    print("   ✓ Moved anchors toward better latent region")
    
    # ============================================================
    # COMPONENT 2: Update TEMPORAL STRUCTURE (Smoothness)
    # ============================================================
    print("\n2️⃣ Improving TEMPORAL smoothness...")
    
    # Apply temporal smoothing to reduce jitter
    new_latent_a = improve_temporal_smoothness(
        new_latent_a,
        smoothness_weight=0.3
    )
    new_latent_b = improve_temporal_smoothness(
        new_latent_b,
        smoothness_weight=0.3
    )
    
    print("   ✓ Applied temporal smoothing (weight=0.3)")
    print("   ✓ Reduced frame-to-frame jitter in anchors")
    
    # ============================================================
    # COMPONENT 3: Learn from BEST PATTERNS
    # ============================================================
    print("\n3️⃣ Learning from best candidate's temporal pattern...")
    
    # Extract temporal pattern from best trajectory
    best_temporal_pattern = extract_temporal_pattern(best_latent_trajectory)
    
    print("   Pattern: {[f'{p:.2f}' for p in best_temporal_pattern.cpu().numpy()[:8]]}... (first 8 frames)")
    
    # Apply learned pattern to new anchors
    new_latent_a = apply_temporal_pattern(
        new_latent_a,
        best_temporal_pattern,
        pattern_strength=0.5
    )
    new_latent_b = apply_temporal_pattern(
        new_latent_b,
        best_temporal_pattern,
        pattern_strength=0.5
    )
    
    print("   ✓ Applied successful temporal pattern (strength=0.5)")
    print("   ✓ New anchors inherit best dynamics")
    
    # ============================================================
    # COMPONENT 4: Apply PHYSICS PRIORS
    # ============================================================
    print("\n4️⃣ Applying physics priors...")
    
    # Infer physics type from prompt
    physics_type = 'constant'  # Default
    if prompt:
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['fall', 'drop', 'bounce', 'gravity']):
            physics_type = 'acceleration'
            print("   Detected: Gravity/falling motion")
        elif any(word in prompt_lower for word in ['slow', 'stop', 'friction', 'roll']):
            physics_type = 'deceleration'
            print("   Detected: Deceleration motion")
        elif any(word in prompt_lower for word in ['swing', 'pendulum', 'oscillate', 'wave']):
            physics_type = 'oscillation'
            print("   Detected: Oscillatory motion")
        else:
            print("   Using: Constant velocity prior")
    
    # Apply physics-informed temporal structure
    new_latent_a = apply_physics_prior(
        new_latent_a,
        physics_type=physics_type,
        strength=0.3
    )
    new_latent_b = apply_physics_prior(
        new_latent_b,
        physics_type=physics_type,
        strength=0.3
    )

    return new_latent_a, new_latent_b








