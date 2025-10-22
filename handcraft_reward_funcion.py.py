import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from torchvision.transforms.functional import resize

# Try to import CLIP

import clip
  
# Global model cache (lazy loading)
global dino_model, clip_model, clip_preprocess
dino_model = None
clip_model = None
clip_preprocess = None

# ============================================================================
# CLIP AND DINO UTILITIES
# ============================================================================

def dino_transform_image_gpu(batch_tensor, n_px, device):
    """
    Transform image for DINO model (GPU-based)
    
    Args:
        batch_tensor: Image tensor [C, H, W]
        n_px: Target size (224 for DINO)
        device: Device
    
    Returns:
        Normalized tensor for DINO
    """
    resized_tensor = resize(batch_tensor, (n_px, n_px), antialias=False)
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    std = torch.tensor([0.229, 0.224, 0.225], device=device)
    
    normalized_tensor = (resized_tensor - mean[:, None, None]) / std[:, None, None]
    
    return normalized_tensor


def load_dino_model(device='cuda'):
    """Lazy load DINOv2 model (latest version)"""
    global dino_model
    
    if dino_model is None:
        print("Loading DINOv2 model...")
        # Use DINOv2 (newer, better version)
        dino_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitb14',  # ViT-B/14 model (better than v1)
            source='github'
        ).to(device)
        dino_model.eval()
        print("✓ DINOv2 model loaded (ViT-B/14)")
    
    return dino_model


def load_clip_model(device='cuda'):
    """Lazy load CLIP model"""
    global clip_model, clip_preprocess
    
    if clip_model is None:
        print("Loading CLIP model...")
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()
        print("✓ CLIP model loaded")
    
    return clip_model, clip_preprocess


# ============================================================================
# DINO-BASED REWARDS (Object Tracking and Consistency)
# ============================================================================

@torch.no_grad()
def dino_object_presence_reward(video: torch.Tensor, prompt: str = None, device='cuda') -> float:
    """
    Object presence and saliency using DINO.
    
    Measures if there's a clear, consistent object in the video.
    Uses DINO attention to detect object saliency.
    
    Args:
        video: Video tensor
        prompt: Text prompt (optional)
        device: Device
    
    Returns:
        Object presence score (0-1)
    """
    # Load DINO
    dino = load_dino_model(device)
    
    if len(video.shape) == 5:
        video = video[0]
    
    C, T, H, W = video.shape
    
    # Sample middle frame (most likely to show main object)
    mid_frame = video[:, T // 2, :, :]
    
    # Denormalize if needed
    if mid_frame.min() < 0:
        mid_frame = (mid_frame + 1) / 2
    
    # Transform for DINO
    frame_transformed = dino_transform_image_gpu(mid_frame, 224, device)
    
    # Get DINO features (includes attention)
    frame_batch = frame_transformed.unsqueeze(0)
    features = dino(frame_batch)
    
    # Use feature magnitude as proxy for object presence
    # Strong features = clear object
    feature_magnitude = features.norm().item()
    
    # Normalize to [0, 1]
    presence_score = np.clip(feature_magnitude / 10.0, 0, 1)
    
    return float(presence_score)

# ============================================================================
# CLIP-BASED REWARDS (Text-Video Alignment)
# ============================================================================

@torch.no_grad()
def clip_text_alignment_reward(video: torch.Tensor, prompt: str, device='cuda') -> float:
    """
    Text-video alignment using CLIP.
    
    Measures how well the video matches the text prompt.
    High score = video content aligns with prompt description.
    
    Args:
        video: Video tensor [B, C, T, H, W] or [C, T, H, W]
        prompt: Text prompt (REQUIRED for CLIP)
        device: Device
    
    Returns:
        Alignment score (0-1), higher = better text-video match
    """
    if prompt is None:
        return 0.5  # No prompt, can't compute alignment
    
    # Load CLIP
    clip_model, clip_preprocess = load_clip_model(device)
    
    # Handle batch dimension
    if len(video.shape) == 5:
        video = video[0]  # [C, T, H, W]
    
    C, T, H, W = video.shape
    
    # Sample frames (use 4-8 frames)
    num_frames_to_sample = min(8, T)
    frame_indices = torch.linspace(0, T-1, num_frames_to_sample).long()
    
    # Encode text
    text_tokens = clip.tokenize([prompt]).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    
    # Encode sampled video frames
    frame_similarities = []
    for idx in frame_indices:
        frame = video[:, idx, :, :]  # [C, H, W]
        
        # Denormalize from [-1, 1] to [0, 1]
        if frame.min() < 0:
            frame = (frame + 1) / 2
        
        # Resize to CLIP input size (224x224)
        frame_resized = resize(frame, (224, 224), antialias=True)
        
        # Normalize for CLIP
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        frame_normalized = (frame_resized - mean[:, None, None]) / std[:, None, None]
        
        # Encode frame
        frame_batch = frame_normalized.unsqueeze(0)  # [1, C, H, W]
        image_features = clip_model.encode_image(frame_batch)
        image_features = F.normalize(image_features, dim=-1)
        
        # Compute similarity
        similarity = F.cosine_similarity(text_features, image_features, dim=-1).item()
        frame_similarities.append(max(0, similarity))  # Clip to [0, 1]
    
    # Average alignment across frames
    alignment_score = np.mean(frame_similarities)
    
    return float(alignment_score)


@torch.no_grad()
def clip_temporal_alignment_reward(video: torch.Tensor, prompt: str, device='cuda') -> float:
    """
    Temporal text alignment using CLIP.
    
    Measures if video progression matches prompt's temporal description.
    Checks if early/middle/late frames match corresponding parts of prompt.
    
    Args:
        video: Video tensor
        prompt: Text prompt with temporal elements
        device: Device
    
    Returns:
        Temporal alignment score (0-1)
    """
    if prompt is None:
        return 0.5
    
    # Load CLIP
    clip_model, _ = load_clip_model(device)
    
    if len(video.shape) == 5:
        video = video[0]
    
    C, T, H, W = video.shape
    
    # Create temporal prompts
    temporal_prompts = [
        f"beginning of {prompt}",
        f"middle of {prompt}",
        f"end of {prompt}",
    ]
    
    # Encode temporal prompts
    text_tokens = clip.tokenize(temporal_prompts).to(device)
    text_features = clip_model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    
    # Sample frames from different temporal regions
    temporal_regions = [
        T // 6,      # Early (frame ~20)
        T // 2,      # Middle (frame ~60)
        T * 5 // 6,  # Late (frame ~100)
    ]
    
    alignments = []
    for region_idx, frame_t in enumerate(temporal_regions):
        frame = video[:, frame_t, :, :]
        
        # Denormalize and resize
        if frame.min() < 0:
            frame = (frame + 1) / 2
        frame_resized = resize(frame, (224, 224), antialias=True)
        
        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device)
        frame_normalized = (frame_resized - mean[:, None, None]) / std[:, None, None]
        
        # Encode
        image_features = clip_model.encode_image(frame_normalized.unsqueeze(0))
        image_features = F.normalize(image_features, dim=-1)
        
        # Match with corresponding temporal prompt
        similarity = F.cosine_similarity(
            text_features[region_idx:region_idx+1], image_features, dim=-1
        ).item()
        
        alignments.append(max(0, similarity))
    
    # Average temporal alignment
    temporal_score = np.mean(alignments)
    
    return float(temporal_score)


# ============================================================================
# COMBINED CLIP+DINO REWARDS
# ============================================================================

@torch.no_grad()
def clip_dino_combined_reward(
    video: torch.Tensor,
    prompt: str,
    device='cuda',
    weights: dict = None,
) -> Dict[str, float]:
    """
    Comprehensive reward combining CLIP and DINO.
    
    Components:
      - CLIP text alignment: Does video match prompt?
      - CLIP temporal alignment: Does progression match prompt?
      - DINO subject consistency: Is object tracked consistently?
      - DINO object presence: Is there a clear subject?
    
    Args:
        video: Video tensor
        prompt: Text prompt
        device: Device
        weights: Custom weights for components (optional)
    
    Returns:
        Dictionary with individual scores and total reward
    """
    if weights is None:
        weights = {
            'clip_alignment': 0.35,
            'clip_temporal': 0.25,
            'dino_consistency': 0.25,
            'dino_presence': 0.15,
        }
    
    # Compute individual components
    clip_align = clip_text_alignment_reward(video, prompt, device)
    clip_temporal = clip_temporal_alignment_reward(video, prompt, device)
    dino_consistency = dino_subject_consistency_reward(video, prompt, device)
    dino_presence = dino_object_presence_reward(video, prompt, device)
    
    # Weighted combination
    total_reward = (
        weights['clip_alignment'] * clip_align +
        weights['clip_temporal'] * clip_temporal +
        weights['dino_consistency'] * dino_consistency +
        weights['dino_presence'] * dino_presence
    )
    
    return {
        'reward': float(total_reward),
        'clip_alignment': float(clip_align),
        'clip_temporal': float(clip_temporal),
        'dino_consistency': float(dino_consistency),
        'dino_presence': float(dino_presence),
    }


@torch.no_grad()
def dino_subject_consistency_reward(video: torch.Tensor, prompt: str = None, device='cuda') -> float:
    """
    Subject consistency using DINO (based on provided example).
    
    Tracks object identity across frames using DINO features.
    Ensures the same object is present throughout the video.
    
    Args:
        video: Video tensor [B, C, T, H, W] or [C, T, H, W]
        prompt: Text prompt (optional, not used for DINO)
        device: Device
    
    Returns:
        Consistency score (0-1), higher = better object tracking
    """
    # Load DINO
    dino = load_dino_model(device)
    
    # Handle batch dimension
    if len(video.shape) == 5:
        video = video[0]  # [C, T, H, W]
    
    C, T, H, W = video.shape
    
    # Sample frames
    num_frames_sample = min(8, T)
    frame_indices = torch.linspace(0, T-1, num_frames_sample).long()
    
    # Transform frames for DINO
    images_list = []
    for idx in frame_indices:
        frame = video[:, idx, :, :]  # [C, H, W]
        # Denormalize if needed
        if frame.min() < 0:
            frame = (frame + 1) / 2
        transformed = dino_transform_image_gpu(frame, 224, device)
        images_list.append(transformed)
    
    # Extract features and compute consistency
    video_sim = 0.0
    
    # Anchor on first frame
    with torch.no_grad():
        anchor_image = images_list[0].unsqueeze(0)
        anchor_features = dino(anchor_image)
        anchor_features = F.normalize(anchor_features, dim=-1, p=2)
    
    # Compare subsequent frames
    former_features = anchor_features
    
    for i in range(1, len(images_list)):
        with torch.no_grad():
            image = images_list[i].unsqueeze(0)
            image_features = dino(image)
            image_features = F.normalize(image_features, dim=-1, p=2)
            
            # Similarity to previous frame
            sim_prev = max(0.0, F.cosine_similarity(former_features, image_features, dim=-1).item())
            
            # Similarity to anchor (first frame)
            sim_anchor = max(0.0, F.cosine_similarity(anchor_features, image_features, dim=-1).item())
            
            # Combined similarity (weighted toward temporal consistency)
            cur_sim = 0.6 * sim_prev + 0.4 * sim_anchor
            video_sim += cur_sim
            
            former_features = image_features
    
    # Average similarity across transitions
    sim_per_frame = video_sim / (len(images_list) - 1) if len(images_list) > 1 else 0.5
    
    return float(sim_per_frame)

# ============================================================================
# SIMPLE HAND-CRAFTED REWARDS (No training needed)
# ============================================================================

@torch.no_grad()
def video_quality_reward(video: torch.Tensor, prompt: str = None) -> float:
    """
    Universal reward: Visual quality via variance
    No training needed
    """
    variance = video.var().item()
    # Normalize to reasonable range
    quality = np.clip(variance / 0.1, 0, 1)
    return float(quality)


@torch.no_grad()
def motion_diversity_reward(video: torch.Tensor, prompt: str = None) -> float:
    """
    Universal reward: Motion presence and diversity
    No training needed
    """
    if len(video.shape) == 5:
        video = video[0]
    
    T = video.shape[1]
    frame_diffs = []
    
    for t in range(T - 1):
        diff = torch.abs(video[:, t+1] - video[:, t]).mean().item()
        frame_diffs.append(diff)
    
    # High std = dynamic motion
    motion_score = np.std(frame_diffs)
    return float(np.clip(motion_score * 10, 0, 1))

# ============================================================================
# PHYSICS-AWARE REWARDS (Motion and Dynamics)
# ============================================================================

@torch.no_grad()
def physics_velocity_reward(video: torch.Tensor, prompt: str = None) -> float:
    """
    Reward based on velocity characteristics (first derivative of motion)
    
    Measures if object motion shows realistic velocity patterns
    """
    if len(video.shape) == 5:
        video = video[0]  # [C, T, H, W]
    
    T = video.shape[1]
    velocities = []
    
    # Compute frame-to-frame differences (velocity proxy)
    for t in range(T - 1):
        velocity = torch.abs(video[:, t+1] - video[:, t]).mean().item()
        velocities.append(velocity)
    
    velocities = np.array(velocities)
    
    # Physics properties
    # 1. Velocity magnitude (is there motion?)
    avg_velocity = np.mean(velocities)
    
    # 2. Velocity smoothness (no sudden jumps)
    velocity_smoothness = 1.0 / (1.0 + np.std(velocities))
    
    # Combined velocity score
    velocity_score = 0.6 * avg_velocity * 100 + 0.4 * velocity_smoothness
    
    return float(np.clip(velocity_score, 0, 1))


@torch.no_grad()
def physics_acceleration_reward(video: torch.Tensor, prompt: str = None) -> float:
    """
    Reward based on acceleration (second derivative of motion)
    
    Realistic physics should show smooth acceleration patterns
    """
    if len(video.shape) == 5:
        video = video[0]
    
    T = video.shape[1]
    
    # Compute velocities (first derivative)
    velocities = []
    for t in range(T - 1):
        vel = torch.abs(video[:, t+1] - video[:, t]).mean().item()
        velocities.append(vel)
    
    if len(velocities) < 2:
        return 0.5
    
    velocities = np.array(velocities)
    
    # Compute accelerations (second derivative)
    accelerations = np.diff(velocities)
    
    # Physics properties
    # 1. Smooth acceleration (realistic physics has smooth changes)
    accel_smoothness = 1.0 / (1.0 + np.std(accelerations))
    
    # 2. Bounded acceleration (no infinite forces)
    max_accel = np.max(np.abs(accelerations))
    accel_bounded = 1.0 / (1.0 + max_accel * 100)
    
    # 3. Natural acceleration patterns (should vary, not constant)
    accel_variance = np.std(accelerations)
    accel_naturalness = np.clip(accel_variance * 10, 0, 1)
    
    # Combined
    accel_score = (
        0.4 * accel_smoothness +
        0.3 * accel_bounded +
        0.3 * accel_naturalness
    )
    
    return float(accel_score)

@torch.no_grad()
def physics_trajectory_smoothness(video: torch.Tensor, prompt: str = None) -> float:
    """
    Reward smooth trajectories (mimics real-world physics)
    
    Real objects follow smooth paths, not jerky/teleporting motions
    """
    if len(video.shape) == 5:
        video = video[0]
    
    T = video.shape[1]
    
    # Compute motion vectors across frames
    motion_vectors = []
    for t in range(T - 1):
        motion = video[:, t+1] - video[:, t]  # [C, H, W]
        motion_vectors.append(motion)
    
    # Measure smoothness of motion direction changes
    direction_changes = []
    for t in range(len(motion_vectors) - 1):
        # How much does motion direction change?
        direction_change = torch.abs(motion_vectors[t+1] - motion_vectors[t]).mean().item()
        direction_changes.append(direction_change)
    
    # Smooth trajectories have small direction changes
    avg_direction_change = np.mean(direction_changes)
    smoothness = 1.0 / (1.0 + avg_direction_change * 100)
    
    return float(smoothness)


@torch.no_grad()
def physics_momentum_conservation(video: torch.Tensor, prompt: str = None) -> float:
    """
    Reward consistent motion (momentum-like behavior)
    
    Objects in motion tend to stay in motion (Newton's first law approximation)
    """
    if len(video.shape) == 5:
        video = video[0]
    
    T = video.shape[1]
    
    # Compute velocities
    velocities = []
    for t in range(T - 1):
        vel = torch.abs(video[:, t+1] - video[:, t]).mean().item()
        velocities.append(vel)
    
    velocities = np.array(velocities)
    
    # Check if velocity maintains direction (momentum)
    # Low variance in velocity magnitude = consistent momentum
    velocity_consistency = 1.0 / (1.0 + np.std(velocities))
    
    # Check if motion doesn't reverse abruptly
    velocity_changes = np.diff(velocities)
    no_abrupt_reversals = 1.0 / (1.0 + np.sum(np.abs(velocity_changes > 0.1)))
    
    momentum_score = 0.6 * velocity_consistency + 0.4 * no_abrupt_reversals
    
    return float(momentum_score)


@torch.no_grad()
def physics_gravity_consistency(video: torch.Tensor, prompt: str = None) -> float:
    """
    Reward downward acceleration patterns (gravity-like)
    
    For falling/bouncing objects, expect consistent downward bias
    """
    if len(video.shape) == 5:
        video = video[0]
    
    T = video.shape[1]
    C, _, H, W = video.shape
    
    # Track center of mass (simplified: average bright pixels)
    vertical_positions = []
    for t in range(T):
        frame = video[:, t, :, :]  # [C, H, W]
        # Weight by intensity to find "object" center
        intensity = frame.mean(dim=0)  # [H, W]
        
        # Compute vertical center of mass
        y_coords = torch.arange(H, device=video.device).float()
        y_center = (intensity.sum(dim=1) * y_coords).sum() / (intensity.sum() + 1e-6)
        vertical_positions.append(y_center.item())
    
    vertical_positions = np.array(vertical_positions)
    
    # Compute vertical velocities
    if len(vertical_positions) < 2:
        return 0.5
    
    vertical_velocities = np.diff(vertical_positions)
    
    # Compute vertical accelerations  
    if len(vertical_velocities) < 2:
        return 0.5
    
    vertical_accelerations = np.diff(vertical_velocities)
    
    # Check for consistent downward bias (positive = downward in pixel coords)
    mean_accel = np.mean(vertical_accelerations)
    gravity_consistency = 1.0 / (1.0 + np.abs(mean_accel - 0.5))  # Expect ~0.5 downward
    
    return float(np.clip(gravity_consistency, 0, 1))


@torch.no_grad()
def combined_physics_reward(video: torch.Tensor, prompt: str = None) -> float:
    """
    Comprehensive physics-aware reward combining multiple dynamics properties
    
    Evaluates:
      - Velocity patterns
      - Acceleration smoothness  
      - Trajectory smoothness
      - Momentum conservation
      - Gravity consistency (if applicable)
    """
    # Core physics metrics
    velocity = physics_velocity_reward(video, prompt)
    acceleration = physics_acceleration_reward(video, prompt)
    trajectory = physics_trajectory_smoothness(video, prompt)
    momentum = physics_momentum_conservation(video, prompt)
    
    # Optional: gravity (may not apply to all motions)
    if prompt and any(word in prompt.lower() for word in ['fall', 'drop', 'bounce', 'gravity']):
        gravity = physics_gravity_consistency(video, prompt)
        # Weight gravity higher for gravity-related tasks
        total = (
            0.2 * velocity +
            0.25 * acceleration +
            0.25 * trajectory +
            0.15 * momentum +
            0.15 * gravity
        )
    else:
        # General motion tasks
        total = (
            0.25 * velocity +
            0.3 * acceleration +
            0.3 * trajectory +
            0.15 * momentum
        )
    
    return float(total)


@torch.no_grad()
def combined_hand_crafted_reward(video: torch.Tensor, prompt: str = None, device: str = 'cuda') -> float:
    """
    Combined reward using CLIP, DINO, video quality, and motion dynamics.
    
    Integrates:
      - CLIP text alignment (if prompt available)
      - DINO object tracking
      - Visual quality metrics
      - Motion dynamics
    
    Args:
        video: Video tensor
        prompt: Text prompt (optional)
        device: Device
    
    Returns:
        Combined reward score (0-1)
    """
    # Quality and motion (always computed)
    quality = video_quality_reward(video, prompt)
    motion = motion_diversity_reward(video, prompt)
    
    # CLIP alignment (if prompt available)
    if prompt:
        try:
            clip_align = clip_text_alignment_reward(video, prompt, device)
        except Exception:
            clip_align = 0.5
    else:
        clip_align = 0.5
    
    # DINO tracking
    try:
        dino_consistency = dino_subject_consistency_reward(video, prompt, device)
    except Exception:
        dino_consistency = 0.5
    
    # Weighted combination
    total = (
        0.3 * clip_align +
        0.3 * dino_consistency +
        0.2 * quality +
        0.2 * motion
    )
    
    return float(total)


# ============================================================================
# COMPREHENSIVE REWARD: CLIP + DINO + PHYSICS
# ============================================================================

@torch.no_grad()
def comprehensive_grpo_reward(
    video: torch.Tensor,
    prompt: str,
    device: str = 'cuda',
    use_clip: bool = True,
    use_dino: bool = True,
    use_physics: bool = True,
) -> Dict[str, float]:
    """
    Ultimate comprehensive reward combining all modalities:
      - CLIP: Text-video alignment (semantic correctness)
      - DINO: Object tracking consistency (subject identity)
      - Physics: Motion dynamics (realistic motion)
    
    Perfect for GRPO as it captures multiple quality dimensions.
    
    Args:
        video: Video tensor [B, C, T, H, W] or [C, T, H, W]
        prompt: Text prompt for the video
        device: Device (default: 'cuda')
        use_clip: Include CLIP text alignment (default: True)
        use_dino: Include DINO object tracking (default: True)
        use_physics: Include physics dynamics (default: True)
    
    Returns:
        Dictionary with:
            'reward': Total combined score (0-1)
            'clip_alignment': CLIP text-video score
            'clip_temporal': CLIP temporal alignment
            'dino_consistency': DINO tracking score
            'dino_presence': DINO object presence
            'physics_velocity': Motion magnitude
            'physics_acceleration': Dynamics quality
            'physics_smoothness': Trajectory continuity
            ... (all component scores)
    
    Example:
        >>> result = comprehensive_grpo_reward(video, "Ball bouncing")
        >>> print(f"Total: {result['reward']:.3f}")
        >>> print(f"  CLIP alignment: {result['clip_alignment']:.3f}")
        >>> print(f"  DINO tracking: {result['dino_consistency']:.3f}")
        >>> print(f"  Physics: {result['physics_smoothness']:.3f}")
    """
    scores = {}
    
    # === CLIP Rewards (Text Alignment) ===
    if use_clip and prompt:
        try:
            scores['clip_alignment'] = clip_text_alignment_reward(video, prompt, device)
            scores['clip_temporal'] = clip_temporal_alignment_reward(video, prompt, device)
        except Exception as e:
            print(f"⚠️ CLIP evaluation failed: {e}")
            scores['clip_alignment'] = 0.5
            scores['clip_temporal'] = 0.5
    else:
        scores['clip_alignment'] = 0.5
        scores['clip_temporal'] = 0.5
    
    # === DINO Rewards (Object Tracking) ===
    if use_dino:
        try:
            scores['dino_consistency'] = dino_subject_consistency_reward(video, prompt, device)
            scores['dino_presence'] = dino_object_presence_reward(video, prompt, device)
        except Exception as e:
            print(f"⚠️ DINO evaluation failed: {e}")
            scores['dino_consistency'] = 0.5
            scores['dino_presence'] = 0.5
    else:
        scores['dino_consistency'] = 0.5
        scores['dino_presence'] = 0.5
    
    # === Physics Rewards (Motion Dynamics) ===
    if use_physics:
        try:
            scores['physics_velocity'] = physics_velocity_reward(video, prompt)
            scores['physics_acceleration'] = physics_acceleration_reward(video, prompt)
            scores['physics_smoothness'] = physics_trajectory_smoothness(video, prompt)
            scores['physics_momentum'] = physics_momentum_conservation(video, prompt)
        except Exception as e:
            print(f"⚠️ Physics evaluation failed: {e}")
            scores['physics_velocity'] = 0.5
            scores['physics_acceleration'] = 0.5
            scores['physics_smoothness'] = 0.5
            scores['physics_momentum'] = 0.5
    else:
        scores['physics_velocity'] = 0.5
        scores['physics_acceleration'] = 0.5
        scores['physics_smoothness'] = 0.5
        scores['physics_momentum'] = 0.5
    
    # === Weighted Combination ===
    # Weights emphasize different aspects based on what's enabled
    if use_clip and use_dino and use_physics:
        # All modalities: Balanced weighting
        total_reward = (
            0.8 * scores['clip_alignment'] +
            0.15 * scores['clip_temporal'] +
            0.15 * scores['dino_consistency'] +
            0.10 * scores['dino_presence'] +
            0.15 * scores['physics_velocity'] +
            0.10 * scores['physics_acceleration'] +
            0.10 * scores['physics_smoothness']
        )
    elif use_clip and use_dino:
        # CLIP + DINO only
        total_reward = (
            0.8 * scores['clip_alignment'] +
            0.3 * scores['clip_temporal'] +
            0.2 * scores['dino_consistency'] +
            0.1 * scores['dino_presence']
        )
    elif use_physics:
        # Physics only
        total_reward = combined_physics_reward(video, prompt)
    else:
        # Fallback
        total_reward = combined_hand_crafted_reward(video, prompt)
    
    scores['reward'] = float(total_reward)
    
    return scores


@torch.no_grad()
def grpo_reward_with_clip_dino(
    video: torch.Tensor,
    prompt: str,
    device: str = 'cuda',
) -> float:
    """
    Simple wrapper for GRPO that returns single reward score.
    
    Uses comprehensive CLIP + DINO + Physics evaluation.
    
    Args:
        video: Video tensor
        prompt: Text prompt
        device: Device
    
    Returns:
        Single combined reward score (0-1)
        
    Example:
        >>> from video_grpo import video_rollout
        >>> from handcraft_reward_funcion import grpo_reward_with_clip_dino
        >>> 
        >>> episodes = video_rollout(pipeline, prompt="Ball bouncing")
        >>> for ep in episodes:
        ...     ep.reward = grpo_reward_with_clip_dino(ep.video, ep.prompt)
    """
    result = comprehensive_grpo_reward(
        video=video,
        prompt=prompt,
        device=device,
        use_clip=True,
        use_dino=True,
        use_physics=True,
    )
    
    return result['reward']