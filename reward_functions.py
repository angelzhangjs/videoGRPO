"""
Trainable Reward Model for Video GRPO with Physics Focus

This module provides:

1. TRAINABLE REWARD MODEL (VideoRewardModel):
   - Neural network that learns from VLM (Gemini) preferences
   - Enhanced architecture for motion and dynamics:
     * Motion encoder: Captures velocity from frame differences
     * Temporal attention: Focuses on key motion moments
     * Multi-head outputs: Velocity, acceleration, smoothness scores
   - Trained ONCE with backprop, then used FROZEN (no backprop in GRPO)

2. PHYSICS-AWARE HAND-CRAFTED REWARDS:
   - physics_velocity_reward: First derivative (motion magnitude)
   - physics_acceleration_reward: Second derivative (force/dynamics)
   - physics_trajectory_smoothness: Path continuity
   - physics_momentum_conservation: Motion consistency
   - physics_gravity_consistency: Downward acceleration patterns
   - combined_physics_reward: All physics metrics weighted

3. USAGE MODES:
   Mode A: Train reward model on VLM data (uses backprop)
   Mode B: Use frozen trained model in GRPO (no backprop, fast)
   Mode C: Use hand-crafted physics rewards (no training needed)

The physics-focused design makes the reward model particularly good at
evaluating videos of physical phenomena: balls rolling, objects falling,
pendulum swinging, fluid motion, etc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path


class VideoRewardModel(nn.Module):
    """
    Trainable neural network that learns to predict video quality scores
    with focus on motion and physics dynamics.
    
    Architecture:
        Video → Spatial Features → Motion Features → Temporal Dynamics → Reward Score
        
    Enhanced for physics:
        - Motion encoder: Captures velocity and acceleration
        - Temporal attention: Understands motion patterns
        - Physics-aware features: Learns dynamics properties
        
    Can be trained on VLM preferences (e.g., Gemini scores)
    Then used frozen for fast GRPO evaluation
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_frames_sample: int = 8,
        focus_on_physics: bool = True,
    ):
        """
        Args:
            feature_dim: Dimension of video features
            hidden_dim: Hidden layer dimension
            num_frames_sample: How many frames to sample from video
            focus_on_physics: If True, adds motion/physics-specific components
        """
        super().__init__()
        
        self.num_frames_sample = num_frames_sample
        self.focus_on_physics = focus_on_physics
        
        # Spatial feature extractor: CNN for individual frames
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Output: 256 x 4 x 4
        )
        
        if focus_on_physics:
            # Motion encoder: Captures velocity (frame differences)
            self.motion_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),  # Output: 64 x 4 x 4
            )
            
            motion_feature_dim = 64 * 4 * 4
        else:
            self.motion_encoder = None
            motion_feature_dim = 0
        
        # Temporal encoder with attention
        lstm_input_dim = 256 * 4 * 4 + motion_feature_dim
        self.temporal_encoder = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=feature_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        
        if focus_on_physics:
            # Temporal attention: Focus on key motion moments
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=8,
                batch_first=True,
            )
        
        # Quality prediction head (enhanced for physics)
        if focus_on_physics:
            # Multi-head output for different aspects
            self.quality_predictor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            
            # Separate heads for different physics properties
            self.velocity_head = nn.Linear(hidden_dim // 2, 1)
            self.acceleration_head = nn.Linear(hidden_dim // 2, 1)
            self.smoothness_head = nn.Linear(hidden_dim // 2, 1)
            self.overall_head = nn.Linear(hidden_dim // 2, 1)
            
        else:
            # Simple single-head predictor
            self.quality_predictor = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, 1),
            )
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Predict reward score for video with physics awareness
        
        Args:
            video: Video tensor [B, C, T, H, W] or [C, T, H, W]
            
        Returns:
            Predicted reward score (scalar or dict if focus_on_physics)
        """
        # Handle batch dimension
        if len(video.shape) == 4:
            video = video.unsqueeze(0)  # Add batch dimension
        
        B, C, T, H, W = video.shape
        
        # Sample frames uniformly
        frame_indices = torch.linspace(0, T-1, self.num_frames_sample).long()
        sampled_frames = video[:, :, frame_indices, :, :]  # [B, C, num_frames, H, W]
        
        # Encode spatial features per frame
        frames_flat = sampled_frames.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        frame_features = self.frame_encoder(frames_flat)  # [B*num_frames, 256, 4, 4]
        frame_features = frame_features.reshape(B, self.num_frames_sample, -1)
        
        # Physics-aware: Also encode motion (frame differences)
        if self.focus_on_physics and self.motion_encoder is not None:
            motion_features_list = []
            for i in range(self.num_frames_sample - 1):
                # Compute frame difference (motion)
                frame_diff = sampled_frames[:, :, i+1] - sampled_frames[:, :, i]
                motion_feat = self.motion_encoder(frame_diff.reshape(-1, C, H, W))
                motion_feat = motion_feat.reshape(B, -1)
                motion_features_list.append(motion_feat)
            
            # Pad last frame (no motion after it)
            motion_features_list.append(torch.zeros_like(motion_features_list[0]))
            motion_features = torch.stack(motion_features_list, dim=1)  # [B, num_frames, motion_dim]
            
            # Concatenate spatial and motion features
            combined_features = torch.cat([frame_features, motion_features], dim=2)
        else:
            combined_features = frame_features
        
        # Temporal encoding with LSTM
        temporal_features, _ = self.temporal_encoder(combined_features)  # [B, num_frames, feature_dim]
        
        # Physics-aware: Apply temporal attention
        if self.focus_on_physics:
            attended_features, _ = self.temporal_attention(
                temporal_features, temporal_features, temporal_features
            )
            # Pool attended features
            video_features = attended_features.mean(dim=1)  # [B, feature_dim]
        else:
            # Take last hidden state
            video_features = temporal_features[:, -1, :]  # [B, feature_dim]
        
        # Predict quality scores
        if self.focus_on_physics:
            # Multi-head prediction for physics properties
            shared_features = self.quality_predictor(video_features)  # [B, hidden_dim//2]
            
            velocity_score = torch.sigmoid(self.velocity_head(shared_features))
            accel_score = torch.sigmoid(self.acceleration_head(shared_features))
            smooth_score = torch.sigmoid(self.smoothness_head(shared_features))
            overall_score = torch.sigmoid(self.overall_head(shared_features))
            
            # Weighted combination emphasizing physics
            quality_score = (
                0.25 * velocity_score +
                0.25 * accel_score +
                0.25 * smooth_score +
                0.25 * overall_score
            )
            
            return quality_score.squeeze()  # [B] or scalar
        else:
            # Simple single score
            quality_score = self.quality_predictor(video_features)
            return quality_score.squeeze()

class VLMRewardTrainer:
    """
    Trainer that uses VLM (Gemini) as teacher to train reward model
    """
    
    def __init__(
        self,
        reward_model: VideoRewardModel,
        vlm_evaluator,  # Your GeminiVLMVerifier
        device: str = "cuda",
        learning_rate: float = 1e-4,
    ):
        """
        Args:
            reward_model: The VideoRewardModel to train
            vlm_evaluator: VLM (e.g., GeminiVLMVerifier) that provides ground truth
            device: Device to train on
            learning_rate: Learning rate for training
        """
        self.reward_model = reward_model.to(device)
        self.vlm_evaluator = vlm_evaluator
        self.device = device
        
        # Optimizer for training
        self.optimizer = torch.optim.Adam(
            self.reward_model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        
        self.training_history = []
        
    def train_on_video_pairs(
        self,
        video_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        num_epochs: int = 10,
        batch_size: int = 4,
    ):
        """
        Train reward model on video comparison pairs.
        
        VLM provides ground truth: which video is better
        Reward model learns to match VLM's judgments
        
        Args:
            video_pairs: List of (video_a, video_b) tuples
            num_epochs: Number of training epochs
            batch_size: Batch size (limited by memory)
        """
        print(f"\n{'='*70}")
        print(f"Training Reward Model on VLM Preferences")
        print(f"{'='*70}")
        print(f"Training pairs: {len(video_pairs)}")
        print(f"Epochs: {num_epochs}")
        print(f"Using VLM: {type(self.vlm_evaluator).__name__}\n")
        
        self.reward_model.train()  # Training mode
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(video_pairs), batch_size):
                batch_pairs = video_pairs[i:i+batch_size]
                
                batch_loss = 0.0
                for video_a, video_b in batch_pairs:
                    # Get VLM ground truth scores
                    vlm_score_a = self._get_vlm_score(video_a)
                    vlm_score_b = self._get_vlm_score(video_b)
                    
                    # Predict scores with reward model
                    pred_score_a = self.reward_model(video_a.to(self.device))
                    pred_score_b = self.reward_model(video_b.to(self.device))
                    
                    # Loss 1: Preference ranking (which is better?)
                    if vlm_score_a > vlm_score_b:
                        # Reward model should predict A > B
                        ranking_loss = -torch.log(
                            torch.sigmoid(pred_score_a - pred_score_b) + 1e-8
                        )
                    else:
                        # Reward model should predict B > A
                        ranking_loss = -torch.log(
                            torch.sigmoid(pred_score_b - pred_score_a) + 1e-8
                        )
                    
                    # Loss 2: Absolute score matching (MSE)
                    mse_loss_a = F.mse_loss(
                        pred_score_a,
                        torch.tensor(vlm_score_a, device=self.device)
                    )
                    mse_loss_b = F.mse_loss(
                        pred_score_b,
                        torch.tensor(vlm_score_b, device=self.device)
                    )
                    
                    # Combined loss
                    loss = ranking_loss + 0.5 * (mse_loss_a + mse_loss_b)
                    batch_loss += loss
                
                # Backprop (THIS is where backprop happens - training the reward model!)
                batch_loss = batch_loss / len(batch_pairs)
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.reward_model.parameters(), 1.0)
                
                # Update reward model weights
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                total_loss += batch_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
            
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
            })

    def _get_vlm_score(self, video: torch.Tensor) -> float:
        """
        Get VLM evaluation score (ground truth)
        """
        # Use VLM to score video
        result = self.vlm_evaluator.verify_video_reasoning(
            video_frames=video,
            prompt="Evaluate video quality",
            reasoning_focus=['quality', 'coherence'],
        )
        return result.overall_score / 10.0  # Normalize to [0, 1]
    
    def save_model(self, path: str):
        """Save trained reward model"""
        torch.save({
            'model_state_dict': self.reward_model.state_dict(),
            'training_history': self.training_history,
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained reward model"""
        checkpoint = torch.load(path)
        self.reward_model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        print(f"✅ Model loaded from {path}")

# ============================================================================
# FROZEN REWARD MODEL FOR GRPO (No backprop when evaluating)
# ============================================================================

@torch.no_grad()  # ← NO backprop during evaluation!
def evaluate_with_learned_reward(
    video: torch.Tensor,
    reward_model: VideoRewardModel,
    device: str = "cuda",
) -> float:
    """
    Evaluate video using FROZEN trained reward model.
    
    This is used during GRPO - no training, just inference.
    
    Args:
        video: Video tensor [1, C, T, H, W] or [C, T, H, W]
        reward_model: Trained VideoRewardModel (frozen)
        device: Device
    
    Returns:
        Predicted reward score (0-1 range typically)
    """
    reward_model.eval()  # Evaluation mode (no dropout, etc.)
    
    # Move video to device
    video = video.to(device)
    
    # Predict reward (no gradients tracked!)
    with torch.no_grad():
        reward_score = reward_model(video)
    
    # Return as float
    if torch.is_tensor(reward_score):
        reward_score = reward_score.item()
    
    return float(reward_score)


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

# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def collect_vlm_training_data(
    video_pipeline,
    vlm_evaluator,
    prompts: List[str],
    videos_per_prompt: int = 10,
    save_path: str = "vlm_training_data.pt",
) -> List[Dict[str, Any]]:
    """
    Generate videos and collect VLM scores for training reward model.
    
    Args:
        video_pipeline: LTX-Video pipeline for generation
        vlm_evaluator: VLM (e.g., GeminiVLMVerifier) for scoring
        prompts: List of prompts to generate videos for
        videos_per_prompt: How many videos per prompt
        save_path: Where to save collected data
    
    Returns:
        List of training examples: [{'video': tensor, 'vlm_score': float}, ...]
    """
    print(f"\n{'='*70}")
    print("Collecting VLM Training Data")
    print(f"{'='*70}")
    print(f"Prompts: {len(prompts)}")
    print(f"Videos per prompt: {videos_per_prompt}")
    print(f"Total videos: {len(prompts) * videos_per_prompt}\n")
    
    training_data = []
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt {prompt_idx+1}/{len(prompts)}: '{prompt}'")
        
        for video_idx in range(videos_per_prompt):
            # Generate video with varied configs
            seed = 2025 + prompt_idx * 100 + video_idx
            guidance = np.random.uniform(5.0, 10.0)
            
            result = video_pipeline(
                prompt=prompt,
                guidance_scale=guidance,
                num_inference_steps=40,
                seed=seed,
                output_type="pt",
            )
            
            video = result.images
            
            # Get VLM score (ground truth)
            print(f"  Video {video_idx+1}/{videos_per_prompt}: Getting VLM score...")
            vlm_result = vlm_evaluator.verify_video_reasoning(
                video_frames=video,
                prompt=prompt,
                reasoning_focus=['quality', 'coherence', 'task_completion'],
            )
            
            vlm_score = vlm_result.overall_score / 10.0  # Normalize to [0, 1]
            
            training_data.append({
                'video': video.cpu(),  # Store on CPU to save GPU memory
                'vlm_score': vlm_score,
                'prompt': prompt,
                'config': {
                    'guidance': guidance,
                    'seed': seed,
                }
            })
            
            print(f"    VLM score: {vlm_score:.3f}")
    
    # Save collected data
    torch.save(training_data, save_path)
    print(f"\n✅ Collected {len(training_data)} training examples")
    print(f"✅ Saved to {save_path}\n")
    
    return training_data


def train_reward_model_from_vlm(
    training_data_path: str,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    save_path: str = "trained_reward_model.pt",
) -> VideoRewardModel:
    """
    Train reward model from VLM-labeled data.
    
    Args:
        training_data_path: Path to collected VLM training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_path: Where to save trained model
    
    Returns:
        Trained VideoRewardModel
    """
    print(f"\n{'='*70}")
    print("Training Reward Model from VLM Data")
    print(f"{'='*70}\n")
    
    # Load training data
    training_data = torch.load(training_data_path)
    print(f"Loaded {len(training_data)} training examples")
    
    # Create reward model
    reward_model = VideoRewardModel().to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        reward_model.train()
        
        total_loss = 0.0
        for i, example in enumerate(training_data):
            video = example['video'].to(device)
            target_score = example['vlm_score']
            
            # Predict score
            pred_score = reward_model(video)
            
            # Loss: Match VLM score
            loss = F.mse_loss(
                pred_score,
                torch.tensor(target_score, device=device)
            )
            
            # Backprop (training reward model, not video model!)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
    
    # Save trained model
    torch.save(reward_model.state_dict(), save_path)
    print(f"\n✅ Training complete! Model saved to {save_path}\n")
    
    return reward_model


# ============================================================================
# USAGE FUNCTIONS FOR GRPO
# ============================================================================

@torch.no_grad()
def grpo_reward_function(
    video: torch.Tensor,
    prompt: str,
    reward_model: Optional[VideoRewardModel] = None,
    use_learned: bool = False,
    use_physics: bool = False,
    device: str = "cuda",
) -> float:
    """
    Main reward function for GRPO.
    
    Can use:
      - Learned reward model (trained on VLM preferences)
      - Physics-aware hand-crafted rewards (motion/dynamics)
      - Standard hand-crafted rewards (quality/consistency)
    
    Args:
        video: Video tensor
        prompt: Text prompt (may be ignored for universal rewards)
        reward_model: Trained VideoRewardModel (if use_learned=True)
        use_learned: Whether to use learned model
        use_physics: Whether to use physics-aware rewards
        device: Device
    
    Returns:
        Reward score
    """
    if use_learned and reward_model is not None:
        # Use trained neural network (frozen, no backprop!)
        reward = evaluate_with_learned_reward(video, reward_model, device)
    elif use_physics:
        # Use physics-aware hand-crafted rewards
        reward = combined_physics_reward(video, prompt)
    return reward


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_training_workflow():
    """
    Example: How to train and use the reward model
    """
    from ltx_video.inference import create_ltx_video_pipeline
    from gemini_vlm_verifier import GeminiVLMVerifier
    
    # === PHASE 1: COLLECT TRAINING DATA ===
    print("Phase 1: Collecting training data with VLM...")
    
    video_pipeline = create_ltx_video_pipeline(...)
    vlm = GeminiVLMVerifier(api_key="YOUR_KEY")
    
    prompts = [
        "How to grasp a cup",
        "How to pour water",
        "How to place object",
        # ... more prompts
    ]
    
    training_data = collect_vlm_training_data(
        video_pipeline=video_pipeline,
        vlm_evaluator=vlm,
        prompts=prompts,
        videos_per_prompt=10,
        save_path="vlm_training_data.pt"
    )
    
    # === PHASE 2: TRAIN REWARD MODEL ===
    print("\nPhase 2: Training reward model...")
    
    reward_model = train_reward_model_from_vlm(
        training_data_path="vlm_training_data.pt",
        num_epochs=20,
        save_path="trained_reward_model.pt"
    )
    
    # === PHASE 3: USE IN GRPO (Frozen, no backprop!) ===
    print("\nPhase 3: Using trained reward model in GRPO...")
    
    from video_grpo import video_rollout
    
    # Load trained model (frozen)
    reward_model = VideoRewardModel()
    reward_model.load_state_dict(torch.load("trained_reward_model.pt"))
    reward_model.eval()  # Freeze for evaluation
    
    # Generate candidates
    candidates = video_rollout(
        video_pipeline=video_pipeline,
        prompt="How to grasp a cup",
        num_candidates_per_prompt=16,
    )
    
    # Evaluate with trained reward model (NO backprop!)
    for episode in candidates:
        episode.reward = grpo_reward_function(
            video=episode.video,
            prompt=episode.prompt,
            reward_model=reward_model,
            use_learned=True,  # Use trained model
        )
    
    # GRPO selection
    best = max(candidates, key=lambda ep: ep.reward)
    print(f"Best reward: {best.reward:.3f}")

