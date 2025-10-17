#!/usr/bin/env python3
"""
Advanced Path Reward Calculation for Full Frame Sequences
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any
import cv2

class AdvancedPathRewardCalculator:
    """
    Comprehensive reward calculation for video paths
    """
    
    def __init__(self, reward_weights: Dict[str, float] = None):
        self.weights = reward_weights or {
            'sequence_quality': 0.25,      # Overall video quality
            'frame_quality': 0.20,        # Average frame quality  
            'temporal_consistency': 0.15,  # Smoothness across time
            'motion_quality': 0.10,       # Natural motion patterns
            'semantic_coherence': 0.15,   # Prompt adherence over time
            'visual_diversity': 0.10,     # Avoid repetitive content
            'narrative_flow': 0.05        # Story progression
        }
    
    def compute_full_path_reward(
        self, 
        video_frames: torch.Tensor,  # [C, T, H, W]
        prompt: str,
        base_reward_function: callable
    ) -> Dict[str, Any]:
        """
        Compute comprehensive reward for entire video path
        """
        T = video_frames.shape[1]  # Number of frames
        
        # 1. Per-frame rewards
        frame_rewards = self._compute_frame_rewards(video_frames, prompt, base_reward_function)
        
        # 2. Sequence-level reward
        sequence_reward = self._compute_sequence_reward(video_frames, prompt, base_reward_function)
        
        # 3. Temporal consistency
        temporal_consistency = self._compute_temporal_consistency(video_frames)
        
        # 4. Motion quality
        motion_quality = self._compute_motion_quality(video_frames)
        
        # 5. Semantic coherence over time
        semantic_coherence = self._compute_semantic_coherence(video_frames, prompt)
        
        # 6. Visual diversity
        visual_diversity = self._compute_visual_diversity(video_frames)
        
        # 7. Narrative flow
        narrative_flow = self._compute_narrative_flow(video_frames, prompt)
        
        # Combine all rewards
        component_rewards = {
            'sequence_quality': sequence_reward,
            'frame_quality': np.mean(frame_rewards),
            'temporal_consistency': temporal_consistency,
            'motion_quality': motion_quality,
            'semantic_coherence': semantic_coherence,
            'visual_diversity': visual_diversity,
            'narrative_flow': narrative_flow
        }
        
        # Weighted combination
        total_reward = sum(
            self.weights[component] * reward 
            for component, reward in component_rewards.items()
        )
        
        return {
            'total_reward': total_reward,
            'component_rewards': component_rewards,
            'frame_rewards': frame_rewards,
            'reward_weights': self.weights,
            'path_length': T
        }
    
    def _compute_frame_rewards(
        self, 
        video_frames: torch.Tensor, 
        prompt: str, 
        reward_function: callable
    ) -> List[float]:
        """
        Compute reward for each individual frame
        """
        frame_rewards = []
        T = video_frames.shape[1]
        
        for t in range(T):
            frame = video_frames[:, t:t+1]  # [C, 1, H, W]
            reward_result = reward_function(frame.unsqueeze(0), prompt)
            frame_rewards.append(reward_result['reward'])
        
        return frame_rewards
    
    def _compute_sequence_reward(
        self, 
        video_frames: torch.Tensor, 
        prompt: str, 
        reward_function: callable
    ) -> float:
        """
        Compute reward for entire sequence as one unit
        """
        # Treat entire video as single input
        full_video = video_frames.unsqueeze(0)  # Add batch dim [1, C, T, H, W]
        sequence_result = reward_function(full_video, prompt)
        return sequence_result['reward']
    
    def _compute_temporal_consistency(self, video_frames: torch.Tensor) -> float:
        """
        Measure smoothness and consistency across time
        """
        # Frame-to-frame differences
        frame_diffs = torch.diff(video_frames, dim=1)  # [C, T-1, H, W]
        
        # Multiple consistency metrics
        
        # 1. Pixel-level consistency
        pixel_variance = frame_diffs.var().item()
        pixel_consistency = 1.0 / (1.0 + pixel_variance * 100)
        
        # 2. Feature-level consistency (using gradients as proxy)
        grad_x = torch.diff(video_frames, dim=-1)  # Horizontal gradients
        grad_y = torch.diff(video_frames, dim=-2)  # Vertical gradients
        
        grad_x_consistency = 1.0 / (1.0 + torch.diff(grad_x, dim=1).var().item() * 50)
        grad_y_consistency = 1.0 / (1.0 + torch.diff(grad_y, dim=1).var().item() * 50)
        
        # 3. Color consistency
        color_means = video_frames.mean(dim=[-2, -1])  # [C, T] - average color per frame
        color_variance = torch.diff(color_means, dim=1).var().item()
        color_consistency = 1.0 / (1.0 + color_variance * 10)
        
        # Combine consistency measures
        temporal_consistency = (
            0.4 * pixel_consistency +
            0.3 * grad_x_consistency +
            0.2 * grad_y_consistency +
            0.1 * color_consistency
        )
        
        return temporal_consistency
    
    def _compute_motion_quality(self, video_frames: torch.Tensor) -> float:
        """
        Evaluate motion patterns across the video path
        """
        frame_diffs = torch.diff(video_frames, dim=1)  # [C, T-1, H, W]
        
        # 1. Motion magnitude
        motion_magnitude = frame_diffs.abs().mean().item()
        
        # 2. Motion smoothness (second derivative)
        motion_acceleration = torch.diff(frame_diffs, dim=1)  # [C, T-2, H, W]
        motion_smoothness = 1.0 / (1.0 + motion_acceleration.var().item() * 1000)
        
        # 3. Motion direction consistency
        # Approximate optical flow using frame differences
        flow_x = frame_diffs.mean(dim=0)  # Average across channels
        flow_consistency = self._compute_flow_consistency(flow_x)
        
        # 4. Optimal motion range (not too static, not too chaotic)
        optimal_motion_range = (0.05, 0.15)  # Adjust based on your data
        motion_range_score = self._score_in_range(
            motion_magnitude, 
            optimal_motion_range[0], 
            optimal_motion_range[1]
        )
        
        motion_quality = (
            0.3 * motion_range_score +
            0.3 * motion_smoothness +
            0.2 * flow_consistency +
            0.2 * min(motion_magnitude * 5, 1.0)  # Reward some motion
        )
        
        return motion_quality
    
    def _compute_semantic_coherence(self, video_frames: torch.Tensor, prompt: str) -> float:
        """
        Measure how well the video maintains semantic meaning over time
        """
        T = video_frames.shape[1]
        
        # Sample frames at different time points
        sample_indices = torch.linspace(0, T-1, min(8, T)).long()
        sampled_frames = video_frames[:, sample_indices]  # [C, sampled_T, H, W]
        
        # Compute semantic consistency (placeholder - would use CLIP or similar)
        # For now, use visual similarity as proxy
        similarities = []
        for i in range(len(sample_indices) - 1):
            frame1 = sampled_frames[:, i]
            frame2 = sampled_frames[:, i+1]
            
            # Cosine similarity between frames
            similarity = F.cosine_similarity(
                frame1.flatten(), 
                frame2.flatten(), 
                dim=0
            ).item()
            similarities.append(similarity)
        
        # High similarity = good semantic coherence
        semantic_coherence = np.mean(similarities)
        return max(0.0, semantic_coherence)  # Ensure non-negative
    
    def _compute_visual_diversity(self, video_frames: torch.Tensor) -> float:
        """
        Reward visual diversity to avoid repetitive content
        """
        T = video_frames.shape[1]
        
        # Compute pairwise frame differences
        diversity_scores = []
        
        # Sample frame pairs to avoid O(T²) complexity
        num_samples = min(20, T * (T-1) // 2)
        indices = torch.randperm(T)[:min(10, T)]
        
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                frame1 = video_frames[:, indices[i]]
                frame2 = video_frames[:, indices[j]]
                
                # L2 distance as diversity measure
                diversity = (frame1 - frame2).pow(2).mean().item()
                diversity_scores.append(diversity)
        
        # Higher diversity = better (but not too high)
        avg_diversity = np.mean(diversity_scores)
        optimal_diversity_range = (0.1, 0.5)  # Adjust based on your data
        
        diversity_score = self._score_in_range(
            avg_diversity,
            optimal_diversity_range[0],
            optimal_diversity_range[1]
        )
        
        return diversity_score
    
    def _compute_narrative_flow(self, video_frames: torch.Tensor, prompt: str) -> float:
        """
        Evaluate narrative progression and story flow
        """
        T = video_frames.shape[1]
        
        # Divide video into segments (beginning, middle, end)
        segment_size = T // 3
        
        if segment_size < 1:
            return 0.5  # Neutral score for very short videos
        
        beginning = video_frames[:, :segment_size]
        middle = video_frames[:, segment_size:2*segment_size] 
        end = video_frames[:, -segment_size:]
        
        # Compute "narrative progression" as increasing complexity/change
        beginning_complexity = beginning.var().item()
        middle_complexity = middle.var().item()
        end_complexity = end.var().item()
        
        # Good narrative: some progression in visual complexity
        progression_score = 0.0
        if middle_complexity > beginning_complexity:
            progression_score += 0.5
        if end_complexity != beginning_complexity:  # Some change from start to end
            progression_score += 0.5
        
        return progression_score
    
    def _compute_flow_consistency(self, flow_field: torch.Tensor) -> float:
        """
        Compute consistency of optical flow field
        """
        # Simple flow consistency using spatial gradients
        flow_grad_x = torch.diff(flow_field, dim=-1)
        flow_grad_y = torch.diff(flow_field, dim=-2)
        
        # Consistent flow has low gradient variance
        consistency = 1.0 / (1.0 + flow_grad_x.var().item() + flow_grad_y.var().item() + 1e-6)
        return min(consistency, 1.0)
    
    def _score_in_range(self, value: float, min_val: float, max_val: float) -> float:
        """
        Score a value based on how well it fits in an optimal range
        """
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val:
            return max(0.0, 1.0 - (min_val - value) / min_val)
        else:  # value > max_val
            return max(0.0, 1.0 - (value - max_val) / max_val)

# Example usage
def example_advanced_reward():
    """
    Example of advanced path reward calculation
    """
    # Mock video frames [C, T, H, W]
    video_frames = torch.randn(3, 16, 224, 224)
    prompt = "A glowing pumpkin on a spooky porch"
    
    # Mock base reward function
    def mock_reward_function(video, prompt):
        return {'reward': torch.randn(1).item()}
    
    # Calculate advanced rewards
    calculator = AdvancedPathRewardCalculator()
    reward_result = calculator.compute_full_path_reward(
        video_frames, prompt, mock_reward_function
    )
    
    print("=== Advanced Path Reward Breakdown ===")
    print(f"Total Reward: {reward_result['total_reward']:.3f}")
    print("\nComponent Rewards:")
    for component, reward in reward_result['component_rewards'].items():
        weight = reward_result['reward_weights'][component]
        contribution = weight * reward
        print(f"  {component:20s}: {reward:.3f} (weight: {weight:.2f}, contrib: {contribution:.3f})")
    
    print(f"\nPath Length: {reward_result['path_length']} frames")
    print(f"Frame Rewards: min={min(reward_result['frame_rewards']):.3f}, "
          f"max={max(reward_result['frame_rewards']):.3f}, "
          f"avg={np.mean(reward_result['frame_rewards']):.3f}")

if __name__ == "__main__":
    example_advanced_reward()
