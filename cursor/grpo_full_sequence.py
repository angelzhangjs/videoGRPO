#!/usr/bin/env python3
"""
GRPO with Full Frame Sequence Calculation
Computes rewards across all frames during and after generation
"""

import torch
import numpy as np
from typing import Dict, List, Callable, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import imageio


@dataclass
class FrameReward:
    """Reward information for a single frame"""
    frame_idx: int
    frame_tensor: torch.Tensor
    reward: float
    reward_components: Dict[str, float]
    

@dataclass
class SequenceReward:
    """Reward information for entire video sequence"""
    frame_rewards: List[FrameReward]
    sequence_reward: float
    temporal_consistency: float
    progression_score: float
    metadata: Dict[str, Any]
    

class FullSequenceRewardCalculator:
    """
    Calculates rewards across the full frame sequence
    Supports both per-frame and sequence-level rewards
    """
    
    def __init__(
        self,
        per_frame_reward_fn: Optional[Callable] = None,
        sequence_reward_fn: Optional[Callable] = None,
        temporal_consistency_fn: Optional[Callable] = None,
    ):
        """
        Args:
            per_frame_reward_fn: Function to compute reward for individual frames
                Signature: (frame: torch.Tensor, frame_idx: int, prompt: str) -> Dict[str, float]
            sequence_reward_fn: Function to compute sequence-level reward
                Signature: (frames: torch.Tensor, prompt: str) -> float
            temporal_consistency_fn: Function to compute temporal consistency
                Signature: (frames: torch.Tensor) -> float
        """
        self.per_frame_reward_fn = per_frame_reward_fn
        self.sequence_reward_fn = sequence_reward_fn
        self.temporal_consistency_fn = temporal_consistency_fn or self._default_temporal_consistency
        
    def _default_temporal_consistency(self, frames: torch.Tensor) -> float:
        """
        Default temporal consistency using frame-to-frame similarity
        
        Args:
            frames: [T, C, H, W] or [C, T, H, W]
        """
        if frames.shape[0] == 3:  # [C, T, H, W]
            frames = frames.permute(1, 0, 2, 3)  # -> [T, C, H, W]
        
        T = frames.shape[0]
        if T < 2:
            return 1.0
        
        # Compute frame-to-frame differences
        diffs = []
        for t in range(T - 1):
            diff = torch.abs(frames[t+1] - frames[t]).mean()
            diffs.append(diff.item())
        
        # Lower difference = higher consistency
        avg_diff = np.mean(diffs)
        consistency = 1.0 / (1.0 + avg_diff)
        
        return consistency
    
    def compute_full_sequence_reward(
        self,
        video_frames: torch.Tensor,
        prompt: str,
        return_per_frame: bool = True,
    ) -> SequenceReward:
        """
        Compute rewards for the full frame sequence
        
        Args:
            video_frames: Video tensor [B, C, T, H, W] or [C, T, H, W]
            prompt: Text prompt for generation
            return_per_frame: Whether to return per-frame rewards
            
        Returns:
            SequenceReward object with full reward breakdown
        """
        # Normalize input shape
        if len(video_frames.shape) == 5:
            video_frames = video_frames[0]  # Take first batch: [C, T, H, W]
        
        C, T, H, W = video_frames.shape
        
        # 1. Compute per-frame rewards
        frame_rewards = []
        if self.per_frame_reward_fn and return_per_frame:
            for frame_idx in range(T):
                frame = video_frames[:, frame_idx, :, :]  # [C, H, W]
                
                # Compute reward for this frame
                reward_dict = self.per_frame_reward_fn(frame, frame_idx, prompt)
                
                frame_reward = FrameReward(
                    frame_idx=frame_idx,
                    frame_tensor=frame,
                    reward=reward_dict.get('reward', 0.0),
                    reward_components=reward_dict,
                )
                frame_rewards.append(frame_reward)
        
        # 2. Compute sequence-level reward
        if self.sequence_reward_fn:
            sequence_reward = self.sequence_reward_fn(video_frames, prompt)
        else:
            # Default: average of per-frame rewards
            if frame_rewards:
                sequence_reward = np.mean([fr.reward for fr in frame_rewards])
            else:
                sequence_reward = 0.0
        
        # 3. Compute temporal consistency
        temporal_consistency = self.temporal_consistency_fn(video_frames)
        
        # 4. Compute progression score (how well does the video progress?)
        progression_score = self._compute_progression_score(frame_rewards)
        
        # 5. Create final sequence reward
        result = SequenceReward(
            frame_rewards=frame_rewards,
            sequence_reward=sequence_reward,
            temporal_consistency=temporal_consistency,
            progression_score=progression_score,
            metadata={
                'num_frames': T,
                'resolution': (H, W),
                'prompt': prompt,
            }
        )
        
        return result
    
    def _compute_progression_score(self, frame_rewards: List[FrameReward]) -> float:
        """
        Compute how well the video progresses (e.g., for task completion)
        Higher score if rewards increase over time
        """
        if len(frame_rewards) < 2:
            return 0.5
        
        rewards = [fr.reward for fr in frame_rewards]
        
        # Check if rewards are generally increasing (good for task completion)
        improvements = 0
        for i in range(1, len(rewards)):
            if rewards[i] > rewards[i-1]:
                improvements += 1
        
        progression = improvements / (len(rewards) - 1)
        return progression


class VideoGRPOWithFullSequence:
    """
    GRPO implementation with full frame sequence reward calculation
    """
    
    def __init__(
        self,
        video_generator,
        reward_calculator: FullSequenceRewardCalculator,
    ):
        self.video_generator = video_generator
        self.reward_calculator = reward_calculator
        
    def generate_with_sequence_rewards(
        self,
        prompt: str,
        num_candidates: int = 4,
        generation_config: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple candidates and compute full sequence rewards
        
        Args:
            prompt: Text prompt
            num_candidates: Number of video candidates to generate
            generation_config: Generation parameters
            
        Returns:
            List of dictionaries containing videos and their sequence rewards
        """
        if generation_config is None:
            generation_config = {}
        
        candidates = []
        
        for i in range(num_candidates):
            print(f"\n=== Generating candidate {i+1}/{num_candidates} ===")
            
            # Vary seed for diversity
            seed = generation_config.get('seed', 2025) + i
            config = generation_config.copy()
            config['seed'] = seed
            
            # Generate video
            video_tensor = self.video_generator.generate(
                prompt=prompt,
                **config
            )
            
            # Compute full sequence rewards
            sequence_reward = self.reward_calculator.compute_full_sequence_reward(
                video_frames=video_tensor,
                prompt=prompt,
                return_per_frame=True,
            )
            
            # Print detailed breakdown
            self._print_reward_breakdown(sequence_reward, i)
            
            candidates.append({
                'video': video_tensor,
                'seed': seed,
                'sequence_reward': sequence_reward,
                'total_reward': sequence_reward.sequence_reward,
                'config': config,
            })
        
        # Sort by total reward
        candidates.sort(key=lambda x: x['total_reward'], reverse=True)
        
        return candidates
    
    def _print_reward_breakdown(self, sequence_reward: SequenceReward, candidate_idx: int):
        """Print detailed reward breakdown"""
        print(f"\nCandidate {candidate_idx} Reward Breakdown:")
        print(f"  Sequence Reward: {sequence_reward.sequence_reward:.4f}")
        print(f"  Temporal Consistency: {sequence_reward.temporal_consistency:.4f}")
        print(f"  Progression Score: {sequence_reward.progression_score:.4f}")
        
        if sequence_reward.frame_rewards:
            print(f"\n  Per-Frame Rewards ({len(sequence_reward.frame_rewards)} frames):")
            # Sample frames for display
            sample_indices = np.linspace(
                0, 
                len(sequence_reward.frame_rewards) - 1, 
                min(5, len(sequence_reward.frame_rewards))
            ).astype(int)
            
            for idx in sample_indices:
                fr = sequence_reward.frame_rewards[idx]
                print(f"    Frame {fr.frame_idx}: {fr.reward:.4f}")
    
    def iterative_refinement(
        self,
        prompt: str,
        num_iterations: int = 3,
        candidates_per_iteration: int = 4,
        generation_config: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Iterative refinement using full sequence rewards
        
        Similar to GRPO's policy update, but at inference time:
        1. Generate candidates
        2. Select best based on sequence rewards
        3. Refine parameters based on best candidate
        4. Repeat
        """
        if generation_config is None:
            generation_config = {
                'guidance_scale': 7.5,
                'num_inference_steps': 40,
            }
        
        best_overall = None
        
        for iteration in range(num_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            # Generate candidates
            candidates = self.generate_with_sequence_rewards(
                prompt=prompt,
                num_candidates=candidates_per_iteration,
                generation_config=generation_config,
            )
            
            # Best from this iteration
            best_candidate = candidates[0]
            
            # Update overall best
            if best_overall is None or best_candidate['total_reward'] > best_overall['total_reward']:
                best_overall = best_candidate
                print(f"\n✅ New best! Reward: {best_overall['total_reward']:.4f}")
            
            # Refine generation config based on best candidate
            generation_config = self._refine_config(
                generation_config,
                best_candidate,
                iteration,
            )
        
        print(f"\n{'='*60}")
        print(f"Final Best Reward: {best_overall['total_reward']:.4f}")
        print(f"{'='*60}")
        
        return best_overall
    
    def _refine_config(
        self,
        config: Dict[str, Any],
        best_candidate: Dict[str, Any],
        iteration: int,
    ) -> Dict[str, Any]:
        """
        Refine generation config based on best candidate's performance
        
        This is analogous to policy update in GRPO
        """
        new_config = config.copy()
        
        # Analyze sequence reward to decide refinement
        seq_reward = best_candidate['sequence_reward']
        
        # If temporal consistency is low, increase guidance
        if seq_reward.temporal_consistency < 0.7:
            new_config['guidance_scale'] = min(
                new_config.get('guidance_scale', 7.5) + 1.0,
                15.0
            )
            print(f"  🔧 Increasing guidance_scale to {new_config['guidance_scale']:.1f}")
        
        # If progression is poor, increase inference steps
        if seq_reward.progression_score < 0.5:
            new_config['num_inference_steps'] = min(
                new_config.get('num_inference_steps', 40) + 10,
                100
            )
            print(f"  🔧 Increasing steps to {new_config['num_inference_steps']}")
        
        # Exploration: slightly randomize seed
        new_config['seed'] = config.get('seed', 2025) + (iteration + 1) * 100
        
        return new_config


# ============================================================
# Example Usage & Integration
# ============================================================

def example_red_dot_per_frame_reward(frame: torch.Tensor, frame_idx: int, prompt: str) -> Dict[str, float]:
    """
    Example per-frame reward for red dot task
    """
    # Convert frame to numpy
    frame_np = frame.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
    frame_np = (frame_np * 255).astype(np.uint8)
    
    import cv2
    
    # Detect red dots
    hsv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Count red pixels
    red_pixel_ratio = red_mask.sum() / (red_mask.shape[0] * red_mask.shape[1])
    
    # Reward based on task
    if "erase" in prompt.lower():
        # Want fewer red dots over time
        expected_ratio = max(0, 1.0 - (frame_idx / 100))  # Assuming 100 frames
        reward = 1.0 - abs(red_pixel_ratio - expected_ratio)
    else:
        reward = 0.5
    
    return {
        'reward': reward,
        'red_pixel_ratio': red_pixel_ratio,
    }


def example_sequence_reward(frames: torch.Tensor, prompt: str) -> float:
    """
    Example sequence-level reward
    """
    # For "erase" task, want monotonic decrease in red pixels
    if "erase" in prompt.lower():
        # Check if red pixels decrease over time
        # (Simplified - would use actual red detection)
        return 0.8
    return 0.5


if __name__ == "__main__":
    print("Full Sequence GRPO Framework Loaded")
    print("\nKey Features:")
    print("  ✅ Per-frame reward calculation")
    print("  ✅ Sequence-level reward aggregation")
    print("  ✅ Temporal consistency measurement")
    print("  ✅ Progression tracking")
    print("  ✅ Iterative refinement with parameter updates")

