#!/usr/bin/env python3
"""
Complete Example: GRPO with Full Frame Sequence Calculation
Demonstrates how to modify GRPO for full frame-level rewards
"""

import sys
sys.path.append('ltx_video_source')

import torch
import numpy as np
import cv2
from typing import Dict, List, Any
from pathlib import Path
import argparse

from grpo_full_sequence import (
    FullSequenceRewardCalculator,
    VideoGRPOWithFullSequence,
    FrameReward,
    SequenceReward,
)

from ltx_grpo_integration import (
    LTXVideoGRPOGenerator,
    GRPOSearchWithLTXVideo,
)


class RedDotFullSequenceReward:
    """
    Full sequence reward calculator for red dot task
    Evaluates EVERY frame in the video sequence
    """
    
    def __init__(self, original_image_path: str = None):
        """
        Args:
            original_image_path: Path to reference image with red dots
        """
        self.original_image_path = original_image_path
        self.reference_red_ratio = None
        
        if original_image_path:
            img = cv2.imread(original_image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.reference_red_ratio = self._compute_red_ratio(img_rgb)
            print(f"Reference red ratio: {self.reference_red_ratio:.4f}")
    
    def _compute_red_ratio(self, frame_rgb: np.ndarray) -> float:
        """
        Compute ratio of red pixels in frame
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        
        # Red color ranges
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create mask
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Compute ratio
        total_pixels = red_mask.shape[0] * red_mask.shape[1]
        red_pixels = red_mask.sum() / 255
        ratio = red_pixels / total_pixels
        
        return ratio
    
    def _compute_red_dots_count(self, frame_rgb: np.ndarray) -> int:
        """
        Count number of distinct red dots in frame
        """
        hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size
        valid_dots = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5 < area < 5000:
                valid_dots += 1
        
        return valid_dots
    
    def per_frame_reward(self, frame: torch.Tensor, frame_idx: int, prompt: str) -> Dict[str, float]:
        """
        Compute reward for a single frame
        THIS IS CALLED FOR EVERY FRAME IN THE SEQUENCE
        
        Args:
            frame: Single frame tensor [C, H, W]
            frame_idx: Frame index in sequence
            prompt: Text prompt
            
        Returns:
            Dictionary with reward and components
        """
        # Convert to numpy
        frame_np = frame.permute(1, 2, 0).cpu().numpy()  # [C, H, W] -> [H, W, C]
        frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
        
        # Compute red metrics
        red_ratio = self._compute_red_ratio(frame_np)
        red_dot_count = self._compute_red_dots_count(frame_np)
        
        # Task-specific reward
        if "erase" in prompt.lower() or "remove" in prompt.lower():
            # Want red dots to decrease over time
            expected_ratio = max(0.0, 1.0 - (frame_idx / 120))  # Linear decrease
            ratio_error = abs(red_ratio - expected_ratio * (self.reference_red_ratio or 0.1))
            reward = 1.0 - ratio_error
            
        elif "appear" in prompt.lower() or "show" in prompt.lower():
            # Want red dots to increase over time
            expected_ratio = min(1.0, frame_idx / 120)
            ratio_error = abs(red_ratio - expected_ratio * 0.1)
            reward = 1.0 - ratio_error
            
        elif "fade" in prompt.lower():
            # Want smooth fading
            expected_ratio = max(0.0, 1.0 - (frame_idx / 120))
            reward = 1.0 - abs(red_ratio - expected_ratio * 0.1)
            
        else:
            # Default: penalize excessive red
            reward = 1.0 - red_ratio
        
        return {
            'reward': np.clip(reward, 0.0, 1.0),
            'red_ratio': red_ratio,
            'red_dot_count': red_dot_count,
            'frame_idx': frame_idx,
        }
    
    def sequence_reward(self, frames: torch.Tensor, prompt: str) -> float:
        """
        Compute reward for entire sequence
        
        Args:
            frames: Full video sequence [C, T, H, W]
            prompt: Text prompt
            
        Returns:
            Sequence-level reward
        """
        C, T, H, W = frames.shape
        
        red_ratios = []
        
        # Compute red ratio for each frame
        for t in range(T):
            frame = frames[:, t, :, :]  # [C, H, W]
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
            red_ratio = self._compute_red_ratio(frame_np)
            red_ratios.append(red_ratio)
        
        red_ratios = np.array(red_ratios)
        
        # Task-specific sequence reward
        if "erase" in prompt.lower() or "remove" in prompt.lower():
            # Want monotonic decrease
            differences = np.diff(red_ratios)
            monotonic_score = (differences <= 0).mean()  # Fraction of decreasing steps
            final_reduction = 1.0 - (red_ratios[-1] / (red_ratios[0] + 1e-6))
            reward = 0.5 * monotonic_score + 0.5 * np.clip(final_reduction, 0, 1)
            
        elif "appear" in prompt.lower():
            # Want monotonic increase
            differences = np.diff(red_ratios)
            monotonic_score = (differences >= 0).mean()
            reward = monotonic_score
            
        elif "fade" in prompt.lower():
            # Want smooth decrease
            smoothness = 1.0 - np.std(np.diff(red_ratios))
            final_low = 1.0 - red_ratios[-1]
            reward = 0.5 * smoothness + 0.5 * final_low
            
        else:
            # Default: average red ratio
            reward = 1.0 - red_ratios.mean()
        
        return np.clip(reward, 0.0, 1.0)
    
    def temporal_consistency(self, frames: torch.Tensor) -> float:
        """
        Measure temporal consistency of the video
        
        Args:
            frames: [C, T, H, W]
            
        Returns:
            Consistency score (0-1)
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
        
        # Lower variance in differences = more consistent
        consistency = 1.0 - (np.std(diffs) / (np.mean(diffs) + 1e-6))
        consistency = np.clip(consistency, 0, 1)
        
        return consistency


def run_grpo_full_sequence_search(
    prompt: str,
    output_dir: str = "outputs/grpo_full_sequence",
    original_image_path: str = None,
    num_rounds: int = 3,
    candidates_per_round: int = 4,
):
    """
    Run GRPO search with full frame sequence reward calculation
    
    Args:
        prompt: Text prompt
        output_dir: Where to save results
        original_image_path: Reference image (optional)
        num_rounds: Number of GRPO rounds
        candidates_per_round: Candidates per round
    """
    print("="*80)
    print("GRPO with Full Frame Sequence Calculation")
    print("="*80)
    print(f"Prompt: {prompt}")
    print(f"Output: {output_dir}")
    print(f"Rounds: {num_rounds}, Candidates per round: {candidates_per_round}")
    print("="*80 + "\n")
    
    # 1. Initialize reward calculator
    print("Initializing reward calculator...")
    red_dot_reward = RedDotFullSequenceReward(original_image_path)
    
    # 2. Create full sequence reward calculator
    reward_calculator = FullSequenceRewardCalculator(
        per_frame_reward_fn=red_dot_reward.per_frame_reward,
        sequence_reward_fn=red_dot_reward.sequence_reward,
        temporal_consistency_fn=red_dot_reward.temporal_consistency,
    )
    
    # 3. Initialize LTX-Video generator
    print("Loading LTX-Video pipeline...")
    ltx_generator = LTXVideoGRPOGenerator(
        pipeline_config_path="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
        device="cuda" if torch.cuda.is_available() else "cpu",
        precision="bfloat16",
    )
    
    # 4. Wrapper for compatibility
    def reward_fn_wrapper(video_tensor: torch.Tensor, prompt: str) -> Dict[str, Any]:
        """Wrapper to compute full sequence rewards"""
        seq_reward = reward_calculator.compute_full_sequence_reward(
            video_frames=video_tensor,
            prompt=prompt,
            return_per_frame=True,
        )
        
        # Print detailed breakdown
        print(f"\n  📊 Full Sequence Reward Breakdown:")
        print(f"     Sequence Reward: {seq_reward.sequence_reward:.4f}")
        print(f"     Temporal Consistency: {seq_reward.temporal_consistency:.4f}")
        print(f"     Progression: {seq_reward.progression_score:.4f}")
        print(f"     Frames Analyzed: {len(seq_reward.frame_rewards)}")
        
        # Sample per-frame rewards
        if seq_reward.frame_rewards:
            sample_frames = [0, len(seq_reward.frame_rewards)//4, 
                           len(seq_reward.frame_rewards)//2,
                           3*len(seq_reward.frame_rewards)//4,
                           len(seq_reward.frame_rewards)-1]
            print(f"     Sample Frame Rewards:")
            for idx in sample_frames:
                if idx < len(seq_reward.frame_rewards):
                    fr = seq_reward.frame_rewards[idx]
                    print(f"       Frame {fr.frame_idx:3d}: {fr.reward:.4f} "
                          f"(red_ratio={fr.reward_components.get('red_ratio', 0):.4f})")
        
        # Combine rewards
        total_reward = (
            0.5 * seq_reward.sequence_reward +
            0.3 * seq_reward.temporal_consistency +
            0.2 * seq_reward.progression_score
        )
        
        return {
            'reward': total_reward,
            'total_reward': total_reward,
            'sequence_reward': seq_reward.sequence_reward,
            'temporal_consistency': seq_reward.temporal_consistency,
            'progression_score': seq_reward.progression_score,
            'frame_rewards': seq_reward.frame_rewards,
            'reward_info': {
                'num_frames': len(seq_reward.frame_rewards),
            }
        }
    
    # 5. Run GRPO search
    print("\nStarting GRPO search with full sequence rewards...\n")
    searcher = GRPOSearchWithLTXVideo(
        ltx_generator=ltx_generator,
        reward_calculator=reward_fn_wrapper,
    )
    
    candidates = searcher.grpo_search(
        prompt=prompt,
        num_candidates_per_round=candidates_per_round,
        num_rounds=num_rounds,
        base_config={
            'height': 512,
            'width': 768,
            'num_frames': 121,  # Full sequence!
            'frame_rate': 24,
            'num_inference_steps': 40,
            'guidance_scale': 7.5,
            'seed': 2025,
        }
    )
    
    # 6. Save results
    print("\nSaving results...")
    searcher.save_results(
        candidates=candidates,
        output_dir=Path(output_dir),
        num_to_save=min(5, len(candidates)),
    )
    
    # 7. Print summary
    print("\n" + "="*80)
    print("GRPO Search Complete!")
    print("="*80)
    print(f"\nTop 3 Candidates:")
    for idx, cand in enumerate(candidates[:3]):
        print(f"\n  Rank {idx+1}:")
        print(f"    Total Reward: {cand['total_reward']:.4f}")
        print(f"    Sequence Reward: {cand['reward_info']['sequence_reward']:.4f}")
        print(f"    Temporal Consistency: {cand['reward_info']['temporal_consistency']:.4f}")
        print(f"    Guidance Scale: {cand['config']['guidance_scale']:.2f}")
        print(f"    Inference Steps: {cand['config']['num_inference_steps']}")
        print(f"    Seed: {cand['config']['seed']}")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*80)
    
    return candidates


def main():
    parser = argparse.ArgumentParser(description="GRPO with Full Frame Sequence Calculation")
    parser.add_argument("--prompt", type=str, 
                       default="A red dot appears and slowly fades away",
                       help="Text prompt for generation")
    parser.add_argument("--output_dir", type=str,
                       default="outputs/grpo_full_sequence",
                       help="Output directory")
    parser.add_argument("--original_image", type=str,
                       default=None,
                       help="Path to original image with red dots (optional)")
    parser.add_argument("--num_rounds", type=int, default=3,
                       help="Number of GRPO rounds")
    parser.add_argument("--candidates_per_round", type=int, default=4,
                       help="Candidates per round")
    
    args = parser.parse_args()
    
    run_grpo_full_sequence_search(
        prompt=args.prompt,
        output_dir=args.output_dir,
        original_image_path=args.original_image,
        num_rounds=args.num_rounds,
        candidates_per_round=args.candidates_per_round,
    )


if __name__ == "__main__":
    main()

