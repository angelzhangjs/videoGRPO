#!/usr/bin/env python3
"""
GRPO for Red Dot Recognition Task
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
import cv2
from pathlib import Path


class RedDotRecognitionReward:
    """
    Reward function that evaluates red dot recognition in generated videos
    """
    
    def __init__(self, original_image_path: str):
        """
        Args:
            original_image_path: Path to the original image with red dots
        """
        self.original_image = cv2.imread(original_image_path)
        self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        # Detect red dots in original image
        self.original_red_dots = self._detect_red_dots(self.original_image_rgb)
        print(f"Detected {len(self.original_red_dots)} red dots in original image")
    
    def _detect_red_dots(self, image_rgb: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect red dots in image using color thresholding
        Returns list of (x, y, radius) tuples
        """
        # Convert to HSV for better red detection
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Red color range in HSV (handle wrap-around)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours (dots)
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        dots = []
        for contour in contours:
            # Get circle parameters
            (x, y), radius = cv2.minEnclosingCircle(contour)
            area = cv2.contourArea(contour)
            
            # Filter by size (adjust thresholds as needed)
            if 5 < area < 5000 and radius > 2:
                dots.append((int(x), int(y), int(radius)))
        
        return dots
    
    def compute_reward(self, video_frames: torch.Tensor, prompt: str) -> Dict[str, float]:
        """
        Compute reward for red dot recognition task
        
        Args:
            video_frames: [B, C, T, H, W] or [C, T, H, W]
            prompt: Text prompt (e.g., "erase the red dots")
        
        Returns:
            Dictionary with reward components
        """
        if len(video_frames.shape) == 5:
            video_frames = video_frames[0]  # Take first batch
        
        # Convert to numpy [T, H, W, C]
        video_np = video_frames.permute(1, 2, 3, 0).cpu().numpy()
        video_np = (video_np * 255).astype(np.uint8)
        
        T = video_np.shape[0]
        
        # Analyze red dots across frames
        rewards = {}
        
        # 1. Red Dot Tracking Score
        tracking_score = self._compute_tracking_score(video_np)
        rewards['tracking_score'] = tracking_score
        
        # 2. Task Completion Score (based on prompt)
        if "erase" in prompt.lower() or "remove" in prompt.lower():
            completion_score = self._compute_erasing_score(video_np)
            task_type = "erasing"
        elif "connect" in prompt.lower() or "link" in prompt.lower():
            completion_score = self._compute_connecting_score(video_np)
            task_type = "connecting"
        elif "highlight" in prompt.lower() or "mark" in prompt.lower():
            completion_score = self._compute_highlighting_score(video_np)
            task_type = "highlighting"
        else:
            completion_score = 0.5
            task_type = "general"
        
        rewards['completion_score'] = completion_score
        rewards['task_type'] = task_type
        
        # 3. Red Dot Disappearance Rate (for erasing task)
        disappearance_rate = self._compute_disappearance_rate(video_np)
        rewards['disappearance_rate'] = disappearance_rate
        
        # 4. Temporal Consistency
        consistency_score = self._compute_temporal_consistency(video_np)
        rewards['consistency_score'] = consistency_score
        
        # 5. Action Plausibility (does the action make sense?)
        plausibility_score = self._compute_action_plausibility(video_np, task_type)
        rewards['plausibility_score'] = plausibility_score
        
        # Combined reward
        weights = {
            'tracking_score': 0.25,
            'completion_score': 0.35,
            'disappearance_rate': 0.20,
            'consistency_score': 0.10,
            'plausibility_score': 0.10
        }
        
        total_reward = sum(rewards[key] * weights[key] 
                          for key in weights.keys() if key in rewards)
        
        rewards['total_reward'] = total_reward
        
        print(f"Reward breakdown: tracking={tracking_score:.3f}, "
              f"completion={completion_score:.3f}, "
              f"disappearance={disappearance_rate:.3f}, "
              f"total={total_reward:.3f}")
        
        return rewards
    
    def _compute_tracking_score(self, video_frames: np.ndarray) -> float:
        """
        Score: How well are red dots tracked across frames?
        """
        T = video_frames.shape[0]
        detected_per_frame = []
        
        for t in range(T):
            frame = video_frames[t]
            dots = self._detect_red_dots(frame)
            detected_per_frame.append(len(dots))
        
        # Score based on consistent detection
        if len(detected_per_frame) == 0:
            return 0.0
        
        # Good tracking = consistent number in early frames
        early_frames = detected_per_frame[:min(5, T//4)]
        if len(early_frames) > 0 and np.mean(early_frames) > 0:
            consistency = 1.0 - (np.std(early_frames) / (np.mean(early_frames) + 1e-6))
            return max(0.0, min(1.0, consistency))
        
        return 0.0
    
    def _compute_erasing_score(self, video_frames: np.ndarray) -> float:
        """
        Score: How well are red dots erased over time?
        """
        T = video_frames.shape[0]
        dot_counts = []
        
        for t in range(T):
            frame = video_frames[t]
            dots = self._detect_red_dots(frame)
            dot_counts.append(len(dots))
        
        if len(dot_counts) == 0:
            return 0.0
        
        # Good erasing = high count at start, low count at end
        start_count = np.mean(dot_counts[:T//4])
        end_count = np.mean(dot_counts[-T//4:])
        
        if start_count == 0:
            return 0.0
        
        reduction_rate = (start_count - end_count) / start_count
        return max(0.0, min(1.0, reduction_rate))
    
    def _compute_connecting_score(self, video_frames: np.ndarray) -> float:
        """
        Score: Are lines/connections appearing between dots?
        """
        T = video_frames.shape[0]
        connection_scores = []
        
        for t in range(min(T, 10)):  # Sample frames
            frame = video_frames[t]
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                   minLineLength=20, maxLineGap=10)
            
            if lines is not None:
                connection_scores.append(len(lines) / 10.0)  # Normalize
            else:
                connection_scores.append(0.0)
        
        return np.mean(connection_scores) if connection_scores else 0.0
    
    def _compute_highlighting_score(self, video_frames: np.ndarray) -> float:
        """
        Score: Are dots being highlighted/emphasized?
        """
        T = video_frames.shape[0]
        
        # Compare brightness/saturation around dots over time
        first_frame = video_frames[0]
        last_frame = video_frames[-1]
        
        first_dots = self._detect_red_dots(first_frame)
        
        if len(first_dots) == 0:
            return 0.0
        
        # Check if dots area became brighter/more prominent
        brightness_increase = 0
        for (x, y, r) in first_dots:
            # Extract region around dot
            roi_first = first_frame[max(0, y-r*2):y+r*2, max(0, x-r*2):x+r*2]
            roi_last = last_frame[max(0, y-r*2):y+r*2, max(0, x-r*2):x+r*2]
            
            if roi_first.size > 0 and roi_last.size > 0:
                bright_first = np.mean(roi_first)
                bright_last = np.mean(roi_last)
                brightness_increase += (bright_last - bright_first)
        
        return max(0.0, min(1.0, brightness_increase / (len(first_dots) * 50)))
    
    def _compute_disappearance_rate(self, video_frames: np.ndarray) -> float:
        """
        Measure how many dots disappear over time
        """
        T = video_frames.shape[0]
        
        first_frame_dots = len(self._detect_red_dots(video_frames[0]))
        last_frame_dots = len(self._detect_red_dots(video_frames[-1]))
        
        if first_frame_dots == 0:
            return 0.0
        
        disappearance_rate = (first_frame_dots - last_frame_dots) / first_frame_dots
        return max(0.0, min(1.0, disappearance_rate))
    
    def _compute_temporal_consistency(self, video_frames: np.ndarray) -> float:
        """
        Ensure smooth transitions (no flickering)
        """
        T = video_frames.shape[0]
        frame_diffs = []
        
        for t in range(T - 1):
            diff = np.abs(video_frames[t+1].astype(float) - video_frames[t].astype(float))
            frame_diffs.append(np.mean(diff))
        
        if len(frame_diffs) == 0:
            return 0.0
        
        # Lower variance in differences = more consistent
        consistency = 1.0 / (1.0 + np.var(frame_diffs))
        return consistency
    
    def _compute_action_plausibility(self, video_frames: np.ndarray, task_type: str) -> float:
        """
        Does the visual change look plausible for the task?
        """
        # Simple heuristic: check if there's visual activity
        T = video_frames.shape[0]
        
        total_motion = 0
        for t in range(T - 1):
            motion = np.abs(video_frames[t+1].astype(float) - video_frames[t].astype(float))
            total_motion += np.mean(motion)
        
        avg_motion = total_motion / (T - 1) if T > 1 else 0
        
        # Normalize to 0-1 range
        plausibility = min(1.0, avg_motion / 20.0)
        return plausibility


def main():
    """
    Example usage of Red Dot GRPO
    """
    print("=== Red Dot Recognition GRPO ===\n")
    
    # Initialize reward function
    reward_fn = RedDotRecognitionReward("ltx_video_source/images/dot.png")
    
    print("\nReward function initialized!")
    print(f"Original image has {len(reward_fn.original_red_dots)} red dots")
    print("\nThis reward function can evaluate:")
    print("  ✓ Red dot tracking accuracy")
    print("  ✓ Task completion (erasing/connecting/highlighting)")
    print("  ✓ Temporal consistency")
    print("  ✓ Action plausibility")
    print("\nIntegrate this with video_grpo_framework.py for full GRPO training!")


if __name__ == "__main__":
    main()

