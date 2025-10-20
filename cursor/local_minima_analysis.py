#!/usr/bin/env python3
"""
Analysis of Local Minima in Per-Frame vs Path-Based Rewards
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import math

class LocalMinimaAnalyzer:
    """
    Demonstrate why per-frame rewards fall into local minima
    """
    
    def __init__(self):
        self.frame_size = (64, 64)  # Simplified for analysis
        
    def simulate_reward_landscape(self, video_length: int = 10) -> dict:
        """
        Simulate reward landscapes for per-frame vs path-based optimization
        """
        results = {
            'per_frame_rewards': [],
            'path_rewards': [],
            'frame_positions': [],
            'path_positions': []
        }
        
        # Create a synthetic video sequence
        base_video = self._create_base_video(video_length)
        
        # Simulate per-frame optimization (local minima prone)
        per_frame_optimized = self._optimize_per_frame(base_video)
        
        # Simulate path-based optimization (global optimization)
        path_optimized = self._optimize_full_path(base_video)
        
        # Analyze reward landscapes
        results['per_frame_analysis'] = self._analyze_per_frame_landscape(per_frame_optimized)
        results['path_analysis'] = self._analyze_path_landscape(path_optimized)
        
        return results
    
    def _create_base_video(self, length: int) -> torch.Tensor:
        """Create a base video sequence for analysis"""
        # Simple synthetic video: moving circle
        video = torch.zeros(3, length, *self.frame_size)
        
        for t in range(length):
            # Circle position moves across frame
            center_x = int(self.frame_size[0] * (t / (length - 1)))
            center_y = self.frame_size[1] // 2
            
            # Create circle
            y, x = torch.meshgrid(
                torch.arange(self.frame_size[0]), 
                torch.arange(self.frame_size[1]),
                indexing='ij'
            )
            
            distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
            circle = (distance < 8).float()
            
            video[:, t] = circle.unsqueeze(0).repeat(3, 1, 1)
        
        return video
    
    def _optimize_per_frame(self, video: torch.Tensor) -> dict:
        """
        Simulate per-frame optimization (prone to local minima)
        """
        optimized_video = video.clone()
        frame_rewards = []
        optimization_paths = []
        
        for t in range(video.shape[1]):
            frame = optimized_video[:, t].clone()
            
            # Simulate frame-level optimization
            best_frame = frame.clone()
            best_reward = self._frame_reward(frame)
            path = [best_reward]
            
            # Gradient descent on individual frame
            for step in range(20):
                # Add noise for exploration (limited by frame-level view)
                noise = torch.randn_like(frame) * 0.1
                candidate_frame = frame + noise
                
                # Clip to valid range
                candidate_frame = torch.clamp(candidate_frame, 0, 1)
                
                # Evaluate frame reward (no temporal context!)
                reward = self._frame_reward(candidate_frame)
                path.append(reward)
                
                if reward > best_reward:
                    best_reward = reward
                    best_frame = candidate_frame
                    frame = candidate_frame  # Update for next iteration
                else:
                    # Stuck in local minimum - frame looks good individually
                    # but may not fit well in sequence
                    break
            
            optimized_video[:, t] = best_frame
            frame_rewards.append(best_reward)
            optimization_paths.append(path)
        
        return {
            'video': optimized_video,
            'frame_rewards': frame_rewards,
            'optimization_paths': optimization_paths,
            'total_reward': sum(frame_rewards) / len(frame_rewards)
        }
    
    def _optimize_full_path(self, video: torch.Tensor) -> dict:
        """
        Simulate path-based optimization (global optimization)
        """
        optimized_video = video.clone()
        path_rewards = []
        
        # Global optimization considering entire sequence
        best_video = optimized_video.clone()
        best_reward = self._path_reward(optimized_video)
        
        for step in range(50):  # More steps for global optimization
            # Add correlated noise across frames (temporal coherence)
            temporal_noise = self._generate_temporal_noise(video.shape)
            candidate_video = optimized_video + temporal_noise
            
            # Clip to valid range
            candidate_video = torch.clamp(candidate_video, 0, 1)
            
            # Evaluate full path reward (considers temporal relationships)
            reward = self._path_reward(candidate_video)
            path_rewards.append(reward)
            
            if reward > best_reward:
                best_reward = reward
                best_video = candidate_video
                optimized_video = candidate_video
            else:
                # Use momentum to escape local minima
                if step % 10 == 0:
                    # Larger exploration step
                    exploration_noise = self._generate_temporal_noise(video.shape) * 0.3
                    optimized_video = best_video + exploration_noise
                    optimized_video = torch.clamp(optimized_video, 0, 1)
        
        return {
            'video': best_video,
            'path_rewards': path_rewards,
            'total_reward': best_reward,
            'final_frame_rewards': [self._frame_reward(best_video[:, t]) for t in range(best_video.shape[1])]
        }
    
    def _frame_reward(self, frame: torch.Tensor) -> float:
        """
        Individual frame reward (limited perspective)
        """
        # Simple frame quality metrics
        sharpness = torch.diff(frame, dim=-1).abs().mean() + torch.diff(frame, dim=-2).abs().mean()
        brightness = frame.mean()
        contrast = frame.std()
        
        # Frame reward only sees individual frame quality
        reward = 0.4 * sharpness + 0.3 * contrast + 0.3 * (1.0 - abs(brightness - 0.5))
        return reward.item()
    
    def _path_reward(self, video: torch.Tensor) -> float:
        """
        Full path reward (global perspective)
        """
        T = video.shape[1]
        
        # 1. Average frame quality
        frame_qualities = [self._frame_reward(video[:, t]) for t in range(T)]
        avg_frame_quality = np.mean(frame_qualities)
        
        # 2. Temporal consistency (key difference!)
        temporal_diffs = torch.diff(video, dim=1)  # [C, T-1, H, W]
        temporal_consistency = 1.0 / (1.0 + temporal_diffs.var().item() * 100)
        
        # 3. Motion smoothness
        if T > 2:
            motion_acceleration = torch.diff(temporal_diffs, dim=1)  # [C, T-2, H, W]
            motion_smoothness = 1.0 / (1.0 + motion_acceleration.var().item() * 1000)
        else:
            motion_smoothness = 1.0
        
        # 4. Global coherence (something per-frame can't see)
        global_coherence = self._compute_global_coherence(video)
        
        # Path reward considers ALL aspects
        path_reward = (
            0.3 * avg_frame_quality +      # Individual frame quality
            0.3 * temporal_consistency +   # Smooth transitions
            0.2 * motion_smoothness +      # Natural motion
            0.2 * global_coherence         # Overall story coherence
        )
        
        return path_reward
    
    def _compute_global_coherence(self, video: torch.Tensor) -> float:
        """
        Global coherence that per-frame optimization can't capture
        """
        # Example: reward videos where object moves consistently
        T = video.shape[1]
        
        # Compute "center of mass" for each frame
        centers = []
        for t in range(T):
            frame = video[:, t].mean(dim=0)  # Average across channels
            
            # Find center of mass
            y_coords, x_coords = torch.meshgrid(
                torch.arange(frame.shape[0]), 
                torch.arange(frame.shape[1]),
                indexing='ij'
            )
            
            total_mass = frame.sum()
            if total_mass > 0:
                center_y = (frame * y_coords.float()).sum() / total_mass
                center_x = (frame * x_coords.float()).sum() / total_mass
                centers.append([center_x.item(), center_y.item()])
            else:
                centers.append([0, 0])
        
        # Reward smooth, consistent motion
        if len(centers) > 1:
            center_diffs = np.diff(centers, axis=0)
            motion_consistency = 1.0 / (1.0 + np.var(center_diffs) * 10)
        else:
            motion_consistency = 1.0
        
        return motion_consistency
    
    def _generate_temporal_noise(self, shape: tuple) -> torch.Tensor:
        """
        Generate temporally correlated noise for path optimization
        """
        C, T, H, W = shape
        
        # Create base noise
        base_noise = torch.randn(C, T, H, W) * 0.05
        
        # Add temporal correlation (smooth across time)
        for t in range(1, T):
            base_noise[:, t] = 0.7 * base_noise[:, t-1] + 0.3 * base_noise[:, t]
        
        return base_noise
    
    def _analyze_per_frame_landscape(self, per_frame_result: dict) -> dict:
        """
        Analyze local minima in per-frame optimization
        """
        paths = per_frame_result['optimization_paths']
        
        # Count local minima (places where optimization got stuck)
        local_minima_count = 0
        early_convergence = 0
        
        for path in paths:
            # Check if optimization converged early (local minimum)
            if len(path) < 15:  # Converged before 15 steps
                early_convergence += 1
            
            # Check for plateaus (stuck in local minimum)
            if len(path) > 5:
                recent_improvement = max(path[-5:]) - min(path[-5:])
                if recent_improvement < 0.01:  # Very small improvement
                    local_minima_count += 1
        
        return {
            'local_minima_count': local_minima_count,
            'early_convergence': early_convergence,
            'total_frames': len(paths),
            'avg_path_length': np.mean([len(p) for p in paths]),
            'local_minima_ratio': local_minima_count / len(paths)
        }
    
    def _analyze_path_landscape(self, path_result: dict) -> dict:
        """
        Analyze global optimization in path-based approach
        """
        rewards = path_result['path_rewards']
        
        # Analyze convergence behavior
        improvement_over_time = np.diff(rewards)
        positive_improvements = sum(1 for x in improvement_over_time if x > 0)
        
        # Check for global optimization characteristics
        final_improvement = rewards[-1] - rewards[0]
        sustained_improvement = len([x for x in improvement_over_time[-10:] if x > 0])
        
        return {
            'total_improvement': final_improvement,
            'positive_steps': positive_improvements,
            'total_steps': len(rewards),
            'sustained_improvement': sustained_improvement,
            'final_reward': rewards[-1],
            'exploration_ratio': positive_improvements / len(improvement_over_time) if improvement_over_time.size > 0 else 0
        }

def demonstrate_local_minima_problem():
    """
    Demonstrate the local minima problem with per-frame rewards
    """
    analyzer = LocalMinimaAnalyzer()
    results = analyzer.simulate_reward_landscape(video_length=8)
    
    print("=== Local Minima Analysis ===")
    print("\n📉 Per-Frame Optimization (Prone to Local Minima):")
    per_frame_analysis = results['per_frame_analysis']
    print(f"  Local minima encountered: {per_frame_analysis['local_minima_count']}/{per_frame_analysis['total_frames']}")
    print(f"  Local minima ratio: {per_frame_analysis['local_minima_ratio']:.2%}")
    print(f"  Early convergence: {per_frame_analysis['early_convergence']} frames")
    print(f"  Average optimization steps: {per_frame_analysis['avg_path_length']:.1f}")
    
    print("\n🌍 Path-Based Optimization (Global Optimization):")
    path_analysis = results['path_analysis']
    print(f"  Total improvement: {path_analysis['total_improvement']:.3f}")
    print(f"  Positive steps: {path_analysis['positive_steps']}/{path_analysis['total_steps']}")
    print(f"  Exploration ratio: {path_analysis['exploration_ratio']:.2%}")
    print(f"  Sustained improvement: {path_analysis['sustained_improvement']}/10 recent steps")
    print(f"  Final reward: {path_analysis['final_reward']:.3f}")
    
    print("\n🎯 Key Insights:")
    print("  • Per-frame optimization gets stuck in local minima")
    print("  • Path-based optimization explores globally")
    print("  • Temporal relationships help escape local minima")
    print("  • Global coherence rewards require full sequence view")

if __name__ == "__main__":
    demonstrate_local_minima_problem()
