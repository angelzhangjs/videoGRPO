#!/usr/bin/env python3
"""
Policy Update in Video Inference Search via Latent Space Optimization
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Callable, List
import numpy as np

class LatentPolicyOptimizer:
    """
    Optimize video generation by updating latent noise (policy) during inference
    """
    
    def __init__(self, video_pipeline, reward_function: Callable):
        self.pipeline = video_pipeline
        self.reward_function = reward_function
        
    def optimize_latent_policy(
        self,
        prompt: str,
        num_optimization_steps: int = 10,
        learning_rate: float = 0.01,
        num_candidates: int = 4,
        generation_config: Dict[str, Any] = None
    ) -> torch.Tensor:
        """
        Optimize latent noise to maximize reward (policy update in latent space)
        """
        if generation_config is None:
            generation_config = {
                'height': 512, 'width': 768, 'num_frames': 160,
                'guidance_scale': 7.5, 'num_inference_steps': 40
            }
        
        # Initialize multiple latent candidates (population-based optimization)
        batch_size = num_candidates
        latent_shape = (
            batch_size, 
            self.pipeline.vae.config.latent_channels,
            generation_config['num_frames'] // self.pipeline.vae.config.temporal_compression_ratio,
            generation_config['height'] // self.pipeline.vae.config.scaling_factor,
            generation_config['width'] // self.pipeline.vae.config.scaling_factor
        )
        
        # Initialize latent noise (this is our "policy parameters")
        latent_noise = torch.randn(latent_shape, device=self.pipeline.device, requires_grad=True)
        
        # Optimizer for latent noise
        optimizer = torch.optim.Adam([latent_noise], lr=learning_rate)
        
        print(f"Optimizing latent policy for: '{prompt}'")
        
        best_video = None
        best_reward = -float('inf')
        
        for step in range(num_optimization_steps):
            optimizer.zero_grad()
            
            # Generate videos from current latent noise
            with torch.autocast(device_type=self.pipeline.device.type):
                # Encode prompt
                prompt_embeds = self.pipeline.encode_prompt([prompt] * batch_size)
                
                # Generate videos using current latent noise
                videos = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    latents=latent_noise,  # Use our optimizable latents
                    height=generation_config['height'],
                    width=generation_config['width'],
                    num_frames=generation_config['num_frames'],
                    guidance_scale=generation_config['guidance_scale'],
                    num_inference_steps=generation_config['num_inference_steps'],
                    output_type="pt"
                ).frames
            
            # Compute rewards for each video
            rewards = []
            for i in range(batch_size):
                video = videos[i:i+1]  # Single video
                reward_result = self.reward_function(video, prompt)
                rewards.append(reward_result['reward'])
            
            rewards = torch.tensor(rewards, device=self.pipeline.device, requires_grad=True)
            
            # Policy gradient: maximize expected reward
            # Use REINFORCE-style gradient estimation
            baseline = rewards.mean().detach()  # Reduce variance
            advantages = rewards - baseline
            
            # Compute policy loss (negative reward to minimize)
            policy_loss = -advantages.mean()
            
            # Backpropagate through latent noise
            policy_loss.backward()
            
            # Update latent noise (policy update!)
            optimizer.step()
            
            # Track best video
            best_idx = rewards.argmax().item()
            if rewards[best_idx].item() > best_reward:
                best_reward = rewards[best_idx].item()
                best_video = videos[best_idx:best_idx+1].detach()
            
            print(f"Step {step+1:2d}: avg_reward={rewards.mean().item():.3f}, "
                  f"best_reward={rewards.max().item():.3f}, loss={policy_loss.item():.3f}")
        
        return best_video
    
    def gradient_guided_search(
        self,
        prompt: str,
        num_iterations: int = 5,
        candidates_per_iteration: int = 8,
        generation_config: Dict[str, Any] = None
    ) -> torch.Tensor:
        """
        Iterative policy improvement using gradient information
        """
        current_latent = None
        best_video = None
        best_reward = -float('inf')
        
        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            
            if current_latent is None:
                # Initialize random latent
                latent_shape = self._get_latent_shape(generation_config)
                current_latent = torch.randn(latent_shape, device=self.pipeline.device)
            
            # Generate candidates around current latent
            candidates = []
            candidate_latents = []
            
            for i in range(candidates_per_iteration):
                # Add noise for exploration
                noise_scale = 0.1 * (1.0 - iteration / num_iterations)  # Decrease over time
                candidate_latent = current_latent + torch.randn_like(current_latent) * noise_scale
                candidate_latents.append(candidate_latent)
                
                # Generate video
                video = self._generate_from_latent(candidate_latent, prompt, generation_config)
                reward_result = self.reward_function(video, prompt)
                
                candidates.append({
                    'video': video,
                    'latent': candidate_latent,
                    'reward': reward_result['reward']
                })
                
                print(f"  Candidate {i+1}: reward={reward_result['reward']:.3f}")
            
            # Select best candidates
            candidates.sort(key=lambda x: x['reward'], reverse=True)
            top_candidates = candidates[:candidates_per_iteration//2]
            
            # Update current latent (policy update)
            # Weighted average of top performers
            weights = torch.softmax(torch.tensor([c['reward'] for c in top_candidates]), dim=0)
            current_latent = sum(w * c['latent'] for w, c in zip(weights, top_candidates))
            
            # Track global best
            if candidates[0]['reward'] > best_reward:
                best_reward = candidates[0]['reward']
                best_video = candidates[0]['video']
            
            print(f"Best reward this iteration: {candidates[0]['reward']:.3f}")
        
        return best_video
    
    def _get_latent_shape(self, generation_config: Dict[str, Any]) -> tuple:
        """Get the shape for latent noise"""
        return (
            1,  # batch_size
            self.pipeline.vae.config.latent_channels,
            generation_config['num_frames'] // self.pipeline.vae.config.temporal_compression_ratio,
            generation_config['height'] // self.pipeline.vae.config.scaling_factor,
            generation_config['width'] // self.pipeline.vae.config.scaling_factor
        )
    
    def _generate_from_latent(
        self, 
        latent: torch.Tensor, 
        prompt: str, 
        generation_config: Dict[str, Any]
    ) -> torch.Tensor:
        """Generate video from specific latent"""
        with torch.no_grad():
            prompt_embeds = self.pipeline.encode_prompt([prompt])
            
            video = self.pipeline(
                prompt_embeds=prompt_embeds,
                latents=latent.unsqueeze(0),
                height=generation_config['height'],
                width=generation_config['width'],
                num_frames=generation_config['num_frames'],
                guidance_scale=generation_config['guidance_scale'],
                num_inference_steps=generation_config['num_inference_steps'],
                output_type="pt"
            ).frames
            
        return video
