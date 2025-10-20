#!/usr/bin/env python3
"""
Policy Update via Prompt Embedding Optimization
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Callable

class PromptPolicyOptimizer:
    """
    Optimize video generation by updating prompt embeddings during inference
    """
    
    def __init__(self, video_pipeline, reward_function: Callable):
        self.pipeline = video_pipeline
        self.reward_function = reward_function
    
    def optimize_prompt_policy(
        self,
        initial_prompt: str,
        num_optimization_steps: int = 15,
        learning_rate: float = 0.001,
        generation_config: Dict[str, Any] = None
    ) -> tuple:
        """
        Optimize prompt embeddings to maximize reward
        """
        if generation_config is None:
            generation_config = {
                'height': 512, 'width': 768, 'num_frames': 160,
                'guidance_scale': 7.5, 'num_inference_steps': 40
            }
        
        # Get initial prompt embedding
        with torch.no_grad():
            initial_embeds = self.pipeline.encode_prompt([initial_prompt])
        
        # Make prompt embeddings optimizable (policy parameters)
        prompt_embeds = initial_embeds.clone().detach().requires_grad_(True)
        
        # Optimizer for prompt embeddings
        optimizer = torch.optim.Adam([prompt_embeds], lr=learning_rate)
        
        print(f"Optimizing prompt policy from: '{initial_prompt}'")
        
        best_video = None
        best_reward = -float('inf')
        
        for step in range(num_optimization_steps):
            optimizer.zero_grad()
            
            # Generate video with current prompt embeddings
            with torch.autocast(device_type=self.pipeline.device.type):
                video = self.pipeline(
                    prompt_embeds=prompt_embeds,
                    height=generation_config['height'],
                    width=generation_config['width'],
                    num_frames=generation_config['num_frames'],
                    guidance_scale=generation_config['guidance_scale'],
                    num_inference_steps=generation_config['num_inference_steps'],
                    output_type="pt"
                ).frames
            
            # Compute reward
            reward_result = self.reward_function(video, initial_prompt)
            reward = torch.tensor(reward_result['reward'], requires_grad=True)
            
            # Policy loss (maximize reward)
            policy_loss = -reward
            
            # Backpropagate through prompt embeddings
            policy_loss.backward()
            
            # Update prompt embeddings (policy update!)
            optimizer.step()
            
            # Track best
            if reward.item() > best_reward:
                best_reward = reward.item()
                best_video = video.detach()
            
            print(f"Step {step+1:2d}: reward={reward.item():.3f}, loss={policy_loss.item():.3f}")
        
        return best_video, prompt_embeds.detach()

class ParameterPolicyOptimizer:
    """
    Optimize generation parameters as policy during inference
    """
    
    def __init__(self, video_pipeline, reward_function: Callable):
        self.pipeline = video_pipeline
        self.reward_function = reward_function
    
    def optimize_parameter_policy(
        self,
        prompt: str,
        num_optimization_steps: int = 20,
        learning_rate: float = 0.01
    ) -> Dict[str, Any]:
        """
        Optimize generation parameters (guidance_scale, etc.) to maximize reward
        """
        # Initialize optimizable parameters (policy)
        guidance_scale = torch.tensor(7.5, requires_grad=True)
        temperature = torch.tensor(1.0, requires_grad=True)  # For sampling-based generation
        
        # Optimizer for generation parameters
        optimizer = torch.optim.Adam([guidance_scale, temperature], lr=learning_rate)
        
        print(f"Optimizing generation parameters for: '{prompt}'")
        
        best_video = None
        best_reward = -float('inf')
        best_params = {}
        
        for step in range(num_optimization_steps):
            optimizer.zero_grad()
            
            # Ensure parameters stay in valid ranges
            current_guidance = torch.clamp(guidance_scale, 1.0, 20.0)
            current_temp = torch.clamp(temperature, 0.1, 2.0)
            
            # Generate video with current parameters
            video = self._generate_with_params(
                prompt, 
                guidance_scale=current_guidance.item(),
                temperature=current_temp.item()
            )
            
            # Compute reward
            reward_result = self.reward_function(video, prompt)
            reward = torch.tensor(reward_result['reward'], requires_grad=True)
            
            # Policy loss
            policy_loss = -reward
            policy_loss.backward()
            
            # Update parameters (policy update!)
            optimizer.step()
            
            # Track best
            if reward.item() > best_reward:
                best_reward = reward.item()
                best_video = video.detach()
                best_params = {
                    'guidance_scale': current_guidance.item(),
                    'temperature': current_temp.item()
                }
            
            print(f"Step {step+1:2d}: reward={reward.item():.3f}, "
                  f"guidance={current_guidance.item():.2f}, temp={current_temp.item():.2f}")
        
        return {
            'best_video': best_video,
            'best_params': best_params,
            'best_reward': best_reward
        }
    
    def _generate_with_params(self, prompt: str, guidance_scale: float, temperature: float) -> torch.Tensor:
        """Generate video with specific parameters"""
        with torch.no_grad():
            prompt_embeds = self.pipeline.encode_prompt([prompt])
            
            video = self.pipeline(
                prompt_embeds=prompt_embeds,
                height=512,
                width=768,
                num_frames=160,
                guidance_scale=guidance_scale,
                num_inference_steps=40,
                output_type="pt"
            ).frames
            
        return video


