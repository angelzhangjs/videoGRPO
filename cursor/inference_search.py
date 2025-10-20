#!/usr/bin/env python3
"""
Inference Search Implementation for LTX-Video
Allows for custom search strategies during video generation
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import yaml
from dataclasses import dataclass

# LTX-Video imports
from ltx_video.models.transformers.transformer3d import Transformer3DModel
from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
from ltx_video.schedulers.rf import RectifiedFlowScheduler
from ltx_video.inference import InferenceConfig
from transformers import T5EncoderModel, T5Tokenizer

@dataclass
class SearchConfig:
    """Configuration for inference search"""
    search_type: str = "beam_search"  # beam_search, random_search, guided_search
    num_candidates: int = 4
    beam_width: int = 3
    search_steps: int = 10
    diversity_penalty: float = 0.1
    quality_threshold: float = 0.7

class LTXVideoSearchGenerator:
    """
    LTX-Video generator with inference search capabilities
    """
    
    def __init__(self, config_path: str, device: str = "cuda"):
        self.device = device
        self.config_path = config_path
        
        # Load pipeline configuration
        with open(config_path, 'r') as f:
            self.pipeline_config = yaml.safe_load(f)
        
        # Initialize components
        self._load_models()
        self._setup_pipeline()
    
    def _load_models(self):
        """Load all model components"""
        print("Loading LTX-Video models...")
        
        # Load transformer model
        transformer_config = self.pipeline_config.get('transformer', {})
        self.transformer = Transformer3DModel.from_pretrained(
            transformer_config.get('path', 'Lightricks/LTX-Video'),
            subfolder="transformer",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Load VAE
        vae_config = self.pipeline_config.get('vae', {})
        self.vae = CausalVideoAutoencoder.from_pretrained(
            vae_config.get('path', 'Lightricks/LTX-Video'),
            subfolder="vae",
            torch_dtype=torch.float16
        ).to(self.device)
        
        # Load text encoder
        text_encoder_path = self.pipeline_config.get('text_encoder', {}).get('path', 'PixArt-alpha/PixArt-Sigma-XL-2-1024-MS')
        self.text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_path,
            subfolder="text_encoder",
            torch_dtype=torch.float16
        ).to(self.device)
        
        self.tokenizer = T5Tokenizer.from_pretrained(
            text_encoder_path,
            subfolder="tokenizer"
        )
        
        # Load scheduler
        self.scheduler = RectifiedFlowScheduler()
        
        print("Models loaded successfully!")
    
    def _setup_pipeline(self):
        """Setup the LTX-Video pipeline"""
        self.pipeline = LTXVideoPipeline(
            transformer=self.transformer,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.scheduler
        )
        
        # Enable memory efficient attention if available
        try:
            self.pipeline.enable_model_cpu_offload()
            print("Enabled CPU offloading for memory efficiency")
        except:
            pass
    
    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode text prompt to embeddings"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            embeddings = self.text_encoder(**inputs).last_hidden_state
        
        return embeddings
    
    def beam_search_generation(
        self, 
        prompt: str, 
        search_config: SearchConfig,
        generation_config: Dict[str, Any]
    ) -> List[torch.Tensor]:
        """
        Implement beam search for video generation
        """
        print(f"Starting beam search with width {search_config.beam_width}")
        
        # Encode prompt
        prompt_embeds = self.encode_prompt(prompt)
        
        # Initialize beam candidates
        candidates = []
        
        # Generate initial candidates
        for i in range(search_config.beam_width):
            seed = generation_config.get('seed', 42) + i
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Generate video
            video = self.pipeline(
                prompt_embeds=prompt_embeds,
                height=generation_config.get('height', 512),
                width=generation_config.get('width', 768),
                num_frames=generation_config.get('num_frames', 160),
                num_inference_steps=generation_config.get('num_inference_steps', 50),
                guidance_scale=generation_config.get('guidance_scale', 7.5),
                generator=generator,
                output_type="pt"
            ).frames
            
            # Score the candidate
            score = self._score_video(video, prompt)
            candidates.append((video, score, seed))
            print(f"Candidate {i+1}: score = {score:.3f}")
        
        # Sort by score and return best candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [cand[0] for cand in candidates]
    
    def guided_search_generation(
        self,
        prompt: str,
        search_config: SearchConfig,
        generation_config: Dict[str, Any],
        guidance_function: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Implement guided search with custom guidance function
        """
        print("Starting guided search generation")
        
        if guidance_function is None:
            guidance_function = self._default_guidance
        
        best_video = None
        best_score = -float('inf')
        
        # Iterative refinement
        for step in range(search_config.search_steps):
            print(f"Search step {step + 1}/{search_config.search_steps}")
            
            # Adjust generation parameters based on guidance
            adjusted_config = guidance_function(generation_config, step, search_config.search_steps)
            
            # Generate video with adjusted parameters
            prompt_embeds = self.encode_prompt(prompt)
            generator = torch.Generator(device=self.device).manual_seed(
                adjusted_config.get('seed', 42) + step
            )
            
            video = self.pipeline(
                prompt_embeds=prompt_embeds,
                height=adjusted_config.get('height', 512),
                width=adjusted_config.get('width', 768),
                num_frames=adjusted_config.get('num_frames', 160),
                num_inference_steps=adjusted_config.get('num_inference_steps', 50),
                guidance_scale=adjusted_config.get('guidance_scale', 7.5),
                generator=generator,
                output_type="pt"
            ).frames
            
            # Score the video
            score = self._score_video(video, prompt)
            print(f"Step {step + 1} score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_video = video
                print(f"New best score: {best_score:.3f}")
        
        return best_video
    
    def _score_video(self, video: torch.Tensor, prompt: str) -> float:
        """
        Reward/Score function for video quality assessment
        This IS a reward function - it evaluates and guides generation quality
        """
        if len(video.shape) != 5:  # B, C, T, H, W
            return 0.0
        
        # Multiple reward components
        rewards = {}
        
        # 1. Motion Quality (temporal consistency)
        motion = torch.diff(video, dim=2).abs().mean().item()
        rewards['motion'] = min(motion * 10, 1.0)  # Normalize
        
        # 2. Visual Complexity (spatial richness)
        variance = video.var().item()
        rewards['complexity'] = min(variance * 5, 1.0)  # Normalize
        
        # 3. Temporal Smoothness (avoid flickering)
        temporal_diff = torch.diff(video, dim=2)
        smoothness = 1.0 / (1.0 + temporal_diff.var().item() * 100)
        rewards['smoothness'] = smoothness
        
        # 4. Color Diversity (avoid monochrome)
        color_std = video.std(dim=[-2, -1]).mean().item()
        rewards['color_diversity'] = min(color_std * 2, 1.0)
        
        # 5. Edge Sharpness (avoid blur)
        # Simple edge detection approximation
        edges_h = torch.diff(video, dim=-2).abs().mean().item()
        edges_v = torch.diff(video, dim=-1).abs().mean().item()
        sharpness = (edges_h + edges_v) / 2
        rewards['sharpness'] = min(sharpness * 20, 1.0)
        
        # Weighted combination (you can adjust these weights)
        weights = {
            'motion': 0.25,
            'complexity': 0.20,
            'smoothness': 0.20,
            'color_diversity': 0.15,
            'sharpness': 0.20
        }
        
        total_reward = sum(rewards[key] * weights[key] for key in rewards)
        
        # Optional: Print detailed scores for debugging
        # print(f"Reward breakdown: {rewards}, Total: {total_reward:.3f}")
        
        return total_reward
    
    def _default_guidance(self, config: Dict[str, Any], step: int, total_steps: int) -> Dict[str, Any]:
        """
        Default guidance function that progressively increases quality
        """
        progress = step / total_steps
        
        # Progressive refinement
        adjusted_config = config.copy()
        adjusted_config['guidance_scale'] = 5.0 + progress * 7.0  # 5.0 -> 12.0
        adjusted_config['num_inference_steps'] = int(30 + progress * 20)  # 30 -> 50
        
        return adjusted_config
    
    def generate_with_search(
        self,
        prompt: str,
        search_config: SearchConfig,
        generation_config: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Main generation method with search
        """
        if search_config.search_type == "beam_search":
            candidates = self.beam_search_generation(prompt, search_config, generation_config)
            return candidates[0]  # Return best candidate
        
        elif search_config.search_type == "guided_search":
            return self.guided_search_generation(prompt, search_config, generation_config)
        
        else:
            raise ValueError(f"Unknown search type: {search_config.search_type}")

# Example usage
def main():
    # Configuration
    config_path = "ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml"
    
    # Initialize generator
    generator = LTXVideoSearchGenerator(config_path, device="cuda:3")
    
    # Search configuration
    search_config = SearchConfig(
        search_type="beam_search",
        num_candidates=4,
        beam_width=3,
        search_steps=5
    )
    
    # Generation configuration
    generation_config = {
        'height': 512,
        'width': 768,
        'num_frames': 160,
        'num_inference_steps': 40,
        'guidance_scale': 7.5,
        'seed': 2025
    }
    
    # Generate video with search
    prompt = "A glowing jack-o'-lantern with flickering candlelight on a spooky porch"
    
    print(f"Generating video with search for prompt: {prompt}")
    video = generator.generate_with_search(prompt, search_config, generation_config)
    
    print(f"Generated video shape: {video.shape}")
    print("Video generation with search completed!")

if __name__ == "__main__":
    main()
