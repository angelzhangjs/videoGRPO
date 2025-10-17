#!/usr/bin/env python3
"""
Neural Network-based Reward Model for Video Quality Assessment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import torchvision.transforms as transforms

class VideoRewardModel(nn.Module):
    """
    Neural network that learns to predict video quality scores
    """
    
    def __init__(self, input_channels=3, hidden_dim=512):
        super().__init__()
        
        # 3D CNN for spatiotemporal features
        self.conv3d_layers = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((4, 4, 4))  # Fixed output size
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)  # Single quality score
        )
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (B, C, T, H, W) tensor
        Returns:
            scores: (B,) tensor of quality scores
        """
        # Extract spatiotemporal features
        features = self.conv3d_layers(video)  # (B, 256, 4, 4, 4)
        
        # Flatten and predict score
        features = features.view(features.size(0), -1)  # (B, 256*4*4*4)
        scores = self.fc_layers(features).squeeze(-1)   # (B,)
        
        return torch.sigmoid(scores)  # Normalize to [0, 1]

class GeminiBasedAlignmentModel:
    """
    Gemini VLM-based reward model for superior prompt-video alignment
    Replaces CLIP with intelligent semantic understanding
    """
    
    def __init__(self, gemini_api_key: str):
        try:
            from gemini_vs_clip_alignment import GeminiTextVideoAligner
            self.gemini_aligner = GeminiTextVideoAligner(gemini_api_key)
            print("✅ Gemini VLM alignment model initialized")
        except ImportError:
            print("⚠️ Gemini VLM not available")
            self.gemini_aligner = None
    
    def compute_alignment_reward(self, video: torch.Tensor, text: str) -> torch.Tensor:
        """
        Compute superior text-video alignment using Gemini VLM
        """
        if not self.gemini_aligner:
            return torch.tensor(0.5)  # Fallback
        
        # Get comprehensive alignment analysis
        alignment_result = self.gemini_aligner.compute_text_video_alignment(video, text)
        
        # Convert to tensor for gradient computation
        alignment_score = torch.tensor(alignment_result.alignment_score, requires_grad=True)
        
        return alignment_score

class CLIPBasedRewardModel(nn.Module):
    """
    CLIP-based reward model for prompt-video alignment (DEPRECATED - use Gemini instead)
    """
    
    def __init__(self):
        super().__init__()
        print("⚠️ CLIP-based alignment is deprecated. Use GeminiBasedAlignmentModel for superior results.")
        try:
            import clip
            self.clip = clip  # Store clip module as instance variable
            self.clip_model, self.preprocess = clip.load("ViT-B/32")
            self.clip_model.eval()
            
            # Freeze CLIP parameters
            for param in self.clip_model.parameters():
                param.requires_grad = False
                
        except ImportError:
            print("CLIP not available. Install with: pip install git+https://github.com/openai/CLIP.git")
            self.clip_model = None
            self.clip = None
    
    def forward(self, video: torch.Tensor, text: str) -> torch.Tensor:
        """
        Compute CLIP similarity between video frames and text
        """
        if self.clip_model is None:
            return torch.zeros(video.size(0))
        
        batch_size, channels, num_frames, height, width = video.shape
        
        # Sample frames from video
        frame_indices = torch.linspace(0, num_frames-1, min(8, num_frames)).long()
        sampled_frames = video[:, :, frame_indices]  # (B, C, 8, H, W)
        
        # Reshape for CLIP processing
        frames = sampled_frames.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
        
        # Encode frames and text
        with torch.no_grad():
            # Resize frames to CLIP input size (224x224)
            frames_resized = F.interpolate(frames, size=(224, 224), mode='bilinear')
            frame_features = self.clip_model.encode_image(frames_resized)
            
            # Encode text
            text_tokens = self.clip.tokenize([text] * batch_size).to(video.device)
            text_features = self.clip_model.encode_text(text_tokens)
        
        # Compute similarities
        frame_features = frame_features.view(batch_size, len(frame_indices), -1)
        text_features = text_features.unsqueeze(1)  # (B, 1, D)
        
        # Average similarity across frames
        similarities = F.cosine_similarity(frame_features, text_features, dim=-1)
        avg_similarity = similarities.mean(dim=1)  # (B,)
        
        return (avg_similarity + 1) / 2  # Normalize to [0, 1]

class HybridRewardModel(nn.Module):
    """
    Combines multiple reward signals
    """
    
    def __init__(self):
        super().__init__()
        self.quality_model = VideoRewardModel()
        self.clip_model = CLIPBasedRewardModel()
        
        # Learnable weights for combining rewards
        self.quality_weight = nn.Parameter(torch.tensor(0.6))
        self.clip_weight = nn.Parameter(torch.tensor(0.4))
    
    def forward(self, video: torch.Tensor, text: str) -> torch.Tensor:
        """
        Combine quality and text-alignment rewards
        """
        quality_score = self.quality_model(video)
        clip_score = self.clip_model(video, text)
        
        # Weighted combination
        total_score = (self.quality_weight * quality_score + 
                      self.clip_weight * clip_score)
        
        return total_score

# Integration with your existing code
class NeuralRewardVideoGenerator:
    """
    Enhanced video generator with neural reward models
    """
    
    def __init__(self, config_path: str, reward_model_path: str = None):
        # Initialize base generator (your existing code)
        # self.base_generator = LTXVideoSearchGenerator(config_path)
        
        # Load or initialize reward model
        self.reward_model = HybridRewardModel()
        
        if reward_model_path and torch.cuda.is_available():
            try:
                checkpoint = torch.load(reward_model_path)
                self.reward_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded reward model from {reward_model_path}")
            except:
                print("Could not load reward model, using random initialization")
        
        self.reward_model.eval()
        if torch.cuda.is_available():
            self.reward_model = self.reward_model.cuda()
    
    def neural_score_video(self, video: torch.Tensor, prompt: str) -> float:
        """
        Score video using neural reward model
        """
        with torch.no_grad():
            if len(video.shape) == 4:  # Add batch dimension if needed
                video = video.unsqueeze(0)
            
            # Ensure video is in correct range [0, 1]
            if video.max() > 1.0:
                video = video / 255.0
            
            score = self.reward_model(video, prompt)
            return score.item()

# Example usage
def example_neural_reward():
    """
    Example of how to use neural reward models
    """
    # Create reward model
    reward_model = HybridRewardModel()
    
    # Dummy video and text
    video = torch.randn(1, 3, 16, 224, 224)  # (B, C, T, H, W)
    text = "A beautiful sunset over the ocean"
    
    # Get reward score
    with torch.no_grad():
        score = reward_model(video, text)
        print(f"Neural reward score: {score.item():.3f}")

if __name__ == "__main__":
    example_neural_reward()
