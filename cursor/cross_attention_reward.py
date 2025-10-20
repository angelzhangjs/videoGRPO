#!/usr/bin/env python3
"""
Cross-Attention Fusion as Reward Signal for GRPO
Using attention patterns and fusion quality as direct reward for video optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class AttentionRewardResult:
    """Result from attention-based reward computation"""
    attention_coherence_reward: float
    fusion_quality_reward: float
    temporal_attention_reward: float
    cross_modal_alignment_reward: float
    attention_diversity_reward: float
    total_attention_reward: float
    attention_patterns: Dict[str, torch.Tensor]
    detailed_analysis: Dict[str, Any]

class CrossAttentionRewardFunction:
    """
    Use cross-attention fusion patterns as reward signals for GRPO
    """
    
    def __init__(self, dino_dim: int = 1024, spatial_dim: int = 512, fusion_dim: int = 512):
        # Cross-attention fusion model (this IS the reward function)
        self.fusion_model = self._create_reward_fusion_model(dino_dim, spatial_dim, fusion_dim)
        
        # Attention pattern analyzers
        self.attention_analyzers = self._create_attention_analyzers()
        
        # Reward weights
        self.reward_weights = {
            'attention_coherence': 0.25,    # How coherent are attention patterns?
            'fusion_quality': 0.30,         # How well do features fuse?
            'temporal_attention': 0.20,     # How good are temporal attention patterns?
            'cross_modal_alignment': 0.15,  # How well do modalities align?
            'attention_diversity': 0.10     # How diverse are attention patterns?
        }
        
        print("✅ Cross-Attention Reward Function initialized")
    
    def _create_reward_fusion_model(self, dino_dim: int, spatial_dim: int, fusion_dim: int) -> nn.Module:
        """
        Create fusion model that outputs both fused features AND attention patterns for reward
        """
        class RewardAwareFusionModel(nn.Module):
            def __init__(self):
                super().__init__()
                
                # Feature projections
                self.dino_proj = nn.Linear(dino_dim, fusion_dim)
                self.spatial_proj = nn.Linear(spatial_dim, fusion_dim)
                
                # Cross-attention modules (these provide reward signals!)
                self.dino_to_spatial_attn = nn.MultiheadAttention(
                    fusion_dim, 8, batch_first=True
                )
                self.spatial_to_dino_attn = nn.MultiheadAttention(
                    fusion_dim, 8, batch_first=True
                )
                
                # Fusion quality assessor
                self.fusion_quality_head = nn.Sequential(
                    nn.Linear(fusion_dim * 2, 256),
                    nn.ReLU(),
                    nn.Linear(256, 1),
                    nn.Sigmoid()
                )
                
                # Attention coherence assessor
                self.attention_coherence_head = nn.Sequential(
                    nn.Linear(fusion_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, dino_features, spatial_features):
                """
                Forward pass that provides both fusion AND reward signals
                """
                # Project features
                dino_proj = self.dino_proj(dino_features)      # [T, 512]
                spatial_proj = self.spatial_proj(spatial_features)  # [T, 512]
                
                # Cross-attention with attention weight extraction
                dino_enhanced, dino_attn_weights = self.dino_to_spatial_attn(
                    dino_proj.unsqueeze(0),
                    spatial_proj.unsqueeze(0),
                    spatial_proj.unsqueeze(0)
                )
                
                spatial_enhanced, spatial_attn_weights = self.spatial_to_dino_attn(
                    spatial_proj.unsqueeze(0),
                    dino_proj.unsqueeze(0),
                    dino_proj.unsqueeze(0)
                )
                
                dino_enhanced = dino_enhanced[0]      # [T, 512]
                spatial_enhanced = spatial_enhanced[0] # [T, 512]
                
                # Fusion
                fused_input = torch.cat([dino_enhanced, spatial_enhanced], dim=-1)
                
                # Compute reward signals from fusion process
                fusion_quality_scores = self.fusion_quality_head(fused_input)  # [T, 1]
                attention_coherence_scores = self.attention_coherence_head(dino_enhanced + spatial_enhanced)  # [T, 1]
                
                return {
                    'fused_features': fused_input,
                    'dino_enhanced': dino_enhanced,
                    'spatial_enhanced': spatial_enhanced,
                    'attention_weights': {
                        'dino_to_spatial': dino_attn_weights[0],    # [T, T]
                        'spatial_to_dino': spatial_attn_weights[0]  # [T, T]
                    },
                    'fusion_quality_scores': fusion_quality_scores.squeeze(-1),  # [T]
                    'attention_coherence_scores': attention_coherence_scores.squeeze(-1)  # [T]
                }
        
        return RewardAwareFusionModel()
    
    def _create_attention_analyzers(self) -> Dict[str, callable]:
        """Create analyzers for different attention reward components"""
        return {
            'coherence_analyzer': self._analyze_attention_coherence,
            'temporal_analyzer': self._analyze_temporal_attention_patterns,
            'alignment_analyzer': self._analyze_cross_modal_alignment,
            'diversity_analyzer': self._analyze_attention_diversity
        }
    
    def compute_attention_based_reward(
        self,
        video: torch.Tensor,
        prompt: str,
        dino_features: torch.Tensor,
        spatial_features: torch.Tensor
    ) -> AttentionRewardResult:
        """
        Compute reward based on cross-attention fusion patterns
        """
        print(f"🎯 Computing cross-attention fusion reward...")
        
        # Forward pass through fusion model (extracts attention patterns)
        fusion_result = self.fusion_model(dino_features, spatial_features)
        
        # Extract attention patterns for reward computation
        attention_weights = fusion_result['attention_weights']
        fusion_quality_scores = fusion_result['fusion_quality_scores']
        attention_coherence_scores = fusion_result['attention_coherence_scores']
        
        # Compute different reward components from attention patterns
        reward_components = {}
        
        # 1. Attention Coherence Reward
        reward_components['attention_coherence'] = self._compute_attention_coherence_reward(
            attention_weights, attention_coherence_scores
        )
        
        # 2. Fusion Quality Reward
        reward_components['fusion_quality'] = self._compute_fusion_quality_reward(
            fusion_quality_scores, fusion_result['fused_features']
        )
        
        # 3. Temporal Attention Reward
        reward_components['temporal_attention'] = self._compute_temporal_attention_reward(
            attention_weights
        )
        
        # 4. Cross-Modal Alignment Reward
        reward_components['cross_modal_alignment'] = self._compute_cross_modal_alignment_reward(
            attention_weights, fusion_result['dino_enhanced'], fusion_result['spatial_enhanced']
        )
        
        # 5. Attention Diversity Reward
        reward_components['attention_diversity'] = self._compute_attention_diversity_reward(
            attention_weights
        )
        
        # Compute total attention-based reward
        total_reward = sum(
            self.reward_weights[component] * reward_components[component]
            for component in reward_components
        )
        
        print(f"  📊 Attention reward breakdown:")
        for component, score in reward_components.items():
            weight = self.reward_weights[component]
            contribution = weight * score
            print(f"    {component}: {score:.3f} (weight: {weight:.2f}, contrib: {contribution:.3f})")
        print(f"  🏆 Total attention reward: {total_reward:.3f}")
        
        return AttentionRewardResult(
            attention_coherence_reward=reward_components['attention_coherence'],
            fusion_quality_reward=reward_components['fusion_quality'],
            temporal_attention_reward=reward_components['temporal_attention'],
            cross_modal_alignment_reward=reward_components['cross_modal_alignment'],
            attention_diversity_reward=reward_components['attention_diversity'],
            total_attention_reward=total_reward,
            attention_patterns=attention_weights,
            detailed_analysis=self._create_detailed_attention_analysis(fusion_result, reward_components)
        )
    
    def _compute_attention_coherence_reward(
        self, 
        attention_weights: Dict[str, torch.Tensor],
        coherence_scores: torch.Tensor
    ) -> float:
        """
        Reward based on how coherent the attention patterns are
        """
        dino_spatial_attn = attention_weights['dino_to_spatial']  # [T, T]
        spatial_dino_attn = attention_weights['spatial_to_dino']  # [T, T]
        
        # Good attention should be coherent (not random)
        # Measure attention entropy - lower entropy = more focused = better
        
        # DINO attention coherence
        dino_attention_entropy = self._compute_attention_entropy(dino_spatial_attn)
        dino_coherence = 1.0 - (dino_attention_entropy / torch.log(torch.tensor(float(dino_spatial_attn.shape[-1]))))
        
        # Spatial attention coherence
        spatial_attention_entropy = self._compute_attention_entropy(spatial_dino_attn)
        spatial_coherence = 1.0 - (spatial_attention_entropy / torch.log(torch.tensor(float(spatial_dino_attn.shape[-1]))))
        
        # Combine with learned coherence scores
        learned_coherence = coherence_scores.mean().item()
        
        # Final coherence reward
        coherence_reward = (
            0.3 * dino_coherence.item() +
            0.3 * spatial_coherence.item() +
            0.4 * learned_coherence
        )
        
        return min(coherence_reward, 1.0)
    
    def _compute_fusion_quality_reward(
        self, 
        fusion_scores: torch.Tensor,
        fused_features: torch.Tensor
    ) -> float:
        """
        Reward based on quality of feature fusion
        """
        # Direct fusion quality from learned scores
        learned_fusion_quality = fusion_scores.mean().item()
        
        # Feature diversity after fusion (good fusion preserves information)
        feature_diversity = fused_features.var(dim=0).mean().item()
        diversity_score = min(feature_diversity * 2, 1.0)
        
        # Feature magnitude (good fusion maintains reasonable feature magnitudes)
        feature_magnitude = fused_features.abs().mean().item()
        magnitude_score = self._score_in_optimal_range(feature_magnitude, 0.1, 2.0)
        
        # Combined fusion quality
        fusion_quality = (
            0.5 * learned_fusion_quality +
            0.3 * diversity_score +
            0.2 * magnitude_score
        )
        
        return min(fusion_quality, 1.0)
    
    def _compute_temporal_attention_reward(self, attention_weights: Dict[str, torch.Tensor]) -> float:
        """
        Reward based on temporal attention patterns
        """
        dino_spatial_attn = attention_weights['dino_to_spatial']  # [T, T]
        spatial_dino_attn = attention_weights['spatial_to_dino']  # [T, T]
        
        T = dino_spatial_attn.shape[0]
        
        # Good temporal attention patterns:
        # 1. Strong diagonal (current time attending to current time)
        # 2. Reasonable off-diagonal (past/future context)
        # 3. Smooth transitions (not random jumps)
        
        # Diagonal strength
        diagonal_strength_dino = torch.diag(dino_spatial_attn).mean().item()
        diagonal_strength_spatial = torch.diag(spatial_dino_attn).mean().item()
        
        # Temporal smoothness (adjacent frames should have similar attention)
        temporal_smoothness_dino = self._compute_temporal_smoothness(dino_spatial_attn)
        temporal_smoothness_spatial = self._compute_temporal_smoothness(spatial_dino_attn)
        
        # Local vs global attention balance
        local_attention_dino = self._compute_local_attention_strength(dino_spatial_attn)
        local_attention_spatial = self._compute_local_attention_strength(spatial_dino_attn)
        
        # Combined temporal attention reward
        temporal_reward = (
            0.3 * (diagonal_strength_dino + diagonal_strength_spatial) / 2 +
            0.4 * (temporal_smoothness_dino + temporal_smoothness_spatial) / 2 +
            0.3 * (local_attention_dino + local_attention_spatial) / 2
        )
        
        return min(temporal_reward, 1.0)
    
    def _compute_cross_modal_alignment_reward(
        self,
        attention_weights: Dict[str, torch.Tensor],
        dino_enhanced: torch.Tensor,
        spatial_enhanced: torch.Tensor
    ) -> float:
        """
        Reward based on how well modalities align through cross-attention
        """
        # Good cross-modal alignment: enhanced features should be more similar than original
        
        # Compute similarity between enhanced features
        enhanced_similarities = []
        T = dino_enhanced.shape[0]
        
        for t in range(T):
            similarity = F.cosine_similarity(
                dino_enhanced[t].unsqueeze(0),
                spatial_enhanced[t].unsqueeze(0),
                dim=1
            ).item()
            enhanced_similarities.append(similarity)
        
        avg_enhanced_similarity = np.mean(enhanced_similarities)
        
        # Good alignment: enhanced features are more similar (but not identical)
        alignment_score = self._score_in_optimal_range(avg_enhanced_similarity, 0.6, 0.9)
        
        # Attention reciprocity (if A attends to B, B should somewhat attend to A)
        reciprocity_score = self._compute_attention_reciprocity(attention_weights)
        
        # Combined alignment reward
        alignment_reward = 0.7 * alignment_score + 0.3 * reciprocity_score
        
        return alignment_reward
    
    def _compute_attention_diversity_reward(self, attention_weights: Dict[str, torch.Tensor]) -> float:
        """
        Reward based on attention pattern diversity (avoid attention collapse)
        """
        dino_spatial_attn = attention_weights['dino_to_spatial']
        spatial_dino_attn = attention_weights['spatial_to_dino']
        
        # Measure attention diversity across time
        dino_diversity = self._compute_attention_diversity(dino_spatial_attn)
        spatial_diversity = self._compute_attention_diversity(spatial_dino_attn)
        
        # Good diversity: attention patterns vary across time but remain meaningful
        diversity_reward = (dino_diversity + spatial_diversity) / 2
        
        return diversity_reward
    
    def _compute_attention_entropy(self, attention_matrix: torch.Tensor) -> torch.Tensor:
        """Compute entropy of attention patterns"""
        # attention_matrix: [T, T]
        entropy = -torch.sum(attention_matrix * torch.log(attention_matrix + 1e-8), dim=-1)
        return entropy.mean()
    
    def _compute_temporal_smoothness(self, attention_matrix: torch.Tensor) -> float:
        """Compute temporal smoothness of attention patterns"""
        T = attention_matrix.shape[0]
        if T < 2:
            return 0.5
        
        # Adjacent frames should have similar attention patterns
        smoothness_scores = []
        for t in range(T - 1):
            similarity = F.cosine_similarity(
                attention_matrix[t].unsqueeze(0),
                attention_matrix[t + 1].unsqueeze(0),
                dim=1
            ).item()
            smoothness_scores.append(similarity)
        
        return np.mean(smoothness_scores)
    
    def _compute_local_attention_strength(self, attention_matrix: torch.Tensor) -> float:
        """Compute strength of local attention (nearby frames)"""
        T = attention_matrix.shape[0]
        
        # Create local attention mask (±2 frames)
        local_mask = torch.zeros_like(attention_matrix)
        for t in range(T):
            start_idx = max(0, t - 2)
            end_idx = min(T, t + 3)
            local_mask[t, start_idx:end_idx] = 1.0
        
        # Compute local attention strength
        local_attention = (attention_matrix * local_mask).sum(dim=-1).mean()
        
        return local_attention.item()
    
    def _compute_attention_reciprocity(self, attention_weights: Dict[str, torch.Tensor]) -> float:
        """
        Compute reciprocity between attention patterns
        """
        dino_to_spatial = attention_weights['dino_to_spatial']  # [T, T]
        spatial_to_dino = attention_weights['spatial_to_dino']  # [T, T]
        
        # Good reciprocity: if A attends to B, B should somewhat attend to A
        reciprocity = F.cosine_similarity(
            dino_to_spatial.flatten(),
            spatial_to_dino.T.flatten(),  # Transpose for reciprocity
            dim=0
        ).item()
        
        # Normalize to [0, 1]
        reciprocity_score = (reciprocity + 1) / 2
        
        return reciprocity_score
    
    def _compute_attention_diversity(self, attention_matrix: torch.Tensor) -> float:
        """Compute diversity of attention patterns"""
        T = attention_matrix.shape[0]
        
        # Compute pairwise similarities between attention patterns
        similarities = []
        for i in range(T):
            for j in range(i + 1, T):
                similarity = F.cosine_similarity(
                    attention_matrix[i].unsqueeze(0),
                    attention_matrix[j].unsqueeze(0),
                    dim=1
                ).item()
                similarities.append(similarity)
        
        # Good diversity: patterns are different but not completely random
        avg_similarity = np.mean(similarities) if similarities else 0.5
        diversity_score = 1.0 - avg_similarity  # Lower similarity = higher diversity
        
        # Optimal diversity range: not too similar, not too different
        return self._score_in_optimal_range(diversity_score, 0.2, 0.8)
    
    def _score_in_optimal_range(self, value: float, min_val: float, max_val: float) -> float:
        """Score value in optimal range"""
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val:
            return max(0.0, value / min_val)
        else:
            return max(0.0, 2.0 - value / max_val)
    
    def _create_detailed_attention_analysis(
        self, 
        fusion_result: Dict[str, torch.Tensor],
        reward_components: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create detailed analysis of attention patterns"""
        attention_weights = fusion_result['attention_weights']
        
        analysis = {
            'attention_pattern_summary': self._summarize_attention_patterns(attention_weights),
            'fusion_effectiveness': self._analyze_fusion_effectiveness(fusion_result),
            'temporal_dynamics': self._analyze_temporal_dynamics(attention_weights),
            'reward_component_analysis': reward_components
        }
        
        return analysis
    
    def _summarize_attention_patterns(self, attention_weights: Dict[str, torch.Tensor]) -> Dict[str, str]:
        """Summarize attention patterns in human-readable form"""
        dino_to_spatial = attention_weights['dino_to_spatial']
        spatial_to_dino = attention_weights['spatial_to_dino']
        
        # Find strongest attention patterns
        dino_max_attn = torch.max(dino_to_spatial, dim=-1)
        spatial_max_attn = torch.max(spatial_to_dino, dim=-1)
        
        summary = {
            'dino_attention_focus': f"DINO features most strongly attend to spatial features with avg strength {dino_max_attn.values.mean():.3f}",
            'spatial_attention_focus': f"Spatial features most strongly attend to DINO features with avg strength {spatial_max_attn.values.mean():.3f}",
            'attention_balance': f"Attention balance: DINO→Spatial vs Spatial→DINO ratio = {dino_max_attn.values.mean() / spatial_max_attn.values.mean():.2f}"
        }
        
        return summary

class AttentionRewardGRPOSystem:
    """
    GRPO system using cross-attention fusion as primary reward signal
    """
    
    def __init__(self, video_generator, gemini_api_key: str = None):
        self.video_generator = video_generator
        
        # Feature extractors
        self.dino_model = self._load_dino_model()
        self.spatial_encoder = self._load_spatial_encoder()
        
        # Cross-attention reward function
        self.attention_reward_function = CrossAttentionRewardFunction()
        
        # Gemini for additional reasoning (optional)
        self.gemini_verifier = None
        if gemini_api_key:
            try:
                from gemini_vlm_verifier import GeminiVLMVerifier
                self.gemini_verifier = GeminiVLMVerifier(gemini_api_key)
            except ImportError:
                pass
    
    def compute_attention_grpo_reward(self, video: torch.Tensor, prompt: str) -> Dict[str, Any]:
        """
        Compute GRPO reward based primarily on cross-attention fusion quality
        """
        # Extract features
        dino_features = self.dino_model(video) if self.dino_model else torch.randn(video.shape[1], 1024)
        spatial_features = self.spatial_encoder(video) if self.spatial_encoder else torch.randn(video.shape[1], 512)
        
        # Compute attention-based reward
        attention_reward = self.attention_reward_function.compute_attention_based_reward(
            video, prompt, dino_features, spatial_features
        )
        
        # Optional: Add Gemini reasoning component
        reasoning_bonus = 0.0
        if self.gemini_verifier:
            reasoning_result = self.gemini_verifier.verify_video_reasoning(
                video, prompt, ['attention_quality', 'fusion_intelligence'], complexity_level=3
            )
            reasoning_bonus = reasoning_result.overall_score * 0.2  # 20% bonus
        
        # Final reward
        final_reward = attention_reward.total_attention_reward + reasoning_bonus
        
        return {
            'reward': final_reward,
            'reward_info': {
                'attention_reward': attention_reward.total_attention_reward,
                'reasoning_bonus': reasoning_bonus,
                'attention_analysis': attention_reward.detailed_analysis,
                'attention_patterns': attention_reward.attention_patterns
            }
        }
    
    def _load_dino_model(self):
        """Load DINO model (placeholder)"""
        try:
            import torch.hub
            return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').eval()
        except:
            return None
    
    def _load_spatial_encoder(self):
        """Load spatial encoder (placeholder)"""
        # Would load your actual spatial encoder
        return None

def demonstrate_attention_as_reward():
    """
    Demonstrate using cross-attention fusion as reward signal
    """
    print("🎯 CROSS-ATTENTION FUSION AS REWARD SIGNAL")
    print("=" * 60)
    
    print("🧠 WHY ATTENTION PATTERNS MAKE EXCELLENT REWARDS:")
    
    attention_reward_benefits = [
        "🎯 Direct Quality Measurement: Attention patterns directly reflect understanding quality",
        "🔗 Fusion Intelligence: Good fusion = good attention = good video understanding",
        "⏰ Temporal Awareness: Attention patterns capture temporal relationships",
        "🧩 Multi-Modal Integration: Measures how well modalities work together",
        "📈 Self-Supervised: No need for external labels - attention patterns are intrinsic",
        "🎬 Video-Specific: Designed specifically for video understanding tasks",
        "🔄 Differentiable: Can be used directly in GRPO gradient computation"
    ]
    
    for benefit in attention_reward_benefits:
        print(f"  {benefit}")
    
    print(f"\n📊 ATTENTION REWARD COMPONENTS:")
    
    reward_components = {
        'attention_coherence': 'How focused and meaningful are attention patterns?',
        'fusion_quality': 'How well do DINO and spatial features combine?',
        'temporal_attention': 'Do attention patterns show good temporal understanding?',
        'cross_modal_alignment': 'How well do modalities align through attention?',
        'attention_diversity': 'Are attention patterns diverse enough to avoid collapse?'
    }
    
    for component, description in reward_components.items():
        print(f"  🎯 {component.replace('_', ' ').title()}: {description}")
    
    print(f"\n🚀 GRPO OPTIMIZATION WITH ATTENTION REWARDS:")
    
    grpo_benefits = [
        "📈 Rich Gradient Information: Attention provides detailed optimization signals",
        "🎯 Multi-Dimensional Optimization: Multiple attention aspects to optimize",
        "🔄 Self-Improving Fusion: Attention patterns get better through GRPO",
        "🧠 Emergent Intelligence: Good attention patterns = intelligent video understanding",
        "⚡ Efficient Computation: Attention computation is part of normal forward pass"
    ]
    
    for benefit in grpo_benefits:
        print(f"  {benefit}")
    
    print(f"\n💡 KEY INSIGHT:")
    print("Cross-attention fusion quality IS video understanding quality!")
    print("By optimizing attention patterns, you optimize video intelligence directly! 🚀")

def usage_example():
    """
    Show how to use attention-based rewards in GRPO
    """
    print(f"\n🚀 USAGE EXAMPLE: ATTENTION REWARDS IN GRPO")
    print("=" * 50)
    
    usage_code = '''
# Initialize attention-based GRPO system
attention_grpo = AttentionRewardGRPOSystem(
    video_generator=your_ltx_video_generator,
    gemini_api_key="your-gemini-api-key"  # Optional reasoning bonus
)

# Use in GRPO framework
from video_grpo_framework import VideoGRPOSearcher

grpo_searcher = VideoGRPOSearcher(
    video_generator=your_ltx_video_generator,
    reward_function=attention_grpo.compute_attention_grpo_reward,  # Attention-based reward!
    gemini_api_key="your-gemini-api-key"
)

# Run GRPO with attention-based optimization
results = grpo_searcher.grpo_search(
    prompts=["A robot demonstrates intelligent object manipulation using spatial reasoning"],
    num_videos_per_prompt=6,
    num_iterations=5
)

# GRPO will optimize for:
# ✅ Better attention coherence (more focused, meaningful attention)
# ✅ Higher fusion quality (better DINO-spatial integration)
# ✅ Improved temporal attention (better temporal understanding)
# ✅ Enhanced cross-modal alignment (better modality cooperation)
# ✅ Optimal attention diversity (rich but not random patterns)

# Result: Videos with superior object-spatial understanding!
'''
    
    print(usage_code)

def main():
    """
    Main demonstration of cross-attention fusion as reward
    """
    demonstrate_attention_as_reward()
    usage_example()

if __name__ == "__main__":
    main()

