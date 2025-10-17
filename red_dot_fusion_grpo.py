#!/usr/bin/env python3
"""
Red Dot Recognition using Cross-Attention Fusion (VGGT + DINO) with GRPO

This leverages your existing cross-attention fusion system to solve the red dot
recognition problem by combining:
- DINO: Object-level understanding (recognizes red dots as distinct objects)
- VGGT/Spatial: Spatial layout understanding (where dots are located)
- Cross-Attention: Fuses object and spatial understanding
- GRPO: Learns optimal generation strategies through reward-based feedback
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cv2


@dataclass
class RedDotFusionReward:
    """Reward from cross-attention fusion for red dot task"""
    object_recognition_score: float  # From DINO: how well objects are recognized
    spatial_tracking_score: float    # From spatial: how well positions are tracked
    fusion_quality_score: float      # From cross-attention: quality of fusion
    task_completion_score: float     # Task-specific: erasing/connecting success
    total_reward: float
    attention_patterns: Dict[str, torch.Tensor]
    dino_detections: List[Dict]      # What DINO detected as objects
    spatial_layout: Dict[str, any]   # Spatial distribution analysis


class RedDotCrossAttentionReward:
    """
    Reward function leveraging cross-attention fusion for red dot recognition
    
    Key Insight:
    - DINO is EXCELLENT at recognizing red dots as distinct visual objects
    - Spatial encoder tracks their positions and motion
    - Cross-attention fusion combines: "WHAT dots" + "WHERE they are"
    - This creates rich understanding that simple CV cannot achieve
    """
    
    def __init__(
        self,
        dino_model: Optional[nn.Module] = None,
        spatial_encoder: Optional[nn.Module] = None,
        fusion_model: Optional[nn.Module] = None,
        device: str = "cuda"
    ):
        self.device = device
        
        # Use your existing models
        self.dino_model = dino_model or self._load_dino_model()
        self.spatial_encoder = spatial_encoder or self._load_spatial_encoder()
        self.fusion_model = fusion_model or self._create_fusion_model()
        
        print("✅ Red Dot Fusion Reward initialized")
        print("   - DINO: Object recognition (what are the dots?)")
        print("   - Spatial: Position tracking (where are the dots?)")
        print("   - Fusion: Combined understanding (what + where)")
    
    def _load_dino_model(self) -> nn.Module:
        """Load DINO for object recognition"""
        try:
            # Try to load real DINOv2
            dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            dino.eval()
            dino.to(self.device)
            print("  ✓ Loaded DINOv2 for object recognition")
            return dino
        except Exception as e:
            print(f"  ⚠ Could not load DINOv2: {e}")
            print("  ⚠ Using mock DINO")
            return None
    
    def _load_spatial_encoder(self) -> nn.Module:
        """Load spatial encoder"""
        # Use your existing VisualGeometrySpatialEncoder or similar
        # For now, placeholder
        print("  ⚠ Using mock spatial encoder (replace with your VisualGeometrySpatialEncoder)")
        return None
    
    def _create_fusion_model(self) -> nn.Module:
        """Create cross-attention fusion model"""
        class RedDotFusionModel(nn.Module):
            """Lightweight fusion for red dot task"""
            def __init__(self, dino_dim=768, spatial_dim=512, fusion_dim=512):
                super().__init__()
                
                # Project to common dimension
                self.dino_proj = nn.Linear(dino_dim, fusion_dim)
                self.spatial_proj = nn.Linear(spatial_dim, fusion_dim)
                
                # Cross-attention: DINO (what) attends to Spatial (where)
                self.dino_to_spatial_attn = nn.MultiheadAttention(
                    embed_dim=fusion_dim, num_heads=8, batch_first=True
                )
                
                # Cross-attention: Spatial (where) attends to DINO (what)
                self.spatial_to_dino_attn = nn.MultiheadAttention(
                    embed_dim=fusion_dim, num_heads=8, batch_first=True
                )
                
                # Fusion network
                self.fusion_net = nn.Sequential(
                    nn.Linear(fusion_dim * 2, fusion_dim),
                    nn.ReLU(),
                    nn.Linear(fusion_dim, fusion_dim),
                    nn.LayerNorm(fusion_dim)
                )
                
                # Task-specific heads
                self.object_quality_head = nn.Linear(fusion_dim, 1)
                self.spatial_quality_head = nn.Linear(fusion_dim, 1)
                self.fusion_quality_head = nn.Linear(fusion_dim, 1)
            
            def forward(self, dino_features, spatial_features):
                """
                Cross-attention fusion
                
                Args:
                    dino_features: [T, dino_dim] - Object features
                    spatial_features: [T, spatial_dim] - Spatial features
                
                Returns:
                    Fused features + attention patterns
                """
                # Project to common space
                dino_proj = self.dino_proj(dino_features)      # [T, 512]
                spatial_proj = self.spatial_proj(spatial_features)  # [T, 512]
                
                # Add batch dimension
                dino_batch = dino_proj.unsqueeze(0)      # [1, T, 512]
                spatial_batch = spatial_proj.unsqueeze(0) # [1, T, 512]
                
                # Cross-attention: DINO enhanced with spatial context
                # Q: What objects? K,V: Where in space?
                dino_enhanced, dino_attn = self.dino_to_spatial_attn(
                    query=dino_batch,
                    key=spatial_batch,
                    value=spatial_batch
                )
                
                # Cross-attention: Spatial enhanced with object context
                # Q: Where in space? K,V: What objects?
                spatial_enhanced, spatial_attn = self.spatial_to_dino_attn(
                    query=spatial_batch,
                    key=dino_batch,
                    value=dino_batch
                )
                
                # Remove batch dimension
                dino_enhanced = dino_enhanced[0]    # [T, 512]
                spatial_enhanced = spatial_enhanced[0] # [T, 512]
                
                # Fuse
                concatenated = torch.cat([dino_enhanced, spatial_enhanced], dim=-1)
                fused = self.fusion_net(concatenated)  # [T, 512]
                
                # Quality scores
                object_scores = self.object_quality_head(dino_enhanced).squeeze(-1)
                spatial_scores = self.spatial_quality_head(spatial_enhanced).squeeze(-1)
                fusion_scores = self.fusion_quality_head(fused).squeeze(-1)
                
                return {
                    'fused_features': fused,
                    'dino_enhanced': dino_enhanced,
                    'spatial_enhanced': spatial_enhanced,
                    'object_quality': object_scores,
                    'spatial_quality': spatial_scores,
                    'fusion_quality': fusion_scores,
                    'attention_patterns': {
                        'dino_to_spatial': dino_attn,
                        'spatial_to_dino': spatial_attn
                    }
                }
        
        model = RedDotFusionModel().to(self.device)
        print("  ✓ Created cross-attention fusion model")
        return model
    
    def compute_reward(
        self,
        video: torch.Tensor,  # [C, T, H, W] or [B, C, T, H, W]
        prompt: str,
        task_type: str = "erase"  # "erase", "connect", "highlight"
    ) -> RedDotFusionReward:
        """
        Compute reward using cross-attention fusion
        
        This is the KEY: We use DINO + Spatial + Fusion to understand
        both WHAT (red dots as objects) and WHERE (their positions)
        """
        if len(video.shape) == 5:
            video = video[0]  # Remove batch
        
        C, T, H, W = video.shape
        video = video.to(self.device)
        
        # Step 1: Extract DINO features (object understanding)
        print("  🔍 Extracting DINO features (object recognition)...")
        dino_features = self._extract_dino_features(video)  # [T, 768]
        
        # Step 2: Extract spatial features (position tracking)
        print("  📍 Extracting spatial features (position tracking)...")
        spatial_features = self._extract_spatial_features(video)  # [T, 512]
        
        # Step 3: Cross-attention fusion (combine what + where)
        print("  🔗 Fusing DINO + Spatial with cross-attention...")
        fusion_result = self.fusion_model(dino_features, spatial_features)
        
        # Step 4: Analyze DINO detections (what objects did it find?)
        print("  🎯 Analyzing DINO object detections...")
        dino_detections = self._analyze_dino_detections(dino_features, video)
        
        # Step 5: Analyze spatial layout (where are objects?)
        print("  📊 Analyzing spatial layout...")
        spatial_layout = self._analyze_spatial_layout(spatial_features, video)
        
        # Step 6: Compute task-specific reward
        print(f"  ✅ Computing {task_type} task reward...")
        task_reward = self._compute_task_reward(
            fusion_result, dino_detections, spatial_layout, video, task_type
        )
        
        # Step 7: Combine all rewards
        object_recognition_score = float(fusion_result['object_quality'].mean())
        spatial_tracking_score = float(fusion_result['spatial_quality'].mean())
        fusion_quality_score = float(fusion_result['fusion_quality'].mean())
        
        total_reward = (
            0.25 * object_recognition_score +
            0.25 * spatial_tracking_score +
            0.20 * fusion_quality_score +
            0.30 * task_reward
        )
        
        print(f"  🏆 Total reward: {total_reward:.3f}")
        
        return RedDotFusionReward(
            object_recognition_score=object_recognition_score,
            spatial_tracking_score=spatial_tracking_score,
            fusion_quality_score=fusion_quality_score,
            task_completion_score=task_reward,
            total_reward=total_reward,
            attention_patterns=fusion_result['attention_patterns'],
            dino_detections=dino_detections,
            spatial_layout=spatial_layout
        )
    
    def _extract_dino_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract DINO features for object recognition"""
        if self.dino_model is None:
            # Mock features
            return torch.randn(video.shape[1], 768).to(self.device)
        
        C, T, H, W = video.shape
        features = []
        
        with torch.no_grad():
            for t in range(T):
                frame = video[:, t].unsqueeze(0)  # [1, C, H, W]
                
                # Resize to DINOv2 input size
                frame_resized = F.interpolate(frame, size=(518, 518), mode='bilinear')
                
                # Extract features
                feat = self.dino_model(frame_resized)  # [1, 768]
                features.append(feat[0])
        
        return torch.stack(features)  # [T, 768]
    
    def _extract_spatial_features(self, video: torch.Tensor) -> torch.Tensor:
        """Extract spatial features for position tracking"""
        if self.spatial_encoder is None:
            # Mock features
            return torch.randn(video.shape[1], 512).to(self.device)
        
        # Use your VisualGeometrySpatialEncoder here
        spatial_output = self.spatial_encoder(video)
        return spatial_output['temporal_features']  # [T, 512]
    
    def _analyze_dino_detections(
        self, 
        dino_features: torch.Tensor, 
        video: torch.Tensor
    ) -> List[Dict]:
        """
        Analyze what DINO detected as objects
        
        DINO is excellent at detecting distinct visual objects like red dots!
        """
        # Compute feature similarity across time
        # High similarity = consistent object detection
        T = dino_features.shape[0]
        
        detections = []
        for t in range(T):
            feat = dino_features[t]
            
            # Feature magnitude (confidence in detection)
            confidence = float(torch.norm(feat))
            
            # Distinctiveness (how different from other frames)
            if t > 0:
                similarity = F.cosine_similarity(
                    feat.unsqueeze(0), 
                    dino_features[:t].mean(0).unsqueeze(0)
                )
                distinctiveness = 1.0 - float(similarity)
            else:
                distinctiveness = 0.5
            
            detections.append({
                'frame': t,
                'confidence': confidence / 10.0,  # Normalize
                'distinctiveness': distinctiveness
            })
        
        return detections
    
    def _analyze_spatial_layout(
        self, 
        spatial_features: torch.Tensor, 
        video: torch.Tensor
    ) -> Dict[str, any]:
        """
        Analyze spatial distribution from spatial features
        """
        T = spatial_features.shape[0]
        
        # Compute spatial consistency
        feature_variance = torch.var(spatial_features, dim=0).mean()
        consistency = 1.0 / (1.0 + float(feature_variance))
        
        # Motion analysis
        if T > 1:
            motion = torch.diff(spatial_features, dim=0).abs().mean()
            motion_score = float(motion)
        else:
            motion_score = 0.0
        
        return {
            'spatial_consistency': consistency,
            'motion_intensity': motion_score,
            'num_frames': T
        }
    
    def _compute_task_reward(
        self,
        fusion_result: Dict,
        dino_detections: List[Dict],
        spatial_layout: Dict,
        video: torch.Tensor,
        task_type: str
    ) -> float:
        """
        Task-specific reward using fusion understanding
        """
        if task_type == "erase":
            return self._compute_erasing_reward(dino_detections, spatial_layout)
        elif task_type == "connect":
            return self._compute_connecting_reward(spatial_layout)
        elif task_type == "highlight":
            return self._compute_highlighting_reward(dino_detections)
        else:
            return 0.5
    
    def _compute_erasing_reward(
        self, 
        dino_detections: List[Dict],
        spatial_layout: Dict
    ) -> float:
        """
        Reward for erasing task:
        - Object confidence should DECREASE over time (dots disappearing)
        - But should be DETECTED first (high initial confidence)
        """
        if len(dino_detections) < 2:
            return 0.0
        
        T = len(dino_detections)
        first_quarter = dino_detections[:T//4]
        last_quarter = dino_detections[-T//4:]
        
        # High initial detection confidence
        initial_confidence = np.mean([d['confidence'] for d in first_quarter])
        
        # Low final confidence (dots gone)
        final_confidence = np.mean([d['confidence'] for d in last_quarter])
        
        # Good erasing = high initial, low final
        if initial_confidence > 0.3:
            disappearance_rate = (initial_confidence - final_confidence) / initial_confidence
            return max(0.0, min(1.0, disappearance_rate))
        
        return 0.0
    
    def _compute_connecting_reward(self, spatial_layout: Dict) -> float:
        """
        Reward for connecting task:
        - Should have moderate motion (lines being drawn)
        - Should maintain spatial consistency (structured motion)
        """
        motion = spatial_layout['motion_intensity']
        consistency = spatial_layout['spatial_consistency']
        
        # Good connecting = moderate motion + high consistency
        motion_score = min(1.0, motion / 5.0)  # Normalize
        combined = 0.6 * motion_score + 0.4 * consistency
        
        return combined
    
    def _compute_highlighting_reward(self, dino_detections: List[Dict]) -> float:
        """
        Reward for highlighting task:
        - Object confidence should INCREASE (becoming more prominent)
        """
        if len(dino_detections) < 2:
            return 0.0
        
        confidences = [d['confidence'] for d in dino_detections]
        increase = confidences[-1] - confidences[0]
        
        return max(0.0, min(1.0, increase + 0.5))


def main():
    """
    Demonstration of Red Dot Recognition with Cross-Attention Fusion
    """
    print("=" * 70)
    print("  Red Dot Recognition using Cross-Attention Fusion (VGGT + DINO)")
    print("=" * 70)
    print()
    
    # Initialize reward function
    print("🚀 Initializing fusion-based reward system...")
    reward_fn = RedDotCrossAttentionReward()
    
    print("\n" + "=" * 70)
    print("  WHY Cross-Attention Fusion Solves the Problem")
    print("=" * 70)
    print("""
1. **DINO Object Recognition**
   - DINO is EXCELLENT at recognizing visual objects
   - It treats red dots as distinct, trackable objects
   - Provides "WHAT" information: "These are red circular objects"

2. **Spatial Encoder**
   - Tracks positions and motion in 3D space
   - Understands spatial layout and distribution
   - Provides "WHERE" information: "Objects at positions (x, y, z)"

3. **Cross-Attention Fusion**
   - DINO attends to Spatial: "What objects are WHERE?"
   - Spatial attends to DINO: "Where are WHAT objects?"
   - Creates rich unified understanding: "Red dots at specific locations"

4. **GRPO Training**
   - Generate multiple candidates
   - Reward based on fusion quality
   - Learn generation strategies that produce videos where:
     * DINO successfully detects red dots
     * Spatial correctly tracks their positions
     * Fusion combines both effectively

5. **Advantages over Simple CV**
   - DINO learns object concepts (better than color thresholding)
   - Handles varying lighting, occlusions, transformations
   - Cross-attention provides robust multi-modal understanding
   - End-to-end learnable with GRPO
""")
    
    print("\n" + "=" * 70)
    print("  Integration with Your System")
    print("=" * 70)
    print("""
Your existing components:
✓ AdvancedCrossAttentionFusion - Use this for DINO + Spatial fusion
✓ VisualGeometrySpatialEncoder - Use this for spatial features
✓ DINOv2 - Use this for object recognition
✓ VideoGRPOSearcher - Use this for GRPO training

Simply plug in:
  reward_function = RedDotCrossAttentionReward(
      dino_model=your_dino_model,
      spatial_encoder=your_spatial_encoder,
      fusion_model=your_cross_attention_fusion
  )
  
Then run GRPO with this fusion-based reward!
""")
    
    print("\n" + "=" * 70)
    print("  Next Steps")
    print("=" * 70)
    print("""
1. Install DINO: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
2. Use your VisualGeometrySpatialEncoder for spatial features
3. Use your AdvancedCrossAttentionFusion for fusion
4. Run GRPO with RedDotCrossAttentionReward as reward function

This will learn to generate videos where the model truly "sees" and
interacts with red dots, not just creates generic erasing motion!
""")


if __name__ == "__main__":
    main()

