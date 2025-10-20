#!/usr/bin/env python3
"""
Gemini VLM Verifier for Self-Improving GRPO Video Generation
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import base64
import io
from PIL import Image
import json
import time
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    print("Gemini not available. Install with: pip install google-generativeai")
    GEMINI_AVAILABLE = False

@dataclass
class VerificationResult:
    """Result from Gemini VLM verification"""
    overall_score: float
    reasoning_analysis: Dict[str, float]
    improvement_suggestions: List[str]
    detailed_feedback: str
    confidence: float
    verification_time: float

class GeminiVLMVerifier:
    """
    Gemini Vision-Language Model as intelligent verifier for video quality and reasoning
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro-vision-latest"):
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI not available")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.verification_history = []
        
    def verify_video_reasoning(
        self,
        video_frames: torch.Tensor,
        prompt: str,
        reasoning_focus: List[str],
        complexity_level: int = 3
    ) -> VerificationResult:
        """
        Use Gemini VLM to verify and score video reasoning quality
        """
        start_time = time.time()
        
        # Convert video frames to format Gemini can process
        frame_images = self._prepare_frames_for_gemini(video_frames)
        
        # Create comprehensive verification prompt
        verification_prompt = self._create_verification_prompt(
            prompt, reasoning_focus, complexity_level
        )
        
        # Get Gemini's analysis
        gemini_response = self._query_gemini_vlm(frame_images, verification_prompt)
        
        # Parse Gemini's response into structured feedback
        verification_result = self._parse_gemini_response(
            gemini_response, reasoning_focus, time.time() - start_time
        )
        
        # Store for self-improvement learning
        self.verification_history.append({
            'prompt': prompt,
            'reasoning_focus': reasoning_focus,
            'complexity_level': complexity_level,
            'result': verification_result,
            'timestamp': time.time()
        })
        
        return verification_result
    
    def _prepare_frames_for_gemini(self, video_frames: torch.Tensor, max_frames: int = 8) -> List[Image.Image]:
        """
        Convert video tensor to PIL Images for Gemini processing
        """
        # video_frames: [C, T, H, W] or [B, C, T, H, W]
        if len(video_frames.shape) == 5:
            video_frames = video_frames[0]  # Take first batch item
        
        C, T, H, W = video_frames.shape
        
        # Sample frames evenly across the video
        frame_indices = torch.linspace(0, T-1, min(max_frames, T)).long()
        sampled_frames = video_frames[:, frame_indices]  # [C, sampled_T, H, W]
        
        # Convert to PIL Images
        pil_images = []
        for t in range(sampled_frames.shape[1]):
            frame = sampled_frames[:, t]  # [C, H, W]
            
            # Normalize to [0, 255] and convert to uint8
            if frame.max() <= 1.0:
                frame = frame * 255
            frame = frame.clamp(0, 255).byte()
            
            # Convert CHW to HWC
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_np)
            pil_images.append(pil_image)
        
        return pil_images
    
    def _create_verification_prompt(
        self,
        original_prompt: str,
        reasoning_focus: List[str],
        complexity_level: int
    ) -> str:
        """
        Create comprehensive prompt for Gemini VLM verification
        """
        base_prompt = f"""
You are an expert video analysis AI tasked with evaluating the reasoning and thinking abilities demonstrated in this video sequence.

ORIGINAL PROMPT: "{original_prompt}"

REASONING FOCUS AREAS: {', '.join(reasoning_focus)}
COMPLEXITY LEVEL: {complexity_level}/5

Please analyze this video sequence and provide detailed feedback on:

1. REASONING QUALITY ASSESSMENT (Score 0-10 for each):
   - Causal Reasoning: How well does the video show cause-effect relationships?
   - Temporal Reasoning: Does the sequence follow logical time progression?
   - Spatial Reasoning: Are 3D relationships and physics realistic?
   - Logical Reasoning: Is there clear step-by-step problem solving?
   - Creative Reasoning: Are solutions novel yet appropriate?
   - Social Reasoning: Are character interactions intelligent?
   - Scientific Reasoning: Is there systematic methodology shown?

2. OVERALL COHERENCE (Score 0-10):
   - Does the video tell a coherent story?
   - Are the reasoning elements well-integrated?
   - Is the complexity appropriate for the level requested?

3. SPECIFIC IMPROVEMENTS NEEDED:
   - What reasoning aspects could be enhanced?
   - Which frames show weak reasoning?
   - How could the logical flow be improved?

4. SELF-IMPROVEMENT SUGGESTIONS:
   - What should the AI focus on in the next iteration?
   - Which reasoning types need more emphasis?
   - How can the generation process be improved?

Please provide your response in this JSON format:
{{
    "reasoning_scores": {{
        "causal_reasoning": score,
        "temporal_reasoning": score,
        "spatial_reasoning": score,
        "logical_reasoning": score,
        "creative_reasoning": score,
        "social_reasoning": score,
        "scientific_reasoning": score
    }},
    "overall_coherence": score,
    "overall_score": average_score,
    "improvement_suggestions": ["suggestion1", "suggestion2", ...],
    "detailed_analysis": "detailed_text_analysis",
    "confidence": confidence_score,
    "next_iteration_focus": ["focus_area1", "focus_area2", ...]
}}
"""
        
        return base_prompt.strip()
    
    def _query_gemini_vlm(self, frame_images: List[Image.Image], prompt: str) -> str:
        """
        Query Gemini VLM with video frames and verification prompt
        """
        try:
            # Prepare content for Gemini
            content = [prompt]
            
            # Add frame images
            for i, image in enumerate(frame_images):
                content.append(f"Frame {i+1}:")
                content.append(image)
            
            # Query Gemini
            response = self.model.generate_content(content)
            return response.text
            
        except Exception as e:
            print(f"Error querying Gemini: {e}")
            return self._fallback_analysis()
    
    def _parse_gemini_response(
        self,
        response: str,
        reasoning_focus: List[str],
        verification_time: float
    ) -> VerificationResult:
        """
        Parse Gemini's response into structured verification result
        """
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                return VerificationResult(
                    overall_score=parsed.get('overall_score', 5.0) / 10.0,  # Normalize to [0,1]
                    reasoning_analysis=parsed.get('reasoning_scores', {}),
                    improvement_suggestions=parsed.get('improvement_suggestions', []),
                    detailed_feedback=parsed.get('detailed_analysis', ''),
                    confidence=parsed.get('confidence', 0.7),
                    verification_time=verification_time
                )
            else:
                # Fallback parsing
                return self._fallback_parse(response, verification_time)
                
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            return self._fallback_parse(response, verification_time)
    
    def _fallback_analysis(self) -> str:
        """Fallback analysis when Gemini is unavailable"""
        return json.dumps({
            "reasoning_scores": {
                "causal_reasoning": 5.0,
                "temporal_reasoning": 5.0,
                "spatial_reasoning": 5.0,
                "logical_reasoning": 5.0,
                "creative_reasoning": 5.0,
                "social_reasoning": 5.0,
                "scientific_reasoning": 5.0
            },
            "overall_coherence": 5.0,
            "overall_score": 5.0,
            "improvement_suggestions": ["Gemini VLM unavailable - using fallback analysis"],
            "detailed_analysis": "Fallback analysis - Gemini VLM not accessible",
            "confidence": 0.3,
            "next_iteration_focus": ["general_improvement"]
        })
    
    def _fallback_parse(self, response: str, verification_time: float) -> VerificationResult:
        """Fallback parsing when JSON parsing fails"""
        return VerificationResult(
            overall_score=0.5,
            reasoning_analysis={},
            improvement_suggestions=["Could not parse Gemini response"],
            detailed_feedback=response,
            confidence=0.3,
            verification_time=verification_time
        )

class SelfImprovingGRPOSystem:
    """
    Self-improving GRPO system using Gemini VLM as verifier
    """
    
    def __init__(self, video_generator, gemini_api_key: str):
        self.video_generator = video_generator
        self.gemini_verifier = GeminiVLMVerifier(gemini_api_key)
        self.improvement_history = []
        self.learned_strategies = {}
        
    def self_improving_generation(
        self,
        prompt: str,
        reasoning_focus: List[str],
        num_self_improvement_cycles: int = 5,
        videos_per_cycle: int = 4
    ) -> Dict[str, Any]:
        """
        Main self-improvement loop using Gemini VLM feedback
        """
        print(f"🔄 SELF-IMPROVING GRPO WITH GEMINI VLM VERIFIER")
        print(f"Prompt: '{prompt}'")
        print(f"Reasoning Focus: {reasoning_focus}")
        print(f"Self-Improvement Cycles: {num_self_improvement_cycles}")
        print("=" * 70)
        
        improvement_results = {}
        current_generation_config = self._initialize_generation_config()
        
        for cycle in range(num_self_improvement_cycles):
            print(f"\n🔄 Self-Improvement Cycle {cycle + 1}/{num_self_improvement_cycles}")
            
            # Generate video candidates
            candidates = self._generate_candidates_with_learned_strategies(
                prompt, current_generation_config, videos_per_cycle
            )
            
            # Get Gemini VLM verification for each candidate
            verifications = []
            for i, candidate in enumerate(candidates):
                print(f"  🔍 Gemini verifying candidate {i+1}...")
                
                verification = self.gemini_verifier.verify_video_reasoning(
                    video_frames=candidate['video'],
                    prompt=prompt,
                    reasoning_focus=reasoning_focus,
                    complexity_level=cycle + 1  # Progressive complexity
                )
                
                verifications.append(verification)
                print(f"    Score: {verification.overall_score:.3f}, "
                      f"Confidence: {verification.confidence:.3f}")
            
            # Learn from Gemini's feedback
            learned_improvements = self._learn_from_gemini_feedback(
                candidates, verifications, reasoning_focus
            )
            
            # Update generation strategy based on learning
            current_generation_config = self._update_generation_strategy(
                current_generation_config, learned_improvements
            )
            
            # Store cycle results
            improvement_results[f"cycle_{cycle + 1}"] = {
                'candidates': candidates,
                'verifications': verifications,
                'learned_improvements': learned_improvements,
                'updated_config': current_generation_config.copy(),
                'best_score': max(v.overall_score for v in verifications),
                'avg_score': np.mean([v.overall_score for v in verifications])
            }
            
            print(f"  📈 Best score this cycle: {improvement_results[f'cycle_{cycle + 1}']['best_score']:.3f}")
            print(f"  📊 Average score: {improvement_results[f'cycle_{cycle + 1}']['avg_score']:.3f}")
        
        # Analyze overall improvement
        improvement_analysis = self._analyze_self_improvement(improvement_results)
        
        return {
            'cycle_results': improvement_results,
            'improvement_analysis': improvement_analysis,
            'final_generation_config': current_generation_config,
            'learned_strategies': self.learned_strategies
        }
    
    def _initialize_generation_config(self) -> Dict[str, Any]:
        """Initialize generation configuration"""
        return {
            'height': 512,
            'width': 768,
            'num_frames': 160,
            'guidance_scale': 7.5,
            'num_inference_steps': 40,
            'base_seed': 2025,
            'reasoning_emphasis': 1.0,  # How much to emphasize reasoning
            'creativity_balance': 0.5,   # Balance between creativity and logic
            'temporal_focus': 0.7       # Focus on temporal coherence
        }
    
    def _generate_candidates_with_learned_strategies(
        self,
        prompt: str,
        config: Dict[str, Any],
        num_candidates: int
    ) -> List[Dict[str, Any]]:
        """
        Generate candidates incorporating learned strategies from previous cycles
        """
        candidates = []
        
        for i in range(num_candidates):
            # Apply learned strategies
            adapted_config = self._apply_learned_strategies(config, i)
            
            # Generate enhanced prompt based on learned improvements
            enhanced_prompt = self._enhance_prompt_with_learning(prompt)
            
            # Generate video
            video = self.video_generator.generate_single(
                prompt=enhanced_prompt,
                **adapted_config
            )
            
            candidate = {
                'video': video,
                'original_prompt': prompt,
                'enhanced_prompt': enhanced_prompt,
                'config_used': adapted_config,
                'candidate_id': i,
                'applied_strategies': list(self.learned_strategies.keys())
            }
            
            candidates.append(candidate)
        
        return candidates
    
    def _apply_learned_strategies(self, base_config: Dict[str, Any], candidate_id: int) -> Dict[str, Any]:
        """
        Apply learned strategies from previous Gemini feedback
        """
        config = base_config.copy()
        
        # Apply learned parameter adjustments
        for strategy_name, strategy_data in self.learned_strategies.items():
            if strategy_name == "guidance_scale_adjustment":
                config['guidance_scale'] += strategy_data.get('adjustment', 0.0)
            elif strategy_name == "temporal_emphasis":
                config['num_frames'] = int(config['num_frames'] * strategy_data.get('multiplier', 1.0))
            elif strategy_name == "reasoning_focus":
                config['reasoning_emphasis'] *= strategy_data.get('emphasis_factor', 1.0)
        
        # Add variation for exploration
        variation_factor = 0.1 * (candidate_id / 4)  # Small variations
        config['guidance_scale'] *= (1.0 + variation_factor)
        config['seed'] = base_config['base_seed'] + candidate_id * 100
        
        return config
    
    def _enhance_prompt_with_learning(self, original_prompt: str) -> str:
        """
        Enhance prompt based on learned improvements from Gemini feedback
        """
        enhancements = []
        
        # Apply learned prompt enhancements
        for strategy_name, strategy_data in self.learned_strategies.items():
            if strategy_name == "causal_emphasis" and strategy_data.get('effective', False):
                enhancements.append("with clear cause and effect relationships")
            elif strategy_name == "temporal_logic" and strategy_data.get('effective', False):
                enhancements.append("showing logical progression over time")
            elif strategy_name == "problem_solving" and strategy_data.get('effective', False):
                enhancements.append("demonstrating step-by-step problem solving")
        
        if enhancements:
            return f"{original_prompt}, {', '.join(enhancements)}"
        else:
            return original_prompt
    
    def _learn_from_gemini_feedback(
        self,
        candidates: List[Dict[str, Any]],
        verifications: List[VerificationResult],
        reasoning_focus: List[str]
    ) -> Dict[str, Any]:
        """
        Extract learning insights from Gemini's feedback
        """
        print("  🧠 Learning from Gemini feedback...")
        
        # Analyze what worked best
        best_verification = max(verifications, key=lambda v: v.overall_score)
        best_candidate_id = verifications.index(best_verification)
        best_candidate = candidates[best_candidate_id]
        
        # Extract successful strategies
        successful_strategies = {
            'best_config': best_candidate['config_used'],
            'best_prompt_enhancement': best_candidate['enhanced_prompt'],
            'reasoning_strengths': [],
            'reasoning_weaknesses': [],
            'improvement_priorities': best_verification.improvement_suggestions
        }
        
        # Analyze reasoning performance
        for reasoning_type, score in best_verification.reasoning_analysis.items():
            if score >= 7.0:  # Strong performance
                successful_strategies['reasoning_strengths'].append(reasoning_type)
            elif score <= 4.0:  # Weak performance
                successful_strategies['reasoning_weaknesses'].append(reasoning_type)
        
        # Update learned strategies
        self._update_learned_strategies(successful_strategies, best_verification)
        
        print(f"    ✅ Strengths: {successful_strategies['reasoning_strengths']}")
        print(f"    ⚠️  Weaknesses: {successful_strategies['reasoning_weaknesses']}")
        print(f"    🎯 Priorities: {successful_strategies['improvement_priorities'][:2]}")
        
        return successful_strategies
    
    def _update_learned_strategies(
        self,
        successful_strategies: Dict[str, Any],
        verification: VerificationResult
    ):
        """
        Update the learned strategies database
        """
        # Learn from successful configurations
        best_config = successful_strategies['best_config']
        
        if 'guidance_scale_adjustment' not in self.learned_strategies:
            self.learned_strategies['guidance_scale_adjustment'] = {'adjustment': 0.0, 'confidence': 0.0}
        
        # Update guidance scale learning
        current_guidance = best_config.get('guidance_scale', 7.5)
        base_guidance = 7.5
        adjustment = current_guidance - base_guidance
        
        # Weighted update based on verification confidence
        old_adjustment = self.learned_strategies['guidance_scale_adjustment']['adjustment']
        old_confidence = self.learned_strategies['guidance_scale_adjustment']['confidence']
        
        new_confidence = (old_confidence + verification.confidence) / 2
        new_adjustment = (old_adjustment * old_confidence + adjustment * verification.confidence) / (old_confidence + verification.confidence)
        
        self.learned_strategies['guidance_scale_adjustment'] = {
            'adjustment': new_adjustment,
            'confidence': new_confidence
        }
        
        # Learn from reasoning strengths
        for strength in successful_strategies['reasoning_strengths']:
            strategy_name = f"{strength}_emphasis"
            if strategy_name not in self.learned_strategies:
                self.learned_strategies[strategy_name] = {'effective': True, 'confidence': verification.confidence}
            else:
                # Increase confidence in this strategy
                self.learned_strategies[strategy_name]['confidence'] = min(
                    self.learned_strategies[strategy_name]['confidence'] + 0.1, 1.0
                )
    
    def _update_generation_strategy(
        self,
        current_config: Dict[str, Any],
        learned_improvements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update generation strategy based on learned improvements
        """
        updated_config = current_config.copy()
        
        # Apply learned parameter adjustments
        if 'best_config' in learned_improvements:
            best_config = learned_improvements['best_config']
            
            # Gradually move toward successful parameters
            learning_rate = 0.3
            for param in ['guidance_scale', 'reasoning_emphasis', 'creativity_balance']:
                if param in best_config:
                    current_val = updated_config.get(param, 1.0)
                    best_val = best_config.get(param, 1.0)
                    updated_config[param] = current_val + learning_rate * (best_val - current_val)
        
        # Adjust based on reasoning weaknesses
        if 'reasoning_weaknesses' in learned_improvements:
            for weakness in learned_improvements['reasoning_weaknesses']:
                if 'temporal' in weakness:
                    updated_config['temporal_focus'] = min(updated_config.get('temporal_focus', 0.7) + 0.1, 1.0)
                elif 'causal' in weakness:
                    updated_config['reasoning_emphasis'] = min(updated_config.get('reasoning_emphasis', 1.0) + 0.2, 2.0)
        
        return updated_config
    
    def _analyze_self_improvement(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the overall self-improvement progress
        """
        cycles = list(results.keys())
        scores = [results[cycle]['best_score'] for cycle in cycles]
        avg_scores = [results[cycle]['avg_score'] for cycle in cycles]
        
        improvement_analysis = {
            'total_improvement': scores[-1] - scores[0],
            'improvement_rate': np.polyfit(range(len(scores)), scores, 1)[0],  # Linear trend
            'consistency_improvement': avg_scores[-1] - avg_scores[0],
            'learning_effectiveness': len(self.learned_strategies),
            'convergence_analysis': self._analyze_convergence(scores),
            'reasoning_evolution': self._analyze_reasoning_evolution(results)
        }
        
        print(f"\n📈 SELF-IMPROVEMENT ANALYSIS:")
        print(f"  Total Improvement: {improvement_analysis['total_improvement']:+.3f}")
        print(f"  Improvement Rate: {improvement_analysis['improvement_rate']:+.3f} per cycle")
        print(f"  Learned Strategies: {improvement_analysis['learning_effectiveness']}")
        print(f"  Convergence: {improvement_analysis['convergence_analysis']}")
        
        return improvement_analysis
    
    def _analyze_convergence(self, scores: List[float]) -> str:
        """Analyze if the system is converging to better solutions"""
        if len(scores) < 3:
            return "Insufficient data"
        
        recent_improvement = scores[-1] - scores[-3]
        if recent_improvement > 0.05:
            return "Strong convergence"
        elif recent_improvement > 0.01:
            return "Moderate convergence"
        elif recent_improvement > -0.01:
            return "Stable (converged)"
        else:
            return "Diverging (needs adjustment)"
    
    def _analyze_reasoning_evolution(self, results: Dict[str, Any]) -> Dict[str, List[float]]:
        """Analyze how different reasoning types evolved over cycles"""
        reasoning_evolution = {}
        
        for cycle_name, cycle_data in results.items():
            verifications = cycle_data['verifications']
            best_verification = max(verifications, key=lambda v: v.overall_score)
            
            for reasoning_type, score in best_verification.reasoning_analysis.items():
                if reasoning_type not in reasoning_evolution:
                    reasoning_evolution[reasoning_type] = []
                reasoning_evolution[reasoning_type].append(score / 10.0)  # Normalize
        
        return reasoning_evolution

# Example usage and integration
def example_self_improving_grpo():
    """
    Example of self-improving GRPO with Gemini VLM verifier
    """
    print("🚀 EXAMPLE: SELF-IMPROVING GRPO WITH GEMINI VLM")
    print("=" * 60)
    
    # Mock components (replace with real implementations)
    class MockVideoGenerator:
        def generate_single(self, prompt, **kwargs):
            return torch.randn(3, 16, 224, 224)
    
    # Initialize system (you'll need a real Gemini API key)
    generator = MockVideoGenerator()
    # system = SelfImprovingGRPOSystem(generator, "your-gemini-api-key")
    
    # Example usage:
    print("Usage example:")
    print("""
    # Initialize with real components
    system = SelfImprovingGRPOSystem(your_video_generator, gemini_api_key)
    
    # Run self-improvement
    results = system.self_improving_generation(
        prompt="A scientist discovers a new phenomenon through experimentation",
        reasoning_focus=["causal_reasoning", "scientific_reasoning", "logical_reasoning"],
        num_self_improvement_cycles=5,
        videos_per_cycle=4
    )
    
    # The system will:
    # 1. Generate video candidates
    # 2. Get Gemini VLM feedback on reasoning quality
    # 3. Learn from feedback and improve generation strategy
    # 4. Repeat with enhanced approach
    # 5. Continuously self-improve based on VLM verification
    """)

if __name__ == "__main__":
    example_self_improving_grpo()


