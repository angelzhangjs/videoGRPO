#!/usr/bin/env python3
"""
Hybrid GRPO System: Combining Gemini VLM Verifier + Reward Functions
Best of both worlds for video generation optimization
"""

import torch
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from gemini_vlm_verifier import GeminiVLMVerifier, VerificationResult
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    @dataclass
    class VerificationResult:
        overall_score: float = 0.5

class RewardType(Enum):
    """Types of rewards in the hybrid system"""
    FAST_REWARD = "fast_reward"        # Quick, computational rewards
    VLM_REWARD = "vlm_reward"          # Intelligent VLM verification
    HYBRID_REWARD = "hybrid_reward"    # Combination of both

@dataclass
class HybridRewardResult:
    """Result from hybrid reward computation"""
    fast_reward: float
    vlm_reward: float
    hybrid_reward: float
    reward_breakdown: Dict[str, float]
    vlm_feedback: Optional[VerificationResult]
    computation_time: Dict[str, float]
    confidence: float

class HybridGRPOSystem:
    """
    Hybrid GRPO system combining fast reward functions with intelligent VLM verification
    """
    
    def __init__(
        self,
        video_generator,
        fast_reward_function: Callable,
        gemini_api_key: str = None,
        hybrid_config: Dict[str, Any] = None
    ):
        self.video_generator = video_generator
        self.fast_reward_function = fast_reward_function
        
        # Initialize VLM verifier if available
        self.vlm_verifier = None
        if GEMINI_AVAILABLE and gemini_api_key:
            self.vlm_verifier = GeminiVLMVerifier(gemini_api_key)
        
        # Hybrid configuration
        self.hybrid_config = hybrid_config or self._default_hybrid_config()
        
        # Learning history
        self.reward_correlation_history = []
        self.learned_reward_weights = self._initialize_reward_weights()
    
    def _default_hybrid_config(self) -> Dict[str, Any]:
        """Default configuration for hybrid system"""
        return {
            'fast_reward_weight': 0.4,      # Weight for computational rewards
            'vlm_reward_weight': 0.6,       # Weight for VLM verification
            'vlm_frequency': 0.5,           # Fraction of candidates to verify with VLM (cost control)
            'adaptive_weighting': True,     # Adapt weights based on correlation
            'correlation_threshold': 0.7,   # When fast rewards are reliable
            'vlm_confidence_threshold': 0.8, # Minimum VLM confidence to trust
            'fast_reward_components': [
                'motion_quality', 'temporal_consistency', 'visual_complexity'
            ],
            'vlm_reasoning_focus': [
                'causal_reasoning', 'temporal_reasoning', 'logical_reasoning'
            ]
        }
    
    def _initialize_reward_weights(self) -> Dict[str, float]:
        """Initialize adaptive reward weights"""
        return {
            'fast_reward_reliability': 0.5,
            'vlm_reward_reliability': 0.8,
            'correlation_strength': 0.0,
            'adaptation_rate': 0.1
        }
    
    def compute_hybrid_reward(
        self,
        video: torch.Tensor,
        prompt: str,
        use_vlm: bool = True,
        reasoning_focus: List[str] = None
    ) -> HybridRewardResult:
        """
        Compute hybrid reward combining fast rewards and VLM verification
        """
        import time
        
        computation_times = {}
        
        # 1. Compute fast reward (always computed - low cost)
        start_time = time.time()
        fast_reward_result = self.fast_reward_function(video, prompt)
        fast_reward = fast_reward_result['reward']
        computation_times['fast_reward'] = time.time() - start_time
        
        # 2. Compute VLM reward (selectively - high cost)
        vlm_reward = 0.5  # Default neutral
        vlm_feedback = None
        
        if use_vlm and self.vlm_verifier:
            start_time = time.time()
            vlm_feedback = self.vlm_verifier.verify_video_reasoning(
                video_frames=video,
                prompt=prompt,
                reasoning_focus=reasoning_focus or self.hybrid_config['vlm_reasoning_focus'],
                complexity_level=3
            )
            vlm_reward = vlm_feedback.overall_score
            computation_times['vlm_reward'] = time.time() - start_time
        else:
            computation_times['vlm_reward'] = 0.0
        
        # 3. Compute hybrid reward with adaptive weighting
        hybrid_reward, reward_breakdown = self._compute_adaptive_hybrid_reward(
            fast_reward, vlm_reward, vlm_feedback
        )
        
        # 4. Update correlation learning
        if vlm_feedback:
            self._update_reward_correlation(fast_reward, vlm_reward, vlm_feedback.confidence)
        
        return HybridRewardResult(
            fast_reward=fast_reward,
            vlm_reward=vlm_reward,
            hybrid_reward=hybrid_reward,
            reward_breakdown=reward_breakdown,
            vlm_feedback=vlm_feedback,
            computation_time=computation_times,
            confidence=vlm_feedback.confidence if vlm_feedback else 0.5
        )
    
    def _compute_adaptive_hybrid_reward(
        self,
        fast_reward: float,
        vlm_reward: float,
        vlm_feedback: Optional[VerificationResult]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute hybrid reward with adaptive weighting based on learned correlations
        """
        # Base weights from configuration
        fast_weight = self.hybrid_config['fast_reward_weight']
        vlm_weight = self.hybrid_config['vlm_reward_weight']
        
        # Adaptive weighting based on learned correlations
        if self.hybrid_config['adaptive_weighting']:
            correlation_strength = self.learned_reward_weights['correlation_strength']
            
            # If fast rewards correlate well with VLM, increase fast weight
            if correlation_strength > self.hybrid_config['correlation_threshold']:
                fast_weight += 0.2 * correlation_strength
                vlm_weight = 1.0 - fast_weight
            
            # If VLM confidence is low, rely more on fast rewards
            if vlm_feedback and vlm_feedback.confidence < self.hybrid_config['vlm_confidence_threshold']:
                fast_weight += 0.3 * (1.0 - vlm_feedback.confidence)
                vlm_weight = 1.0 - fast_weight
        
        # Ensure weights sum to 1
        total_weight = fast_weight + vlm_weight
        fast_weight /= total_weight
        vlm_weight /= total_weight
        
        # Compute hybrid reward
        hybrid_reward = fast_weight * fast_reward + vlm_weight * vlm_reward
        
        reward_breakdown = {
            'fast_reward_contribution': fast_weight * fast_reward,
            'vlm_reward_contribution': vlm_weight * vlm_reward,
            'fast_weight': fast_weight,
            'vlm_weight': vlm_weight,
            'correlation_strength': self.learned_reward_weights['correlation_strength']
        }
        
        return hybrid_reward, reward_breakdown
    
    def _update_reward_correlation(self, fast_reward: float, vlm_reward: float, vlm_confidence: float):
        """
        Learn correlation between fast rewards and VLM rewards for adaptive weighting
        """
        # Only learn from high-confidence VLM feedback
        if vlm_confidence >= self.hybrid_config['vlm_confidence_threshold']:
            correlation = abs(fast_reward - vlm_reward)  # Lower = better correlation
            correlation_score = 1.0 - min(correlation, 1.0)  # Convert to [0,1]
            
            # Update learned correlation with exponential moving average
            adaptation_rate = self.learned_reward_weights['adaptation_rate']
            current_correlation = self.learned_reward_weights['correlation_strength']
            
            self.learned_reward_weights['correlation_strength'] = (
                (1 - adaptation_rate) * current_correlation + 
                adaptation_rate * correlation_score
            )
            
            # Store for analysis
            self.reward_correlation_history.append({
                'fast_reward': fast_reward,
                'vlm_reward': vlm_reward,
                'vlm_confidence': vlm_confidence,
                'correlation_score': correlation_score
            })
    
    def hybrid_grpo_search(
        self,
        prompts: List[str],
        gemini_api_key: str = None,
        num_videos_per_prompt: int = 4,
        num_iterations: int = 5,
        generation_config: Dict[str, Any] = None,
        device: torch.device = torch.device("cuda")
    ) -> Dict[str, Any]:
        """
        Main hybrid GRPO search combining fast rewards and VLM verification
        """
        print("🔥 HYBRID GRPO: FAST REWARDS + GEMINI VLM VERIFIER")
        print("=" * 60)
        
        if generation_config is None:
            generation_config = {
                'height': 512, 'width': 768, 'num_frames': 160,
                'guidance_scale': 7.5, 'num_inference_steps': 40, 'base_seed': 2025
            }
        
        all_results = {}
        learned_strategies = {}
        
        for iteration in range(num_iterations):
            print(f"\n=== Hybrid GRPO Iteration {iteration + 1}/{num_iterations} ===")
            
            # Generate video candidates
            episodes = self._generate_hybrid_candidates(
                prompts, generation_config, num_videos_per_prompt, device
            )
            
            # Compute hybrid rewards for all candidates
            hybrid_results = self._compute_hybrid_rewards_batch(
                episodes, iteration, num_iterations
            )
            
            # Learn from hybrid feedback
            iteration_learning = self._learn_from_hybrid_feedback(hybrid_results)
            learned_strategies.update(iteration_learning)
            
            # Update generation strategy based on hybrid learning
            generation_config = self._update_hybrid_generation_strategy(
                generation_config, learned_strategies, hybrid_results
            )
            
            # Select best episodes using hybrid rewards
            best_episodes = self._select_best_hybrid_episodes(episodes, hybrid_results)
            
            # Store iteration results
            all_results[f"iteration_{iteration + 1}"] = {
                'episodes': episodes,
                'hybrid_results': hybrid_results,
                'best_episodes': best_episodes,
                'learned_strategies': iteration_learning,
                'generation_config': generation_config.copy(),
                'performance_metrics': self._compute_performance_metrics(hybrid_results)
            }
            
            # Print iteration summary
            self._print_iteration_summary(all_results[f"iteration_{iteration + 1}"])
        
        # Final analysis
        final_analysis = self._analyze_hybrid_performance(all_results)
        
        return {
            'iteration_results': all_results,
            'final_analysis': final_analysis,
            'learned_strategies': learned_strategies,
            'reward_correlation_history': self.reward_correlation_history
        }
    
    def _generate_hybrid_candidates(
        self,
        prompts: List[str],
        generation_config: Dict[str, Any],
        num_videos_per_prompt: int,
        device: torch.device
    ) -> List[Dict[str, Any]]:
        """
        Generate candidates optimized for hybrid evaluation
        """
        candidates = []
        
        for prompt_idx, prompt in enumerate(prompts):
            for video_idx in range(num_videos_per_prompt):
                # Create parameter variations
                varied_config = self._create_hybrid_variation(generation_config, video_idx)
                varied_config['seed'] = generation_config['base_seed'] + prompt_idx * 100 + video_idx
                
                # Generate video
                video = self.video_generator.generate_single(
                    prompt=prompt,
                    **varied_config
                )
                
                candidate = {
                    'prompt': prompt,
                    'video': video,
                    'generation_params': varied_config,
                    'prompt_idx': prompt_idx,
                    'video_idx': video_idx
                }
                
                candidates.append(candidate)
        
        return candidates
    
    def _compute_hybrid_rewards_batch(
        self,
        episodes: List[Dict[str, Any]],
        current_iteration: int,
        total_iterations: int
    ) -> List[HybridRewardResult]:
        """
        Compute hybrid rewards for a batch of episodes with intelligent VLM usage
        """
        hybrid_results = []
        
        # Determine VLM usage strategy
        vlm_usage_strategy = self._determine_vlm_usage_strategy(
            episodes, current_iteration, total_iterations
        )
        
        for i, episode in enumerate(episodes):
            print(f"  Computing hybrid reward {i+1}/{len(episodes)}")
            
            # Decide whether to use VLM for this candidate
            use_vlm = vlm_usage_strategy[i]
            
            # Compute hybrid reward
            hybrid_result = self.compute_hybrid_reward(
                video=episode['video'],
                prompt=episode['prompt'],
                use_vlm=use_vlm,
                reasoning_focus=self.hybrid_config['vlm_reasoning_focus']
            )
            
            hybrid_results.append(hybrid_result)
            
            print(f"    Fast: {hybrid_result.fast_reward:.3f}, "
                  f"VLM: {hybrid_result.vlm_reward:.3f}, "
                  f"Hybrid: {hybrid_result.hybrid_reward:.3f}")
        
        return hybrid_results
    
    def _determine_vlm_usage_strategy(
        self,
        episodes: List[Dict[str, Any]],
        current_iteration: int,
        total_iterations: int
    ) -> List[bool]:
        """
        Intelligent strategy for when to use expensive VLM verification
        """
        num_episodes = len(episodes)
        vlm_frequency = self.hybrid_config['vlm_frequency']
        
        # Strategy 1: Progressive VLM usage (more VLM in later iterations)
        iteration_progress = current_iteration / total_iterations
        adjusted_frequency = vlm_frequency * (0.5 + 0.5 * iteration_progress)
        
        # Strategy 2: Use VLM for most promising candidates (based on fast rewards)
        if current_iteration > 0:
            # Quick fast reward pre-screening
            fast_rewards = []
            for episode in episodes:
                quick_reward = self.fast_reward_function(episode['video'], episode['prompt'])
                fast_rewards.append(quick_reward['reward'])
            
            # Use VLM for top candidates + some random exploration
            sorted_indices = np.argsort(fast_rewards)[::-1]  # Descending order
            num_vlm_candidates = int(num_episodes * adjusted_frequency)
            
            # Top candidates + random sampling
            top_candidates = sorted_indices[:num_vlm_candidates//2]
            random_candidates = np.random.choice(
                sorted_indices[num_vlm_candidates//2:],
                size=num_vlm_candidates - len(top_candidates),
                replace=False
            )
            
            vlm_indices = set(list(top_candidates) + list(random_candidates))
        else:
            # First iteration: random sampling
            num_vlm_candidates = int(num_episodes * adjusted_frequency)
            vlm_indices = set(np.random.choice(num_episodes, size=num_vlm_candidates, replace=False))
        
        # Create usage strategy
        vlm_usage = [i in vlm_indices for i in range(num_episodes)]
        
        print(f"  VLM Usage: {sum(vlm_usage)}/{num_episodes} candidates "
              f"({sum(vlm_usage)/num_episodes:.1%})")
        
        return vlm_usage
    
    def _create_hybrid_variation(self, base_config: Dict[str, Any], video_idx: int) -> Dict[str, Any]:
        """
        Create parameter variations optimized for hybrid evaluation
        """
        config = base_config.copy()
        
        # Vary parameters for both fast and VLM rewards
        
        # Fast reward optimization: focus on computational metrics
        if video_idx % 4 == 0:
            config['guidance_scale'] *= 1.1  # Higher guidance for visual quality
            config['num_inference_steps'] += 10  # More steps for temporal consistency
        
        # VLM reward optimization: focus on reasoning
        elif video_idx % 4 == 1:
            config['num_frames'] = int(config.get('num_frames', 160) * 1.2)  # More frames for reasoning
            config['guidance_scale'] *= 0.9  # Lower guidance for creativity
        
        # Balanced optimization
        elif video_idx % 4 == 2:
            # Keep base parameters
            pass
        
        # Exploration
        else:
            config['guidance_scale'] += np.random.normal(0, 1.0)  # Random exploration
            config['guidance_scale'] = max(3.0, min(15.0, config['guidance_scale']))
        
        return config
    
    def _learn_from_hybrid_feedback(self, hybrid_results: List[HybridRewardResult]) -> Dict[str, Any]:
        """
        Learn from the combination of fast rewards and VLM feedback
        """
        learning = {}
        
        # Analyze correlation between fast and VLM rewards
        vlm_results = [hr for hr in hybrid_results if hr.vlm_feedback is not None]
        
        if len(vlm_results) >= 3:  # Need minimum samples for correlation analysis
            fast_scores = [hr.fast_reward for hr in vlm_results]
            vlm_scores = [hr.vlm_reward for hr in vlm_results]
            
            # Compute correlation
            correlation = np.corrcoef(fast_scores, vlm_scores)[0, 1]
            learning['reward_correlation'] = correlation
            
            print(f"  📊 Fast-VLM Correlation: {correlation:.3f}")
            
            # Learn when fast rewards are reliable
            if correlation > self.hybrid_config['correlation_threshold']:
                learning['fast_rewards_reliable'] = True
                learning['recommended_vlm_frequency'] = max(0.2, self.hybrid_config['vlm_frequency'] - 0.1)
            else:
                learning['fast_rewards_unreliable'] = True
                learning['recommended_vlm_frequency'] = min(0.8, self.hybrid_config['vlm_frequency'] + 0.1)
        
        # Learn from VLM-specific insights
        high_confidence_vlm = [hr for hr in vlm_results if hr.confidence > 0.8]
        if high_confidence_vlm:
            # Extract common improvement themes
            all_suggestions = []
            for hr in high_confidence_vlm:
                all_suggestions.extend(hr.vlm_feedback.improvement_suggestions)
            
            # Identify patterns
            if any('temporal' in suggestion.lower() for suggestion in all_suggestions):
                learning['temporal_focus_needed'] = True
            
            if any('causal' in suggestion.lower() for suggestion in all_suggestions):
                learning['causal_emphasis_needed'] = True
            
            if any('creative' in suggestion.lower() for suggestion in all_suggestions):
                learning['creativity_boost_needed'] = True
        
        return learning
    
    def _update_hybrid_generation_strategy(
        self,
        current_config: Dict[str, Any],
        learned_strategies: Dict[str, Any],
        hybrid_results: List[HybridRewardResult]
    ) -> Dict[str, Any]:
        """
        Update generation strategy based on hybrid learning
        """
        updated_config = current_config.copy()
        
        # Apply fast reward learnings
        if 'fast_rewards_reliable' in learned_strategies:
            # Fast rewards are working well, can reduce VLM usage
            self.hybrid_config['vlm_frequency'] = learned_strategies.get(
                'recommended_vlm_frequency', self.hybrid_config['vlm_frequency']
            )
        
        # Apply VLM learnings
        if 'temporal_focus_needed' in learned_strategies:
            updated_config['num_frames'] = int(updated_config.get('num_frames', 160) * 1.15)
            print("  🎯 Learned: Increasing temporal focus (more frames)")
        
        if 'causal_emphasis_needed' in learned_strategies:
            updated_config['guidance_scale'] *= 1.1
            print("  🎯 Learned: Emphasizing causal relationships (higher guidance)")
        
        if 'creativity_boost_needed' in learned_strategies:
            updated_config['guidance_scale'] *= 0.95
            print("  🎯 Learned: Boosting creativity (lower guidance)")
        
        # Learn optimal parameters from best hybrid results
        best_results = sorted(hybrid_results, key=lambda x: x.hybrid_reward, reverse=True)[:2]
        if best_results:
            # This would require access to generation params - simplified for now
            print("  📈 Learning from best hybrid performers")
        
        return updated_config
    
    def _select_best_hybrid_episodes(
        self,
        episodes: List[Dict[str, Any]],
        hybrid_results: List[HybridRewardResult]
    ) -> List[Dict[str, Any]]:
        """
        Select best episodes based on hybrid rewards
        """
        # Combine episodes with their hybrid results
        episode_results = list(zip(episodes, hybrid_results))
        
        # Sort by hybrid reward
        episode_results.sort(key=lambda x: x[1].hybrid_reward, reverse=True)
        
        # Select top performers
        num_selected = max(1, len(episodes) // 2)
        best_episodes = [ep for ep, hr in episode_results[:num_selected]]
        
        print(f"  🏆 Selected {num_selected} best episodes based on hybrid rewards")
        
        return best_episodes
    
    def _compute_performance_metrics(self, hybrid_results: List[HybridRewardResult]) -> Dict[str, float]:
        """
        Compute performance metrics for the hybrid system
        """
        fast_rewards = [hr.fast_reward for hr in hybrid_results]
        vlm_rewards = [hr.vlm_reward for hr in hybrid_results if hr.vlm_feedback]
        hybrid_rewards = [hr.hybrid_reward for hr in hybrid_results]
        
        metrics = {
            'avg_fast_reward': np.mean(fast_rewards),
            'avg_vlm_reward': np.mean(vlm_rewards) if vlm_rewards else 0.0,
            'avg_hybrid_reward': np.mean(hybrid_rewards),
            'fast_reward_std': np.std(fast_rewards),
            'vlm_reward_std': np.std(vlm_rewards) if vlm_rewards else 0.0,
            'hybrid_reward_std': np.std(hybrid_rewards),
            'vlm_usage_rate': len(vlm_rewards) / len(hybrid_results),
            'avg_vlm_confidence': np.mean([hr.confidence for hr in hybrid_results if hr.vlm_feedback])
        }
        
        return metrics
    
    def _print_iteration_summary(self, iteration_result: Dict[str, Any]):
        """Print summary of iteration performance"""
        metrics = iteration_result['performance_metrics']
        
        print(f"  📊 Performance Summary:")
        print(f"    Average Rewards: Fast={metrics['avg_fast_reward']:.3f}, "
              f"VLM={metrics['avg_vlm_reward']:.3f}, Hybrid={metrics['avg_hybrid_reward']:.3f}")
        print(f"    VLM Usage: {metrics['vlm_usage_rate']:.1%}, "
              f"Confidence: {metrics.get('avg_vlm_confidence', 0):.3f}")
        
        if iteration_result['learned_strategies']:
            print(f"    🧠 New Learnings: {list(iteration_result['learned_strategies'].keys())}")
    
    def _analyze_hybrid_performance(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze overall performance of the hybrid system
        """
        iterations = list(all_results.keys())
        
        # Track improvement over iterations
        hybrid_scores = []
        vlm_usage_rates = []
        correlation_strengths = []
        
        for iteration in iterations:
            metrics = all_results[iteration]['performance_metrics']
            hybrid_scores.append(metrics['avg_hybrid_reward'])
            vlm_usage_rates.append(metrics['vlm_usage_rate'])
            
            # Get correlation strength at this iteration
            if self.reward_correlation_history:
                recent_correlations = [
                    entry['correlation_score'] 
                    for entry in self.reward_correlation_history[-5:]  # Last 5 entries
                ]
                correlation_strengths.append(np.mean(recent_correlations))
            else:
                correlation_strengths.append(0.0)
        
        analysis = {
            'total_improvement': hybrid_scores[-1] - hybrid_scores[0],
            'improvement_rate': np.polyfit(range(len(hybrid_scores)), hybrid_scores, 1)[0],
            'vlm_efficiency': {
                'initial_usage': vlm_usage_rates[0],
                'final_usage': vlm_usage_rates[-1],
                'usage_optimization': vlm_usage_rates[0] - vlm_usage_rates[-1]  # Positive = more efficient
            },
            'correlation_learning': {
                'initial_correlation': correlation_strengths[0],
                'final_correlation': correlation_strengths[-1],
                'correlation_improvement': correlation_strengths[-1] - correlation_strengths[0]
            },
            'hybrid_advantages': self._identify_hybrid_advantages(all_results)
        }
        
        print(f"\n🎯 HYBRID SYSTEM ANALYSIS:")
        print(f"  Total Improvement: {analysis['total_improvement']:+.3f}")
        print(f"  VLM Usage Optimization: {analysis['vlm_efficiency']['usage_optimization']:+.1%}")
        print(f"  Correlation Learning: {analysis['correlation_learning']['correlation_improvement']:+.3f}")
        
        return analysis
    
    def _identify_hybrid_advantages(self, all_results: Dict[str, Any]) -> List[str]:
        """
        Identify specific advantages of the hybrid approach
        """
        advantages = []
        
        # Check if hybrid outperformed individual components
        final_iteration = list(all_results.keys())[-1]
        final_metrics = all_results[final_iteration]['performance_metrics']
        
        if final_metrics['avg_hybrid_reward'] > final_metrics['avg_fast_reward']:
            advantages.append("Hybrid rewards outperformed fast rewards alone")
        
        if final_metrics['avg_hybrid_reward'] > final_metrics['avg_vlm_reward']:
            advantages.append("Hybrid rewards outperformed VLM rewards alone")
        
        # Check for efficiency gains
        if final_metrics['vlm_usage_rate'] < 0.8:  # Using VLM efficiently
            advantages.append("Achieved high performance with efficient VLM usage")
        
        # Check for learning effectiveness
        if len(self.reward_correlation_history) > 5:
            recent_correlation = np.mean([
                entry['correlation_score'] 
                for entry in self.reward_correlation_history[-5:]
            ])
            if recent_correlation > 0.7:
                advantages.append("Successfully learned correlation between fast and VLM rewards")
        
        return advantages

# Integration with existing framework
class EnhancedVideoGRPOSearcher:
    """
    Enhanced GRPO searcher with hybrid reward capabilities
    """
    
    def __init__(self, video_generator, fast_reward_function: Callable, gemini_api_key: str = None):
        self.hybrid_system = HybridGRPOSystem(
            video_generator, fast_reward_function, gemini_api_key
        )
    
    def search_with_hybrid_rewards(
        self,
        prompts: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main search method using hybrid rewards
        """
        return self.hybrid_system.hybrid_grpo_search(prompts, **kwargs)

# Example usage
def example_hybrid_grpo():
    """
    Example of hybrid GRPO system
    """
    print("🔥 EXAMPLE: HYBRID GRPO SYSTEM")
    print("=" * 50)
    
    # Mock components
    class MockVideoGenerator:
        def generate_single(self, prompt, **kwargs):
            return torch.randn(3, kwargs.get('num_frames', 16), 224, 224)
    
    def mock_fast_reward(video, prompt):
        # Fast computational reward
        motion = torch.diff(video, dim=1).abs().mean().item()
        quality = video.var().item()
        return {
            'reward': 0.6 * motion + 0.4 * quality,
            'reward_info': {'motion': motion, 'quality': quality}
        }
    
    # Initialize hybrid system
    generator = MockVideoGenerator()
    hybrid_system = HybridGRPOSystem(
        video_generator=generator,
        fast_reward_function=mock_fast_reward,
        gemini_api_key=None  # Would use real API key
    )
    
    # Run hybrid search
    results = hybrid_system.hybrid_grpo_search(
        prompts=["A scientist conducts an experiment showing cause and effect"],
        num_videos_per_prompt=4,
        num_iterations=3
    )
    
    print(f"\nHybrid search completed!")
    print(f"Learned strategies: {len(results['learned_strategies'])}")
    print(f"Correlation history entries: {len(results['reward_correlation_history'])}")

if __name__ == "__main__":
    example_hybrid_grpo()


