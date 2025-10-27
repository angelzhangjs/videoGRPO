"""
Main script for Visual Intelligence GRPO Video Generation

This script demonstrates how to use our enhanced GRPO algorithm with deep reasoning
capabilities for generating videos that exhibit advanced visual intelligence.

Features:
- Multi-modal reasoning rewards (causal, spatial, abstract, physics)
- Self-improving system with VLM feedback
- Hierarchical complexity scaling
- Advanced physics understanding
- Comprehensive reasoning analysis

Usage:
    python main_visual_intelligence.py --prompt "A robot learning to solve a Rubik's cube" \
                                      --reasoning-types causal spatial abstract \
                                      --complexity-level 4 \
                                      --self-improve
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add project paths
sys.path.append("ltx_video_source")
sys.path.append(".")

# Import our enhanced system
from visual_intelligence_grpo import (
    VisualIntelligenceGRPO, 
    ReasoningConfig, 
    create_reasoning_config
)

# Import existing components
from main import create_ltx_video_pipeline, check_gpu_memory

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

def create_reasoning_prompts() -> Dict[str, Dict[str, Any]]:
    """
    Create a collection of reasoning-focused prompts for testing
    """
    return {
        "physics_reasoning": {
            "prompt": "A ball bouncing down a staircase, demonstrating realistic physics with each bounce getting smaller due to energy loss",
            "reasoning_types": ["causal", "spatial", "physics"],
            "complexity_level": 3,
            "description": "Tests understanding of energy conservation, gravity, and cause-effect relationships"
        },
        
        "problem_solving": {
            "prompt": "A robot methodically solving a Rubik's cube, showing step-by-step logical reasoning and spatial understanding",
            "reasoning_types": ["causal", "spatial", "abstract", "problem_solving"],
            "complexity_level": 4,
            "description": "Tests sequential reasoning, spatial manipulation, and problem-solving strategies"
        },
        
        "causal_chain": {
            "prompt": "A domino effect where one falling domino causes a chain reaction that eventually triggers a Rube Goldberg machine",
            "reasoning_types": ["causal", "temporal", "physics"],
            "complexity_level": 3,
            "description": "Tests understanding of causal chains and temporal sequences"
        },
        
        "spatial_reasoning": {
            "prompt": "A person navigating through a complex 3D maze, showing understanding of spatial relationships and path planning",
            "reasoning_types": ["spatial", "abstract", "problem_solving"],
            "complexity_level": 4,
            "description": "Tests 3D spatial understanding and strategic planning"
        },
        
        "abstract_concepts": {
            "prompt": "A visual metaphor showing 'time is money' - clocks transforming into coins in a creative, meaningful way",
            "reasoning_types": ["abstract", "metaphorical", "creative"],
            "complexity_level": 5,
            "description": "Tests abstract thinking and metaphorical understanding"
        },
        
        "biological_motion": {
            "prompt": "A bird learning to fly, showing the progression from awkward flapping to graceful soaring with realistic wing dynamics",
            "reasoning_types": ["physics", "biological", "causal"],
            "complexity_level": 3,
            "description": "Tests understanding of biological motion patterns and learning processes"
        },
        
        "material_properties": {
            "prompt": "Different materials (rubber ball, glass marble, clay ball) being dropped and showing their unique deformation and bounce characteristics",
            "reasoning_types": ["physics", "material_properties", "causal"],
            "complexity_level": 3,
            "description": "Tests understanding of material properties and physics"
        },
        
        "social_reasoning": {
            "prompt": "Two people collaborating to move a heavy object, showing coordination, communication, and shared problem-solving",
            "reasoning_types": ["social", "causal", "problem_solving"],
            "complexity_level": 4,
            "description": "Tests social intelligence and collaborative reasoning"
        }
    }

def setup_output_directory(base_name: str = "visual_intelligence") -> Path:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/{base_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def analyze_reasoning_progression(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze how reasoning capabilities improve across GRPO rounds
    """
    rounds = results.get('rounds', [])
    if not rounds:
        return {}
    
    progression = {
        'intelligence_scores': [],
        'reward_progression': [],
        'reasoning_improvements': {},
        'complexity_scaling': []
    }
    
    for round_data in rounds:
        # Extract intelligence scores
        reasoning_scores = round_data.get('reasoning_scores', [])
        if reasoning_scores:
            intelligence_score = reasoning_scores[0].get('intelligence_score', 0)
            progression['intelligence_scores'].append(intelligence_score)
        
        # Extract reward progression
        progression['reward_progression'].append(round_data.get('best_reward', 0))
    
    # Calculate improvement metrics
    if len(progression['intelligence_scores']) > 1:
        initial_score = progression['intelligence_scores'][0]
        final_score = progression['intelligence_scores'][-1]
        improvement = final_score - initial_score
        
        progression['total_improvement'] = improvement
        progression['improvement_rate'] = improvement / len(progression['intelligence_scores'])
    
    return progression

def generate_reasoning_report(results: Dict[str, Any], output_dir: Path):
    """
    Generate comprehensive reasoning analysis report
    """
    report = {
        "generation_summary": {
            "prompt": results['prompt'],
            "reasoning_types": results['reasoning_config'].reasoning_types,
            "complexity_level": results['reasoning_config'].complexity_level,
            "total_rounds": len(results['rounds']),
            "timestamp": datetime.now().isoformat()
        },
        
        "performance_metrics": {
            "final_intelligence_score": results['reasoning_analysis']['overall_intelligence'],
            "best_reward": max([r['best_reward'] for r in results['rounds']]),
            "avg_reward": np.mean([r['avg_reward'] for r in results['rounds']]),
            "reasoning_breakdown": results['reasoning_analysis']['reasoning_breakdown']
        },
        
        "reasoning_analysis": results['reasoning_analysis'],
        "progression_analysis": analyze_reasoning_progression(results),
        "improvement_trajectory": results.get('improvement_trajectory', [])
    }
    
    # Save detailed report
    report_path = output_dir / "reasoning_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate human-readable summary
    summary_path = output_dir / "reasoning_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("🧠 VISUAL INTELLIGENCE GRPO - REASONING ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"📝 PROMPT: {results['prompt']}\n")
        f.write(f"🎯 REASONING TYPES: {', '.join(results['reasoning_config'].reasoning_types)}\n")
        f.write(f"📊 COMPLEXITY LEVEL: {results['reasoning_config'].complexity_level}/5\n\n")
        
        f.write("🏆 PERFORMANCE SUMMARY:\n")
        f.write(f"  • Overall Intelligence Score: {report['performance_metrics']['final_intelligence_score']:.3f}\n")
        f.write(f"  • Best Reward: {report['performance_metrics']['best_reward']:.3f}\n")
        f.write(f"  • Average Reward: {report['performance_metrics']['avg_reward']:.3f}\n\n")
        
        f.write("🧠 REASONING BREAKDOWN:\n")
        for reasoning_type, score in report['performance_metrics']['reasoning_breakdown'].items():
            f.write(f"  • {reasoning_type}: {score:.3f}\n")
        
        f.write(f"\n✅ STRENGTHS: {', '.join(results['reasoning_analysis']['strengths'])}\n")
        f.write(f"⚠️  WEAKNESSES: {', '.join(results['reasoning_analysis']['weaknesses'])}\n")
        f.write(f"💡 RECOMMENDATIONS: {', '.join(results['reasoning_analysis']['recommendations'])}\n")
        
        # Progression analysis
        progression = report['progression_analysis']
        if 'total_improvement' in progression:
            f.write(f"\n📈 IMPROVEMENT METRICS:\n")
            f.write(f"  • Total Intelligence Improvement: {progression['total_improvement']:.3f}\n")
            f.write(f"  • Improvement Rate per Round: {progression['improvement_rate']:.3f}\n")
    
    print(f"📊 Reasoning report saved to {output_dir}")
    return report

def main():
    parser = argparse.ArgumentParser(description="Visual Intelligence GRPO Video Generation")
    
    # Basic generation parameters
    parser.add_argument("--prompt", type=str, help="Text prompt for video generation")
    parser.add_argument("--preset", type=str, choices=list(create_reasoning_prompts().keys()),
                       help="Use a preset reasoning prompt")
    
    # Reasoning configuration
    parser.add_argument("--reasoning-types", nargs="+", 
                       choices=["causal", "spatial", "abstract", "physics", "problem_solving", 
                               "temporal", "biological", "material_properties", "social"],
                       default=["causal", "spatial", "physics"],
                       help="Types of reasoning to evaluate")
    parser.add_argument("--complexity-level", type=int, default=3, choices=[1, 2, 3, 4, 5],
                       help="Reasoning complexity level (1=basic, 5=advanced)")
    parser.add_argument("--physics-level", type=int, default=2, choices=[1, 2, 3, 4],
                       help="Physics understanding level")
    parser.add_argument("--reasoning-weight", type=float, default=0.7,
                       help="Weight of reasoning rewards vs traditional rewards")
    
    # GRPO parameters
    parser.add_argument("--num-candidates", type=int, default=12,
                       help="Number of candidates per GRPO round")
    parser.add_argument("--num-rounds", type=int, default=4,
                       help="Number of GRPO optimization rounds")
    
    # Self-improvement
    parser.add_argument("--self-improve", action="store_true",
                       help="Enable self-improvement with VLM feedback")
    parser.add_argument("--gemini-api-key", type=str,
                       help="Gemini API key for VLM feedback")
    
    # Output
    parser.add_argument("--output-dir", type=str,
                       help="Output directory (default: auto-generated)")
    parser.add_argument("--save-analysis", action="store_true", default=True,
                       help="Save detailed reasoning analysis")
    
    # GPU configuration
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use for generation")
    
    args = parser.parse_args()
    
    # Setup
    print("🧠 VISUAL INTELLIGENCE GRPO - DEEP THINKING VIDEO GENERATION")
    print("=" * 70)
    
    # Check GPU
    if not check_gpu_memory():
        print("⚠️ GPU memory warning - consider reducing parameters")
    
    # Determine prompt and configuration
    if args.preset:
        preset_data = create_reasoning_prompts()[args.preset]
        prompt = preset_data["prompt"]
        reasoning_types = preset_data["reasoning_types"]
        complexity_level = preset_data["complexity_level"]
        print(f"📋 Using preset: {args.preset}")
        print(f"📝 Description: {preset_data['description']}")
    else:
        prompt = args.prompt or "A ball bouncing down stairs with realistic physics"
        reasoning_types = args.reasoning_types
        complexity_level = args.complexity_level
    
    print(f"🎯 Prompt: '{prompt}'")
    print(f"🧠 Reasoning Types: {reasoning_types}")
    print(f"📊 Complexity Level: {complexity_level}/5")
    
    # Create reasoning configuration
    reasoning_config = create_reasoning_config(
        reasoning_types=reasoning_types,
        complexity_level=complexity_level,
        physics_level=args.physics_level,
        self_improvement_cycles=3 if args.self_improve else 0,
        reasoning_weight=args.reasoning_weight,
        enable_meta_reasoning=True
    )
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = setup_output_directory("visual_intelligence")
    
    print(f"💾 Output directory: {output_dir}")
    
    try:
        # Initialize video pipeline
        print("\n🔧 Initializing LTX-Video pipeline...")
        video_pipeline = create_ltx_video_pipeline()
        
        # Initialize Visual Intelligence GRPO system
        print("🧠 Initializing Visual Intelligence GRPO...")
        gemini_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
        vi_grpo = VisualIntelligenceGRPO(video_pipeline, gemini_key)
        
        # Generate video with deep reasoning
        print(f"\n🎬 Starting generation with {args.num_rounds} GRPO rounds...")
        results = vi_grpo.generate_with_deep_reasoning(
            prompt=prompt,
            reasoning_config=reasoning_config,
            num_candidates=args.num_candidates,
            num_grpo_rounds=args.num_rounds,
            save_path=str(output_dir)
        )
        
        # Analysis and reporting
        if args.save_analysis:
            print("\n📊 Generating reasoning analysis report...")
            report = generate_reasoning_report(results, output_dir)
            
            # Print summary
            print("\n🏆 GENERATION COMPLETE - SUMMARY:")
            print(f"  • Overall Intelligence Score: {results['reasoning_analysis']['overall_intelligence']:.3f}")
            print(f"  • Best Round Reward: {max([r['best_reward'] for r in results['rounds']]):.3f}")
            print(f"  • Reasoning Strengths: {', '.join(results['reasoning_analysis']['strengths'][:3])}")
            
            if results['reasoning_analysis']['weaknesses']:
                print(f"  • Areas for Improvement: {', '.join(results['reasoning_analysis']['weaknesses'][:2])}")
        
        # Save final video
        best_video = results['best_video']
        video_save_path = output_dir / "best_reasoning_video.pt"
        torch.save(best_video.video, video_save_path)
        print(f"🎥 Best video saved to: {video_save_path}")
        
        print(f"\n✅ Visual Intelligence GRPO completed successfully!")
        print(f"📁 All results saved to: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

def demo_reasoning_capabilities():
    """
    Demonstrate the reasoning capabilities with multiple test cases
    """
    print("🧠 VISUAL INTELLIGENCE GRPO - REASONING CAPABILITIES DEMO")
    print("=" * 70)
    
    reasoning_prompts = create_reasoning_prompts()
    
    # Select a few interesting test cases
    demo_cases = ["physics_reasoning", "problem_solving", "abstract_concepts"]
    
    for case_name in demo_cases:
        case_data = reasoning_prompts[case_name]
        print(f"\n🎯 Test Case: {case_name.upper()}")
        print(f"📝 Prompt: {case_data['prompt']}")
        print(f"🧠 Reasoning Types: {case_data['reasoning_types']}")
        print(f"📊 Complexity: {case_data['complexity_level']}/5")
        print(f"💡 Description: {case_data['description']}")
        print("-" * 50)

if __name__ == "__main__":
    # Check if running in demo mode
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_reasoning_capabilities()
    else:
        exit_code = main()
        sys.exit(exit_code)
