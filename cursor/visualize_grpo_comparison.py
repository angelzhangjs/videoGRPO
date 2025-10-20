#!/usr/bin/env python3
"""
Visualization: Old GRPO vs Full Sequence GRPO
Creates diagrams to show the difference
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def visualize_reward_comparison():
    """
    Visualize the difference between single-reward and full-sequence GRPO
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ============= Old GRPO (Single Reward) =============
    ax1.set_title('Old GRPO: Single Reward Per Video', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Video frames
    for i in range(8):
        rect = FancyBboxPatch(
            (i * 1.2, 7), 0.9, 1.5,
            boxstyle="round,pad=0.05",
            edgecolor='steelblue',
            facecolor='lightblue',
            linewidth=2
        )
        ax1.add_patch(rect)
        ax1.text(i * 1.2 + 0.45, 7.75, f'F{i}', ha='center', va='center', fontsize=9)
    
    # Arrow to single reward
    arrow1 = FancyArrowPatch(
        (4, 6.5), (4, 4),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color='black',
        linewidth=2
    )
    ax1.add_patch(arrow1)
    
    # Single reward box
    reward_box = FancyBboxPatch(
        (2.5, 2), 3, 1.5,
        boxstyle="round,pad=0.1",
        edgecolor='red',
        facecolor='lightcoral',
        linewidth=3
    )
    ax1.add_patch(reward_box)
    ax1.text(4, 2.75, 'Reward = 0.73', ha='center', va='center', 
             fontsize=12, fontweight='bold')
    
    # Limitations text
    ax1.text(5, 0.5, '❌ No per-frame insight\n❌ No temporal analysis\n❌ Limited debugging', 
             ha='left', va='bottom', fontsize=10, color='darkred',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============= Full Sequence GRPO =============
    ax2.set_title('Full Sequence GRPO: Frame-Level Rewards', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Video frames
    for i in range(8):
        rect = FancyBboxPatch(
            (i * 1.2, 7), 0.9, 1.5,
            boxstyle="round,pad=0.05",
            edgecolor='steelblue',
            facecolor='lightblue',
            linewidth=2
        )
        ax2.add_patch(rect)
        ax2.text(i * 1.2 + 0.45, 7.75, f'F{i}', ha='center', va='center', fontsize=9)
    
    # Individual frame rewards
    frame_rewards = [0.82, 0.78, 0.75, 0.71, 0.68, 0.73, 0.76, 0.79]
    for i, reward in enumerate(frame_rewards):
        arrow = FancyArrowPatch(
            (i * 1.2 + 0.45, 6.8), (i * 1.2 + 0.45, 5.5),
            arrowstyle='->,head_width=0.2,head_length=0.2',
            color='green',
            linewidth=1.5
        )
        ax2.add_patch(arrow)
        
        # Small reward box
        small_box = FancyBboxPatch(
            (i * 1.2 + 0.1, 4.8), 0.7, 0.6,
            boxstyle="round,pad=0.03",
            edgecolor='green',
            facecolor='lightgreen',
            linewidth=1.5
        )
        ax2.add_patch(small_box)
        ax2.text(i * 1.2 + 0.45, 5.1, f'{reward:.2f}', 
                ha='center', va='center', fontsize=7)
    
    # Aggregation arrow
    arrow2 = FancyArrowPatch(
        (4.5, 4.5), (4.5, 3.5),
        arrowstyle='->,head_width=0.4,head_length=0.4',
        color='black',
        linewidth=2
    )
    ax2.add_patch(arrow2)
    
    # Aggregated metrics
    metrics_box = FancyBboxPatch(
        (1.5, 1.5), 6, 1.8,
        boxstyle="round,pad=0.1",
        edgecolor='green',
        facecolor='lightgreen',
        linewidth=3
    )
    ax2.add_patch(metrics_box)
    ax2.text(4.5, 2.7, 'Sequence: 0.75', ha='center', va='center', fontsize=10, fontweight='bold')
    ax2.text(4.5, 2.3, 'Temporal: 0.88 | Progression: 0.68', ha='center', va='center', fontsize=9)
    ax2.text(4.5, 1.9, 'Total: 0.77', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Benefits text
    ax2.text(5, 0.3, '✅ Per-frame analysis\n✅ Temporal consistency\n✅ Progression tracking', 
             ha='left', va='bottom', fontsize=10, color='darkgreen',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('outputs/grpo_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: outputs/grpo_comparison.png")
    plt.show()


def plot_temporal_analysis():
    """
    Plot showing temporal analysis capabilities
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Frame indices
    frames = np.arange(0, 121)
    
    # ============= Plot 1: Per-Frame Rewards =============
    ax = axes[0, 0]
    
    # Simulate different reward patterns
    good_rewards = 0.8 + 0.1 * np.sin(frames / 20) + np.random.normal(0, 0.02, len(frames))
    bad_rewards = 0.5 + np.random.normal(0, 0.15, len(frames))
    
    ax.plot(frames, good_rewards, label='Good Video (consistent)', linewidth=2, color='green')
    ax.plot(frames, bad_rewards, label='Bad Video (inconsistent)', linewidth=2, color='red', alpha=0.7)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Target threshold')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Per-Frame Reward')
    ax.set_title('Per-Frame Reward Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ============= Plot 2: Red Dot Progression =============
    ax = axes[0, 1]
    
    # Simulate red dot fading
    ideal_fade = 0.1 * np.exp(-frames / 30)
    actual_fade = ideal_fade + np.random.normal(0, 0.005, len(frames))
    poor_fade = 0.1 * (1 - frames / 121) + np.random.normal(0, 0.02, len(frames))
    
    ax.plot(frames, ideal_fade, label='Ideal Fade', linewidth=2, color='blue', linestyle='--')
    ax.plot(frames, actual_fade, label='Actual (good)', linewidth=2, color='green')
    ax.plot(frames, poor_fade, label='Actual (poor)', linewidth=2, color='red', alpha=0.7)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Red Pixel Ratio')
    ax.set_title('Red Dot Fading Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ============= Plot 3: Temporal Consistency =============
    ax = axes[1, 0]
    
    # Frame-to-frame differences
    good_diffs = np.abs(np.diff(good_rewards))
    bad_diffs = np.abs(np.diff(bad_rewards))
    
    ax.plot(frames[1:], good_diffs, label='Good Video', linewidth=2, color='green')
    ax.plot(frames[1:], bad_diffs, label='Bad Video', linewidth=2, color='red', alpha=0.7)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('|Reward Change|')
    ax.set_title('Temporal Consistency (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.text(0.98, 0.95, f'Good std: {good_diffs.std():.4f}\nBad std: {bad_diffs.std():.4f}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ============= Plot 4: GRPO Rounds Improvement =============
    ax = axes[1, 1]
    
    rounds = np.arange(1, 4)
    candidates_per_round = 4
    
    # Simulate improvement across rounds
    round1_rewards = [0.62, 0.68, 0.71, 0.65]
    round2_rewards = [0.73, 0.75, 0.78, 0.71]
    round3_rewards = [0.81, 0.83, 0.85, 0.79]
    
    all_rewards = [round1_rewards, round2_rewards, round3_rewards]
    
    for i, round_rewards in enumerate(all_rewards):
        x = np.random.normal(i+1, 0.05, len(round_rewards))
        ax.scatter(x, round_rewards, s=100, alpha=0.6, label=f'Round {i+1}')
    
    # Best per round
    best_per_round = [max(r) for r in all_rewards]
    ax.plot(rounds, best_per_round, 'o-', linewidth=3, markersize=10, 
            color='gold', label='Best per round', zorder=10)
    
    ax.set_xlabel('GRPO Round')
    ax.set_ylabel('Total Reward')
    ax.set_title('GRPO Improvement Across Rounds')
    ax.set_xticks(rounds)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/temporal_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: outputs/temporal_analysis.png")
    plt.show()


def create_architecture_diagram():
    """
    Create architecture diagram showing the flow
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Full Sequence GRPO Architecture', 
            ha='center', fontsize=16, fontweight='bold')
    
    # ========== Layer 1: Video Generation ==========
    gen_box = FancyBboxPatch(
        (4, 8), 6, 0.8,
        boxstyle="round,pad=0.1",
        edgecolor='blue',
        facecolor='lightblue',
        linewidth=2
    )
    ax.add_patch(gen_box)
    ax.text(7, 8.4, 'LTX-Video Generation', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch((7, 7.9), (7, 7.3), arrowstyle='->', 
                           mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(arrow)
    
    # ========== Layer 2: Video Frames ==========
    for i in range(8):
        frame_box = FancyBboxPatch(
            (3.5 + i*0.9, 6.5), 0.7, 0.6,
            boxstyle="round,pad=0.05",
            edgecolor='steelblue',
            facecolor='lightcyan',
            linewidth=1.5
        )
        ax.add_patch(frame_box)
        ax.text(3.85 + i*0.9, 6.8, f'F{i}', ha='center', va='center', fontsize=8)
    
    # Arrows to per-frame rewards
    for i in range(8):
        arrow = FancyArrowPatch(
            (3.85 + i*0.9, 6.4), (3.85 + i*0.9, 5.8),
            arrowstyle='->', mutation_scale=15, linewidth=1.5, color='green'
        )
        ax.add_patch(arrow)
    
    # ========== Layer 3: Per-Frame Rewards ==========
    for i in range(8):
        reward_box = FancyBboxPatch(
            (3.5 + i*0.9, 5.0), 0.7, 0.6,
            boxstyle="round,pad=0.05",
            edgecolor='green',
            facecolor='lightgreen',
            linewidth=1.5
        )
        ax.add_patch(reward_box)
        reward_val = 0.75 + np.random.uniform(-0.1, 0.1)
        ax.text(3.85 + i*0.9, 5.3, f'{reward_val:.2f}', ha='center', va='center', fontsize=7)
    
    # ========== Layer 4: Aggregation ==========
    # Left branch: Sequence Reward
    seq_arrow = FancyArrowPatch((5, 4.9), (3, 3.5), arrowstyle='->', 
                               mutation_scale=20, linewidth=2, color='purple')
    ax.add_patch(seq_arrow)
    
    seq_box = FancyBboxPatch(
        (1, 3), 3, 0.8,
        boxstyle="round,pad=0.1",
        edgecolor='purple',
        facecolor='plum',
        linewidth=2
    )
    ax.add_patch(seq_box)
    ax.text(2.5, 3.4, 'Sequence Reward\n0.75', ha='center', va='center', fontsize=9)
    
    # Middle branch: Temporal Consistency
    temp_arrow = FancyArrowPatch((7, 4.9), (7, 3.5), arrowstyle='->', 
                                mutation_scale=20, linewidth=2, color='orange')
    ax.add_patch(temp_arrow)
    
    temp_box = FancyBboxPatch(
        (5.5, 3), 3, 0.8,
        boxstyle="round,pad=0.1",
        edgecolor='orange',
        facecolor='moccasin',
        linewidth=2
    )
    ax.add_patch(temp_box)
    ax.text(7, 3.4, 'Temporal Consistency\n0.88', ha='center', va='center', fontsize=9)
    
    # Right branch: Progression
    prog_arrow = FancyArrowPatch((9, 4.9), (11, 3.5), arrowstyle='->', 
                                mutation_scale=20, linewidth=2, color='brown')
    ax.add_patch(prog_arrow)
    
    prog_box = FancyBboxPatch(
        (10, 3), 3, 0.8,
        boxstyle="round,pad=0.1",
        edgecolor='brown',
        facecolor='bisque',
        linewidth=2
    )
    ax.add_patch(prog_box)
    ax.text(11.5, 3.4, 'Progression\n0.68', ha='center', va='center', fontsize=9)
    
    # ========== Layer 5: Total Reward ==========
    # Arrows to total
    for x in [2.5, 7, 11.5]:
        arrow = FancyArrowPatch((x, 2.9), (7, 2.2), arrowstyle='->', 
                               mutation_scale=20, linewidth=2, color='black')
        ax.add_patch(arrow)
    
    total_box = FancyBboxPatch(
        (5, 1.5), 4, 0.8,
        boxstyle="round,pad=0.1",
        edgecolor='red',
        facecolor='lightcoral',
        linewidth=3
    )
    ax.add_patch(total_box)
    ax.text(7, 1.9, 'Total Reward: 0.77', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # ========== Layer 6: Parameter Update ==========
    update_arrow = FancyArrowPatch((7, 1.4), (7, 0.8), arrowstyle='->', 
                                  mutation_scale=20, linewidth=2, color='black')
    ax.add_patch(update_arrow)
    
    update_box = FancyBboxPatch(
        (4, 0.2), 6, 0.5,
        boxstyle="round,pad=0.1",
        edgecolor='darkgreen',
        facecolor='lightgreen',
        linewidth=2
    )
    ax.add_patch(update_box)
    ax.text(7, 0.45, 'Update Generation Parameters (GRPO)', 
            ha='center', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("Saved: outputs/architecture_diagram.png")
    plt.show()


if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    
    print("Creating visualizations...")
    print("\n1. GRPO Comparison...")
    visualize_reward_comparison()
    
    print("\n2. Temporal Analysis...")
    plot_temporal_analysis()
    
    print("\n3. Architecture Diagram...")
    create_architecture_diagram()
    
    print("\n✅ All visualizations created!")
    print("Check outputs/ directory for PNG files")

