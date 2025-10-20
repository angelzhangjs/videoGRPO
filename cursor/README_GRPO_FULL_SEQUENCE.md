# GRPO with Full Frame Sequence Calculation - Quick Start

## 🚀 What This Does

Enhances your GRPO video generation with **frame-by-frame reward calculation** instead of a single reward per video. This enables:

- ✅ **121 rewards per video** (one per frame) instead of 1
- ✅ **Temporal consistency** tracking
- ✅ **Task progression** analysis (e.g., red dots fading smoothly)
- ✅ **Better debugging** (see exactly which frames are good/bad)

## 📁 New Files Created

```
grpo_full_sequence.py           # Core framework
ltx_grpo_integration.py         # LTX-Video integration
grpo_full_sequence_example.py   # Complete working example
visualize_grpo_comparison.py    # Visualization tools
GRPO_FULL_SEQUENCE_GUIDE.md     # Detailed documentation
```

## ⚡ Quick Start

### Option 1: Run the Example

```bash
# Basic usage
python grpo_full_sequence_example.py \
    --prompt "A red dot appears and slowly fades away" \
    --num_rounds 3 \
    --candidates_per_round 4

# With reference image
python grpo_full_sequence_example.py \
    --prompt "Erase all the red dots" \
    --original_image data/red_dots.png \
    --output_dir outputs/red_dot_erase \
    --num_rounds 4 \
    --candidates_per_round 6
```

### Option 2: Integrate into Your Code

```python
from grpo_full_sequence import FullSequenceRewardCalculator
from ltx_grpo_integration import LTXVideoGRPOGenerator, GRPOSearchWithLTXVideo

# 1. Define your per-frame reward function
def my_per_frame_reward(frame, frame_idx, prompt):
    # Your custom logic here
    # frame: [C, H, W] tensor
    # frame_idx: 0 to num_frames-1
    # prompt: text prompt
    
    # Example: detect red pixels
    red_ratio = detect_red_pixels(frame)
    reward = 1.0 - red_ratio  # Higher reward for less red
    
    return {
        'reward': reward,
        'red_ratio': red_ratio,
    }

# 2. Create reward calculator
reward_calc = FullSequenceRewardCalculator(
    per_frame_reward_fn=my_per_frame_reward,
)

# 3. Create generator
generator = LTXVideoGRPOGenerator(
    pipeline_config_path="ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml",
    device="cuda",
)

# 4. Run GRPO search
searcher = GRPOSearchWithLTXVideo(generator, reward_calc.compute_full_sequence_reward)
candidates = searcher.grpo_search(
    prompt="Your prompt here",
    num_rounds=3,
    num_candidates_per_round=4,
)

# 5. Best video is candidates[0]
best_video = candidates[0]['video']
best_reward_info = candidates[0]['reward_info']
```

## 📊 Visualizations

```bash
# Create comparison diagrams
python visualize_grpo_comparison.py
```

This creates:
- `outputs/grpo_comparison.png` - Old vs new GRPO comparison
- `outputs/temporal_analysis.png` - Frame-level analysis plots
- `outputs/architecture_diagram.png` - System architecture

## 🎯 Key Differences from Old GRPO

| Feature | Old GRPO | **Full Sequence GRPO** |
|---------|----------|------------------------|
| Rewards per video | 1 | **121** (all frames) |
| Temporal awareness | ❌ | ✅ |
| Frame-level debugging | ❌ | ✅ |
| Task progression | ❌ | ✅ |
| Reward detail | Scalar | **Rich breakdown** |

## 📖 Understanding the Pipeline

### LTX-Video Inference Flow

```
Text → T5 Encoder → Latent Init → Denoising Loop → VAE Decode → Video
                                    ↑
                                    |
                           (All 121 frames generated
                            simultaneously in latent space)
```

**Key insight**: The entire video is generated as one unit in latent space, then decoded to pixels. This means:
- You can't compute rewards **during** generation
- You **can** compute detailed rewards **after** generation on all frames
- Full sequence analysis is fast (no re-generation needed)

### GRPO Flow with Full Sequence

```
Round 1:
  Generate 4 candidates → Evaluate ALL frames → Select best → Update params
  
Round 2:
  Generate 4 candidates → Evaluate ALL frames → Select best → Update params
  
Round 3:
  Generate 4 candidates → Evaluate ALL frames → Select best → Update params

Final: Best overall candidate
```

## 🔍 Example Output

```
=== Round 1/3 ===
Candidate 1/4
  📊 Full Sequence Reward Breakdown:
     Sequence Reward: 0.7234
     Temporal Consistency: 0.8512
     Progression: 0.6789
     Frames Analyzed: 121
     Sample Frame Rewards:
       Frame   0: 0.8123 (red_ratio=0.0852)
       Frame  30: 0.7456 (red_ratio=0.0621)
       Frame  60: 0.7891 (red_ratio=0.0412)
       Frame  90: 0.8234 (red_ratio=0.0198)
       Frame 120: 0.8567 (red_ratio=0.0045)

✅ Round 1 Best Reward: 0.7891
Updated config: guidance=8.5, steps=45

[... Rounds 2 and 3 ...]

GRPO Search Complete!
Best Overall Reward: 0.8523

Results saved to: outputs/grpo_full_sequence/
```

## 🎨 Customization Examples

### Example 1: Object Appearance Task

```python
def object_appearance_reward(frame, frame_idx, prompt):
    # Want object to appear gradually
    object_conf = detect_object(frame, "person")
    
    expected_conf = min(1.0, frame_idx / 60)  # Linear increase
    error = abs(object_conf - expected_conf)
    reward = 1.0 - error
    
    return {
        'reward': reward,
        'object_confidence': object_conf,
    }
```

### Example 2: Motion Smoothness

```python
def motion_smoothness_reward(frames, prompt):
    # Compute optical flow between consecutive frames
    flows = []
    for t in range(frames.shape[1] - 1):
        flow = compute_optical_flow(frames[:, t], frames[:, t+1])
        flows.append(flow)
    
    # Smooth motion = consistent flow magnitudes
    flow_magnitudes = [f.norm() for f in flows]
    smoothness = 1.0 - np.std(flow_magnitudes) / np.mean(flow_magnitudes)
    
    return smoothness
```

### Example 3: Color Transition

```python
def color_transition_reward(frame, frame_idx, prompt):
    # Want scene to transition from blue to red
    blue_ratio = compute_color_ratio(frame, color='blue')
    red_ratio = compute_color_ratio(frame, color='red')
    
    progress = frame_idx / 120  # 0 to 1
    expected_blue = 1.0 - progress
    expected_red = progress
    
    reward = (
        (1.0 - abs(blue_ratio - expected_blue)) * 0.5 +
        (1.0 - abs(red_ratio - expected_red)) * 0.5
    )
    
    return {'reward': reward}
```

## 🐛 Debugging Tips

### 1. Visualize Per-Frame Rewards

```python
import matplotlib.pyplot as plt

frame_rewards = candidate['reward_info']['frame_rewards']
rewards = [fr.reward for fr in frame_rewards]

plt.plot(rewards)
plt.xlabel('Frame')
plt.ylabel('Reward')
plt.title('Per-Frame Reward Over Time')
plt.show()
```

### 2. Find Problematic Frames

```python
# Find frames with low rewards
bad_frames = [
    fr for fr in frame_rewards 
    if fr.reward < 0.5
]

print(f"Found {len(bad_frames)} problematic frames:")
for fr in bad_frames[:5]:
    print(f"  Frame {fr.frame_idx}: {fr.reward:.3f}")
```

### 3. Compare Candidates

```python
for i, cand in enumerate(candidates[:3]):
    print(f"\nCandidate {i+1}:")
    print(f"  Total: {cand['total_reward']:.3f}")
    print(f"  Sequence: {cand['reward_info']['sequence_reward']:.3f}")
    print(f"  Temporal: {cand['reward_info']['temporal_consistency']:.3f}")
    print(f"  Config: {cand['config']}")
```

## 📚 Additional Resources

- **Detailed Guide**: See `GRPO_FULL_SEQUENCE_GUIDE.md` for full documentation
- **Code Examples**: Check `grpo_full_sequence_example.py` for working code
- **LTX-Video Docs**: See `ltx_video_source/` for model documentation

## 💡 Tips for Best Results

1. **Start with more candidates**: Try 6-8 candidates per round for better exploration
2. **Use reference images**: Helps calibrate red dot detection thresholds
3. **Visualize results**: Always plot frame rewards to understand what's happening
4. **Adjust weights**: Tune the balance between sequence/temporal/progression rewards
5. **Iterate rounds**: 3-5 rounds usually sufficient for convergence

## 🤝 Contributing

If you improve the framework:
1. Document your reward function
2. Share example outputs
3. Update this README

## ❓ FAQ

**Q: Why compute rewards after generation, not during?**  
A: LTX-Video generates all frames simultaneously in latent space. We can't evaluate pixel-level features until after VAE decoding.

**Q: Is this slower than old GRPO?**  
A: Frame evaluation is fast (~0.1s per video). The bottleneck is still video generation (~5-10s per video).

**Q: Can I use intermediate latent rewards?**  
A: Possible but challenging. Latent-space metrics don't correspond well to pixel-space task completion.

**Q: How many frames should I evaluate?**  
A: All of them! That's the point of "full sequence" 😊 But you can sample if needed for speed.

---

Happy optimizing! 🚀

For questions or issues, check `GRPO_FULL_SEQUENCE_GUIDE.md` or the code comments.

