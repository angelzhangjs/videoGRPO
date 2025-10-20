# GRPO with Full Frame Sequence Calculation - Complete Guide

## 📚 Table of Contents

1. [LTX-Video Inference Pipeline Overview](#ltx-video-inference-pipeline-overview)
2. [Current GRPO Limitations](#current-grpo-limitations)
3. [Full Frame Sequence Solution](#full-frame-sequence-solution)
4. [Implementation Details](#implementation-details)
5. [Usage Examples](#usage-examples)
6. [Key Modifications](#key-modifications)

---

## 🎬 LTX-Video Inference Pipeline Overview

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LTX-Video Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Text Encoding (T5)                                       │
│     ↓                                                        │
│  2. Latent Initialization                                    │
│     ↓                                                        │
│  3. Denoising Loop (timestep iteration)                      │
│     │                                                        │
│     ├─ Transformer prediction                               │
│     ├─ Classifier-Free Guidance (CFG)                       │
│     ├─ Spatio-Temporal Guidance (STG)                       │
│     └─ Latent update                                        │
│     ↓                                                        │
│  4. VAE Decoding (latents → pixels)                         │
│     ↓                                                        │
│  5. Video Output [B, C, T, H, W]                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components in `inference.py`

#### 1. **Pipeline Creation** (`create_ltx_video_pipeline`, lines 207-291)

```python
def create_ltx_video_pipeline(ckpt_path, precision, ...):
    # Loads:
    # - VAE (CausalVideoAutoencoder)
    # - Transformer (Transformer3DModel)
    # - Text Encoder (T5)
    # - Scheduler (RectifiedFlowScheduler)
    return LTXVideoPipeline(...)
```

#### 2. **Denoising Loop** (`LTXVideoPipeline.__call__`, lines 1115-1291)

**This is the core of video generation and where GRPO modifications happen!**

```python
for i, t in enumerate(timesteps):  # Line 1115
    # 1. Prepare latent input
    latent_model_input = torch.cat([latents] * num_conds)
    
    # 2. Transformer forward pass
    noise_pred = self.transformer(
        latent_model_input,
        encoder_hidden_states=prompt_embeds,
        timestep=current_timestep,
        ...
    )
    
    # 3. Apply guidance
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    
    # 4. Update latents (denoising step)
    latents = self.denoising_step(
        latents, noise_pred, current_timestep, ...
    )
```

**Key insight**: The entire video is generated **in latent space** during the loop. Each timestep refines **all frames simultaneously**.

#### 3. **VAE Decoding** (lines 1327-1333)

```python
image = vae_decode(
    latents,  # [B, C, T_latent, H_latent, W_latent]
    self.vae,
    is_video,
    vae_per_channel_normalize=True,
)
# Output: [B, C, T, H, W] - Full video in pixel space
```

**All frames are decoded at once**, not frame-by-frame!

#### 4. **Frame Processing** (lines 601-633)

```python
for i in range(images.shape[0]):  # Batch dimension
    video_np = images[i].permute(1, 2, 3, 0).cpu().float().numpy()
    # [C, T, H, W] → [T, H, W, C]
    video_np = (video_np * 255).astype(np.uint8)
    
    # Save video
    with imageio.get_writer(output_filename, fps=fps) as video:
        for frame in video_np:  # Iterate frames for saving
            video.append_data(frame)
```

---

## ❌ Current GRPO Limitations

### Problem 1: Single Reward per Video

Your current `grpo_generate.sh` script:

```bash
for seed in 2025 2026 2027 2028; do
    python inference.py \
        --prompt "$PROMPT" \
        --seed $seed \
        --guidance_scale 5.0 \
        --output_path "$OUTPUT_DIR/stage1/"
done

# Then manually compare videos
```

**Issue**: Reward is computed **after** full generation, only **once per video**.

### Problem 2: No Frame-Level Feedback

```python
# Typical reward function
def compute_reward(video: torch.Tensor, prompt: str) -> float:
    # Some aggregate metric
    return overall_score
```

**Issue**: No understanding of **how the video progresses frame-by-frame**.

### Problem 3: Limited Exploration

- Only varies seeds and hyperparameters
- No feedback loop during generation
- No frame-level optimization

---

## ✅ Full Frame Sequence Solution

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│              Full Sequence GRPO Architecture                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Generate Video                                              │
│      ↓                                                       │
│  [Frame 0] [Frame 1] ... [Frame T]                          │
│      ↓          ↓              ↓                             │
│  Reward_0  Reward_1  ...  Reward_T  ← PER-FRAME REWARDS     │
│      │          │              │                             │
│      └──────────┴──────────────┘                             │
│                 ↓                                            │
│         Sequence-Level Reward                                │
│                 ↓                                            │
│      Temporal Consistency Check                              │
│                 ↓                                            │
│         Progression Score                                    │
│                 ↓                                            │
│         TOTAL REWARD                                         │
│                 ↓                                            │
│    Update Generation Parameters (GRPO-style)                │
│                 ↓                                            │
│         Generate Next Candidate                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Differences

| Aspect | Old GRPO | **Full Sequence GRPO** |
|--------|----------|------------------------|
| Reward Granularity | 1 per video | **T per video** (every frame) |
| Temporal Understanding | None | **Explicit** (consistency + progression) |
| Feedback Detail | Scalar | **Rich breakdown** (per-frame + sequence) |
| Optimization Target | Video-level | **Frame-level + sequence-level** |

---

## 🔧 Implementation Details

### 1. Full Sequence Reward Calculator

**File**: `grpo_full_sequence.py`

```python
class FullSequenceRewardCalculator:
    def compute_full_sequence_reward(
        self, 
        video_frames: torch.Tensor,  # [C, T, H, W]
        prompt: str
    ) -> SequenceReward:
        """
        Computes rewards for EVERY frame + sequence-level metrics
        """
        # 1. Per-frame rewards
        frame_rewards = []
        for frame_idx in range(T):
            frame = video_frames[:, frame_idx, :, :]
            reward = self.per_frame_reward_fn(frame, frame_idx, prompt)
            frame_rewards.append(reward)
        
        # 2. Sequence-level reward
        sequence_reward = self.sequence_reward_fn(video_frames, prompt)
        
        # 3. Temporal consistency
        temporal_consistency = self._compute_consistency(video_frames)
        
        # 4. Progression (task completion over time)
        progression = self._compute_progression(frame_rewards)
        
        return SequenceReward(
            frame_rewards=frame_rewards,
            sequence_reward=sequence_reward,
            temporal_consistency=temporal_consistency,
            progression_score=progression,
        )
```

**Key Features**:
- ✅ Evaluates **every single frame**
- ✅ Computes temporal consistency between consecutive frames
- ✅ Tracks task progression (e.g., red dots fading over time)
- ✅ Returns detailed breakdown for analysis

### 2. Red Dot Full Sequence Reward

**File**: `grpo_full_sequence_example.py`

```python
class RedDotFullSequenceReward:
    def per_frame_reward(
        self, 
        frame: torch.Tensor,  # [C, H, W]
        frame_idx: int, 
        prompt: str
    ) -> Dict[str, float]:
        """
        THIS IS CALLED FOR EVERY FRAME!
        """
        # Detect red dots in this frame
        red_ratio = self._compute_red_ratio(frame)
        red_count = self._count_red_dots(frame)
        
        if "erase" in prompt.lower():
            # Expected: red decreases linearly
            expected = max(0, 1.0 - (frame_idx / total_frames))
            reward = 1.0 - abs(red_ratio - expected)
        elif "fade" in prompt.lower():
            # Expected: smooth exponential decay
            expected = math.exp(-frame_idx / 30)
            reward = 1.0 - abs(red_ratio - expected)
        
        return {
            'reward': reward,
            'red_ratio': red_ratio,
            'red_count': red_count,
        }
    
    def sequence_reward(
        self, 
        frames: torch.Tensor,  # [C, T, H, W]
        prompt: str
    ) -> float:
        """
        Evaluate the FULL sequence behavior
        """
        red_ratios = [self._compute_red_ratio(frames[:, t]) 
                      for t in range(T)]
        
        if "erase" in prompt:
            # Want monotonic decrease
            diffs = np.diff(red_ratios)
            monotonic_score = (diffs <= 0).mean()
            final_reduction = 1.0 - (red_ratios[-1] / red_ratios[0])
            return 0.5 * monotonic_score + 0.5 * final_reduction
```

**Why This Works Better**:

1. **Frame-level granularity**: Knows exactly which frames are good/bad
2. **Temporal awareness**: Understands if red dots fade smoothly vs. abruptly
3. **Task-specific**: Different reward logic for "erase", "fade", "appear", etc.
4. **Debuggable**: Can visualize reward per frame

### 3. LTX-Video Integration

**File**: `ltx_grpo_integration.py`

```python
class LTXVideoGRPOGenerator:
    def generate_with_full_sequence_tracking(
        self,
        prompt: str,
        reward_calculator: Callable,
        **kwargs
    ):
        # 1. Generate full video
        video = self.generate(prompt=prompt, **kwargs)
        # video shape: [1, C, T, H, W]
        
        # 2. Compute full sequence rewards
        reward_info = reward_calculator(video, prompt)
        # reward_info contains:
        #   - frame_rewards: List[FrameReward]
        #   - sequence_reward: float
        #   - temporal_consistency: float
        #   - progression_score: float
        
        return video, reward_info
```

### 4. GRPO Search with Full Sequence

**File**: `ltx_grpo_integration.py`

```python
class GRPOSearchWithLTXVideo:
    def grpo_search(
        self,
        prompt: str,
        num_rounds: int = 3,
        num_candidates_per_round: int = 4,
    ):
        for round_idx in range(num_rounds):
            # Generate candidates with varied parameters
            for cand_idx in range(num_candidates_per_round):
                config = self._vary_parameters(base_config, round_idx, cand_idx)
                
                # Generate + evaluate with FULL SEQUENCE rewards
                video, reward_info = generator.generate_with_full_sequence_tracking(
                    prompt=prompt,
                    reward_calculator=full_sequence_reward_fn,
                    **config
                )
                
                # reward_info now contains detailed frame-level analysis!
                
            # Update parameters based on best candidate
            best_config = top_candidates[0]['config']
            base_config = self._update_config(base_config, best_config)
```

---

## 📖 Usage Examples

### Example 1: Basic Full Sequence GRPO

```bash
python grpo_full_sequence_example.py \
    --prompt "A red dot appears and slowly fades away" \
    --output_dir outputs/red_dot_fade \
    --num_rounds 3 \
    --candidates_per_round 4
```

**What happens**:
1. Generates 4 video candidates (round 1)
2. For EACH candidate:
   - Evaluates ALL 121 frames
   - Computes per-frame red dot detection
   - Checks if fading is smooth and monotonic
   - Measures temporal consistency
3. Selects best candidate
4. Updates generation parameters (guidance scale, steps)
5. Repeats for rounds 2 and 3

**Output**:
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
```

### Example 2: Red Dot Erasing Task

```python
from grpo_full_sequence_example import run_grpo_full_sequence_search

candidates = run_grpo_full_sequence_search(
    prompt="Erase all the red dots from the image",
    original_image_path="data/red_dots_original.png",
    num_rounds=4,
    candidates_per_round=6,
)

# Analyze best candidate
best = candidates[0]
frame_rewards = best['reward_info']['frame_rewards']

# Plot frame-by-frame reward
import matplotlib.pyplot as plt
rewards = [fr.reward for fr in frame_rewards]
red_ratios = [fr.reward_components['red_ratio'] for fr in frame_rewards]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.title('Per-Frame Reward')
plt.subplot(1, 2, 2)
plt.plot(red_ratios)
plt.title('Red Pixel Ratio Over Time')
plt.show()
```

### Example 3: Custom Reward Function

```python
def custom_per_frame_reward(frame, frame_idx, prompt):
    """Your custom reward logic"""
    # Example: Reward based on object detection
    objects = detect_objects(frame)
    
    if "person appears" in prompt:
        # Want person confidence to increase
        person_conf = objects.get('person', 0.0)
        expected_conf = min(1.0, frame_idx / 60)
        reward = 1.0 - abs(person_conf - expected_conf)
    
    return {
        'reward': reward,
        'objects': objects,
    }

# Use with full sequence calculator
reward_calc = FullSequenceRewardCalculator(
    per_frame_reward_fn=custom_per_frame_reward,
    sequence_reward_fn=your_sequence_fn,
)
```

---

## 🔑 Key Modifications to Make

### Modification 1: Enable Frame-by-Frame Evaluation

**Before** (your current `red_dot_grpo.py`):
```python
def compute_reward(self, video_frames: torch.Tensor, prompt: str) -> Dict:
    # Process entire video at once
    tracking_score = self._compute_tracking_score(video_frames)
    completion_score = self._compute_erasing_score(video_frames)
    
    return {'total_reward': tracking_score * 0.5 + completion_score * 0.5}
```

**After** (full sequence):
```python
def compute_full_sequence_reward(self, video_frames: torch.Tensor, prompt: str):
    # 1. Per-frame evaluation
    frame_rewards = []
    for t in range(num_frames):
        frame = video_frames[:, t, :, :]
        frame_reward = self.evaluate_single_frame(frame, t, prompt)
        frame_rewards.append(frame_reward)
    
    # 2. Sequence-level metrics
    sequence_reward = self.evaluate_sequence(video_frames, prompt)
    temporal_consistency = self.compute_consistency(video_frames)
    
    return SequenceReward(
        frame_rewards=frame_rewards,
        sequence_reward=sequence_reward,
        temporal_consistency=temporal_consistency,
    )
```

### Modification 2: Track Temporal Progression

**Add progression tracking**:
```python
def _compute_progression_score(self, frame_rewards: List[FrameReward]) -> float:
    """How well does the video achieve the task over time?"""
    rewards = [fr.reward for fr in frame_rewards]
    
    # For "erase" task: rewards should increase
    improvements = sum(
        1 for i in range(1, len(rewards)) 
        if rewards[i] > rewards[i-1]
    )
    
    return improvements / (len(rewards) - 1)
```

### Modification 3: Update GRPO Policy Based on Frame Analysis

**Before**:
```python
# Just pick best video
best_video = max(candidates, key=lambda x: x['reward'])
```

**After**:
```python
# Analyze WHERE the video fails frame-by-frame
best_candidate = max(candidates, key=lambda x: x['total_reward'])
frame_rewards = best_candidate['frame_rewards']

# Identify problematic regions
problem_frames = [fr for fr in frame_rewards if fr.reward < 0.5]

if len(problem_frames) > 0:
    # Problem is in specific frames - might need more steps
    config['num_inference_steps'] += 10
elif temporal_consistency < 0.7:
    # Temporal jitter - increase guidance
    config['guidance_scale'] += 1.0
```

---

## 🎯 Summary

### What You Learned

1. **LTX-Video Pipeline**: 
   - Denoising loop operates in latent space
   - All frames generated simultaneously
   - VAE decodes full sequence at once

2. **Full Sequence GRPO**:
   - Evaluates **every frame** individually
   - Computes temporal consistency
   - Tracks task progression over time
   - Provides detailed debugging information

3. **Implementation**:
   - `grpo_full_sequence.py`: Core framework
   - `ltx_grpo_integration.py`: LTX-Video hooks
   - `grpo_full_sequence_example.py`: Complete working example

### Key Takeaways

✅ **Frame-level granularity** >> Video-level scores  
✅ **Temporal consistency** matters for video quality  
✅ **Progression tracking** enables task-specific optimization  
✅ **Detailed feedback** enables better parameter updates  

### Next Steps

1. Run the example: `python grpo_full_sequence_example.py`
2. Customize the per-frame reward for your task
3. Experiment with different sequence-level metrics
4. Visualize frame-by-frame rewards to debug
5. Iterate on parameter update strategy

---

## 📚 Additional Resources

- **LTX-Video Paper**: Understanding the transformer architecture
- **GRPO Paper**: Group Relative Policy Optimization principles
- **Diffusion Models**: How denoising works in latent space

Happy optimizing! 🚀

