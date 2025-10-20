# GRPO Full Sequence - Complete Index

## 📑 Quick Navigation

### 🚀 Getting Started (Read First!)

1. **[SUMMARY_GRPO_WALKTHROUGH.md](SUMMARY_GRPO_WALKTHROUGH.md)** ⭐ START HERE
   - Overview of everything
   - What you asked for vs what you got
   - Quick examples
   - **Recommended first read**

2. **[README_GRPO_FULL_SEQUENCE.md](README_GRPO_FULL_SEQUENCE.md)**
   - Quick start guide
   - Usage examples
   - FAQ
   - **For getting started quickly**

### 📚 Deep Documentation

3. **[GRPO_FULL_SEQUENCE_GUIDE.md](GRPO_FULL_SEQUENCE_GUIDE.md)**
   - Complete 20+ page guide
   - LTX-Video pipeline walkthrough
   - Implementation details
   - Advanced customization
   - **For comprehensive understanding**

### 💻 Code Files

4. **[grpo_full_sequence.py](grpo_full_sequence.py)**
   - Core framework
   - `FullSequenceRewardCalculator` class
   - `VideoGRPOWithFullSequence` class
   - Generic, reusable code

5. **[ltx_grpo_integration.py](ltx_grpo_integration.py)**
   - LTX-Video specific integration
   - `LTXVideoGRPOGenerator` class
   - `GRPOSearchWithLTXVideo` class
   - Hooks into inference pipeline

6. **[grpo_full_sequence_example.py](grpo_full_sequence_example.py)** ⭐ RUN THIS
   - Complete working example
   - Red dot task implementation
   - Command-line interface
   - **Executable demo**

7. **[visualize_grpo_comparison.py](visualize_grpo_comparison.py)**
   - Creates comparison diagrams
   - Temporal analysis plots
   - Architecture diagrams
   - Run to generate visualizations

### 📁 Original Files (Your Existing Code)

- `ltx_video_source/ltx_video/inference.py` - Main inference pipeline
- `grpo_generate.sh` - Your original GRPO script
- `red_dot_grpo.py` - Your red dot reward implementation
- `video_grpo_framework.py` - Your GRPO framework

## 🎯 Reading Order by Goal

### Goal 1: "I just want to run it"

```
1. README_GRPO_FULL_SEQUENCE.md (Quick Start section)
2. Run: python grpo_full_sequence_example.py
3. Check outputs/
```

### Goal 2: "I want to understand how it works"

```
1. SUMMARY_GRPO_WALKTHROUGH.md
2. GRPO_FULL_SEQUENCE_GUIDE.md (Architecture sections)
3. Read grpo_full_sequence.py (with comments)
4. Run visualizations: python visualize_grpo_comparison.py
```

### Goal 3: "I want to customize for my task"

```
1. README_GRPO_FULL_SEQUENCE.md (Customization Examples)
2. grpo_full_sequence_example.py (study RedDotFullSequenceReward)
3. Modify per_frame_reward() for your task
4. Test with small num_rounds first
```

### Goal 4: "I want to integrate with my existing code"

```
1. SUMMARY_GRPO_WALKTHROUGH.md (Integration section)
2. Look at your red_dot_grpo.py
3. Enhance with FullSequenceRewardCalculator
4. Use GRPOSearchWithLTXVideo for search
```

## 📊 File Dependency Graph

```
grpo_full_sequence.py
    ↓ (used by)
ltx_grpo_integration.py
    ↓ (used by)
grpo_full_sequence_example.py
    ↓ (uses)
ltx_video_source/ltx_video/inference.py
```

## 🔑 Key Concepts by File

### SUMMARY_GRPO_WALKTHROUGH.md
- ✅ Overall understanding
- ✅ Before/after comparison
- ✅ Quick integration guide

### GRPO_FULL_SEQUENCE_GUIDE.md
- ✅ LTX-Video pipeline deep dive
- ✅ Denoising loop explanation
- ✅ Full sequence architecture
- ✅ Modification strategies

### grpo_full_sequence.py
- ✅ `FrameReward` dataclass
- ✅ `SequenceReward` dataclass
- ✅ `FullSequenceRewardCalculator`
- ✅ Per-frame reward computation
- ✅ Temporal consistency
- ✅ Progression tracking

### ltx_grpo_integration.py
- ✅ `LTXVideoGRPOGenerator`
- ✅ `GRPOSearchWithLTXVideo`
- ✅ Pipeline integration
- ✅ Parameter search

### grpo_full_sequence_example.py
- ✅ `RedDotFullSequenceReward`
- ✅ Complete workflow example
- ✅ Command-line interface
- ✅ Visualization of results

## 📈 Learning Path

### Beginner (Never used GRPO before)

```
Day 1:
  - Read README_GRPO_FULL_SEQUENCE.md
  - Run grpo_full_sequence_example.py with default params
  - Run visualize_grpo_comparison.py

Day 2:
  - Read SUMMARY_GRPO_WALKTHROUGH.md
  - Understand the output from Day 1
  - Experiment with different prompts

Day 3:
  - Read relevant sections of GRPO_FULL_SEQUENCE_GUIDE.md
  - Modify per_frame_reward for simple task
  - Run again and compare results
```

### Intermediate (Used GRPO, want full sequence)

```
Step 1: Read SUMMARY_GRPO_WALKTHROUGH.md (30 min)
Step 2: Look at your existing reward function
Step 3: Refactor to per-frame + sequence (grpo_full_sequence_example.py as reference)
Step 4: Test with 1 round, 2 candidates first
Step 5: Scale up to full search
```

### Advanced (Want to customize deeply)

```
1. Read GRPO_FULL_SEQUENCE_GUIDE.md fully
2. Study ltx_grpo_integration.py implementation
3. Read ltx_video_source/ltx_video/pipelines/pipeline_ltx_video.py
4. Implement custom:
   - per_frame_reward_fn
   - sequence_reward_fn
   - temporal_consistency_fn
5. Customize parameter update strategy in _update_config
```

## 🎨 Visualization Files Generated

After running `visualize_grpo_comparison.py`:

```
outputs/
  ├── grpo_comparison.png         # Old vs New GRPO
  ├── temporal_analysis.png       # Frame-level analysis
  └── architecture_diagram.png    # System architecture
```

## 🔧 Configuration Files

- `ltx_video_source/configs/ltxv-2b-0.9.8-distilled.yaml` - Model config
- Command-line args in `grpo_full_sequence_example.py`

## 📝 Usage Patterns

### Pattern 1: Quick Test

```bash
python grpo_full_sequence_example.py \
    --prompt "A red dot fades away" \
    --num_rounds 1 \
    --candidates_per_round 2
```

### Pattern 2: Full Search

```bash
python grpo_full_sequence_example.py \
    --prompt "Erase the red dots" \
    --original_image data/red_dots.png \
    --num_rounds 4 \
    --candidates_per_round 6 \
    --output_dir outputs/red_dot_full_search
```

### Pattern 3: Custom Integration

```python
from grpo_full_sequence import FullSequenceRewardCalculator
from ltx_grpo_integration import GRPOSearchWithLTXVideo

# Your custom code here
```

## 🐛 Debugging Guide

### Problem: "I don't understand the output"

→ Read: SUMMARY_GRPO_WALKTHROUGH.md, section "What Full Sequence Analysis Provides"

### Problem: "Rewards are all low"

→ Check:
1. Is your per_frame_reward function working?
2. Print sample frame rewards to debug
3. Visualize with matplotlib (see README examples)

### Problem: "Integration not working"

→ Check:
1. Are you passing video in correct shape? [1, C, T, H, W]
2. Is per_frame_reward function signature correct?
3. See ltx_grpo_integration.py for examples

### Problem: "Want to modify parameters"

→ See: GRPO_FULL_SEQUENCE_GUIDE.md, section "Key Modifications to Make"

## 💬 Common Questions

**Q: Which file do I run?**  
A: `grpo_full_sequence_example.py`

**Q: Which file do I read first?**  
A: `SUMMARY_GRPO_WALKTHROUGH.md`

**Q: How do I customize rewards?**  
A: Modify the `per_frame_reward()` function (examples in README)

**Q: Where is the LTX-Video pipeline explained?**  
A: `GRPO_FULL_SEQUENCE_GUIDE.md`, section "LTX-Video Inference Pipeline Overview"

**Q: How do I integrate with my code?**  
A: `SUMMARY_GRPO_WALKTHROUGH.md`, section "Integration with Your Existing Code"

## 📚 Additional Resources

### External Documentation
- LTX-Video: Check `ltx_video_source/README.md` (if exists)
- GRPO Paper: Search for "Group Relative Policy Optimization"
- Diffusion Models: Background on denoising process

### Internal Code References
- Transformer: `ltx_video_source/ltx_video/models/transformers/transformer3d.py`
- VAE: `ltx_video_source/ltx_video/models/autoencoders/causal_video_autoencoder.py`
- Pipeline: `ltx_video_source/ltx_video/pipelines/pipeline_ltx_video.py`

## ✅ Checklist for Success

Before you start:
- [ ] Read SUMMARY_GRPO_WALKTHROUGH.md
- [ ] Understand the difference between old and new GRPO
- [ ] Know what "full sequence" means (121 rewards vs 1)

To run the example:
- [ ] Have LTX-Video weights downloaded
- [ ] CUDA available (or modify to use CPU)
- [ ] Run: `python grpo_full_sequence_example.py`

To customize:
- [ ] Understand your task's reward criteria
- [ ] Write per_frame_reward function
- [ ] Test with 1 round first
- [ ] Visualize results

To integrate:
- [ ] Import FullSequenceRewardCalculator
- [ ] Import GRPOSearchWithLTXVideo
- [ ] Define your reward functions
- [ ] Run search

## 🎉 You're All Set!

Everything you need is here. Start with `SUMMARY_GRPO_WALKTHROUGH.md` and go from there!

---

**Questions?** Check the FAQ sections in:
- README_GRPO_FULL_SEQUENCE.md
- GRPO_FULL_SEQUENCE_GUIDE.md

**Issues?** See debugging guide above.

**Ready?** Run: `python grpo_full_sequence_example.py`

Good luck! 🚀

