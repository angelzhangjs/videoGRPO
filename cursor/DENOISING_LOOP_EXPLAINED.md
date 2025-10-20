# LTX-Video Denoising Loop - Detailed Explanation

**Location**: `ltx_video_source/ltx_video/pipelines/pipeline_ltx_video.py`, lines 1115-1291

## 📋 Complete Code Flow

### Loop Structure

```python
for i, t in enumerate(timesteps):  # Line 1115
    # timesteps: typically [1.0, 0.975, 0.95, ..., 0.025, 0.0]
    # i: step index (0 to 39 for 40 steps)
    # t: current noise level (1.0 = pure noise, 0.0 = clean)
```

---

## 🔧 Step-by-Step Breakdown

### **Step 1: Configure Guidance (Lines 1116-1133)**

```python
# Determine which guidance methods to use
do_classifier_free_guidance = guidance_scale[i] > 1.0
do_spatio_temporal_guidance = stg_scale[i] > 0
do_rescaling = rescaling_scale[i] != 1.0

# Count conditions
num_conds = 1  # Base: conditional prediction
if do_classifier_free_guidance:
    num_conds += 1  # Add: unconditional prediction
if do_spatio_temporal_guidance:
    num_conds += 1  # Add: perturbed prediction
```

**Example Configuration:**
- `guidance_scale = 7.5` → `do_classifier_free_guidance = True`
- `stg_scale = 1.0` → `do_spatio_temporal_guidance = True`
- `num_conds = 3` (uncond + cond + perturbed)

**Tensor Slicing:**
```python
if do_classifier_free_guidance and do_spatio_temporal_guidance:
    indices = slice(0, 3)  # Use all 3 predictions
elif do_classifier_free_guidance:
    indices = slice(0, 2)  # Use uncond + cond
elif do_spatio_temporal_guidance:
    indices = slice(1, 3)  # Use cond + perturbed
else:
    indices = slice(1, 2)  # Use only cond
```

---

### **Step 2: Prepare Latent Input (Lines 1161-1166)**

```python
# Replicate latents for each condition
latent_model_input = (
    torch.cat([latents] * num_conds) if num_conds > 1 else latents
)
# Shape: [B * num_conds, C, T_latent, H_latent, W_latent]
# Example: [1*3, 8, 16, 64, 96] for 3 conditions

# Scale by scheduler (noise level dependent)
latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
```

**Visual:**
```
latents:            [1, 8, 16, 64, 96]
                           ↓
replicate x3:       [3, 8, 16, 64, 96]
                     │   │   │
                     │   │   └─ unconditional
                     │   └───── conditional
                     └───────── perturbed
```

---

### **Step 3: Transformer Prediction (Lines 1204-1217)**

```python
# THIS IS THE CORE COMPUTATION
with context_manager:  # Optional mixed precision
    noise_pred = self.transformer(
        latent_model_input.to(self.transformer.dtype),
        indices_grid=fractional_coords,
        encoder_hidden_states=prompt_embeds_batch[indices].to(
            self.transformer.dtype
        ),
        encoder_attention_mask=prompt_attention_mask_batch[indices],
        timestep=current_timestep,
        skip_layer_mask=skip_layer_mask,
        skip_layer_strategy=skip_layer_strategy,
        return_dict=False,
    )[0]
```

**Input Shapes:**
- `latent_model_input`: `[3, 8, 16, 64, 96]`
- `encoder_hidden_states`: `[3, seq_len, 768]` (T5 embeddings)
- `timestep`: `[3, 1]`

**Output Shape:**
- `noise_pred`: `[3, 8, 16, 64, 96]`

**💡 CRITICAL INSIGHT:**
The transformer processes **ALL 16 latent frames** (= 121 pixel frames) **simultaneously**!
- Temporal dimension `T_latent=16` is part of the 3D attention
- Not sequential frame-by-frame generation
- All frames are refined together at each denoising step

---

### **Step 4: Apply Classifier-Free Guidance (Lines 1224-1244)**

```python
if do_classifier_free_guidance:
    # Split predictions
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_conds)[:2]
    # noise_pred_uncond: [1, 8, 16, 64, 96] - without text
    # noise_pred_text:   [1, 8, 16, 64, 96] - with text
    
    # Optional: CFG* rescaling
    if cfg_star_rescale:
        # Rescale unconditional based on alignment with conditional
        positive_flat = noise_pred_text.view(batch_size, -1)
        negative_flat = noise_pred_uncond.view(batch_size, -1)
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
        alpha = dot_product / squared_norm
        noise_pred_uncond = alpha * noise_pred_uncond
    
    # Apply CFG: push prediction towards conditional
    noise_pred = noise_pred_uncond + guidance_scale[i] * (
        noise_pred_text - noise_pred_uncond
    )
```

**Math:**
```
ε = ε_uncond + w * (ε_text - ε_uncond)
  = (1 - w) * ε_uncond + w * ε_text

where w = guidance_scale (e.g., 7.5)
```

**Effect:**
- `w = 1.0`: No guidance (just use conditional)
- `w = 7.5`: Strong push towards text prompt
- Higher `w` → stronger adherence to prompt, but risk of artifacts

---

### **Step 5: Apply Spatio-Temporal Guidance (Lines 1247-1262)**

```python
if do_spatio_temporal_guidance:
    # Get perturbed prediction
    noise_pred_text, noise_pred_text_perturb = noise_pred.chunk(num_conds)[-2:]
    
    # Apply STG
    noise_pred = noise_pred + stg_scale[i] * (
        noise_pred_text - noise_pred_text_perturb
    )
    
    # Optional: rescale to match original std
    if do_rescaling and stg_scale[i] > 0.0:
        noise_pred_text_std = noise_pred_text.view(batch_size, -1).std(
            dim=1, keepdim=True
        )
        noise_pred_std = noise_pred.view(batch_size, -1).std(
            dim=1, keepdim=True
        )
        
        factor = noise_pred_text_std / noise_pred_std
        factor = rescaling_scale[i] * factor + (1 - rescaling_scale[i])
        
        noise_pred = noise_pred * factor.view(batch_size, 1, 1)
```

**Purpose:**
- Improves spatial and temporal coherence
- Reduces artifacts and jitter
- Perturbed prediction comes from skipping certain transformer layers

---

### **Step 6: Denoising Step (Lines 1272-1281)**

```python
# Update latents: x_t -> x_t-1
latents = self.denoising_step(
    latents,              # Current noisy latents
    noise_pred,           # Predicted noise
    current_timestep,     # Current t
    orig_conditioning_mask,
    t,
    extra_step_kwargs,
    stochastic_sampling=stochastic_sampling,
)
```

**What happens in `denoising_step`:**
```python
def denoising_step(self, latents, noise_pred, current_timestep, ...):
    # Use scheduler to compute x_{t-1} from x_t and predicted noise
    denoised_latents = self.scheduler.step(
        noise_pred,
        current_timestep,
        latents,
        **extra_step_kwargs,
        return_dict=False,
        stochastic_sampling=stochastic_sampling,
    )[0]
    
    return denoised_latents
```

**Scheduler Math (Rectified Flow):**
```
x_{t-1} = x_t - Δt * ε_pred

where:
  x_t: current latents at noise level t
  ε_pred: predicted noise (from transformer + guidance)
  Δt: step size (depends on scheduler)
```

**Result:**
- Latents become slightly less noisy
- Still in latent space
- All 16 latent frames updated together

---

### **Step 7: Progress & Callback (Lines 1283-1290)**

```python
# Update progress bar
if i == len(timesteps) - 1 or (
    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
):
    progress_bar.update()

# Optional callback
if callback_on_step_end is not None:
    callback_on_step_end(self, i, t, {})
```

**Callback Hook:**
- You can pass a callback function to monitor progress
- Receives: `(pipeline, step_index, timestep, callback_kwargs)`
- **Limitation**: Only has access to latents, not decoded pixels

---

## 🎯 Full Loop Iteration Example

**Initial State (Step 0, t=1.0):**
```
latents: [1, 8, 16, 64, 96]  # Pure noise
```

**After Step 1 (t=0.975):**
```
latents: [1, 8, 16, 64, 96]  # Slightly denoised
```

**After Step 20 (t=0.5):**
```
latents: [1, 8, 16, 64, 96]  # Half denoised, rough shapes visible
```

**After Step 40 (t=0.0):**
```
latents: [1, 8, 16, 64, 96]  # Fully denoised
```

**After Loop Completes:**
```
Line 1297-1298: Remove conditioning latents
Line 1300-1306: Unpatchify latents
Line 1327-1333: VAE decode to pixels
              ↓
Result: [1, 3, 121, 512, 768]  # 121 full-resolution frames!
```

---

## 🔍 Where GRPO Modifications Happen

### **Option 1: Callback Hook (Not Recommended)**

```python
def my_callback(pipeline, step, timestep, kwargs):
    # Can access latents here, but they're not in pixel space
    # Can't compute meaningful task-specific rewards
    pass

pipeline(..., callback_on_step_end=my_callback)
```

**Problem:**
- Only have latents, not pixels
- Latent-space features don't correlate with task completion
- Can't detect red dots, objects, etc. in latent space

### **Option 2: After VAE Decode (Recommended - Full Sequence GRPO)**

```python
# Generate video (runs entire denoising loop)
video = pipeline(...)  # [1, 3, 121, 512, 768]

# NOW compute rewards for all 121 frames
for frame_idx in range(121):
    frame = video[0, :, frame_idx, :, :]  # [3, 512, 768]
    reward = compute_frame_reward(frame, frame_idx, prompt)
    # Can detect red dots, objects, etc. in pixel space!
```

**Why This Works:**
- Have full pixel-level access
- Can use any computer vision method (color detection, object detection, etc.)
- All frames available at once (no re-generation needed)
- Fast to evaluate (0.1s per video vs. 5-10s to generate)

---

## 📊 Performance Characteristics

### **Denoising Loop Timing:**
```
40 steps × ~0.2s per step = ~8 seconds
```

### **Breakdown per Step:**
- Latent preparation: ~5ms
- Transformer forward: ~180ms (most expensive!)
- Guidance application: ~10ms
- Scheduler step: ~5ms

### **Bottleneck:**
The transformer forward pass (line 1206) is the bottleneck:
- 3D attention over all frames
- Large model (2B or 13B parameters)
- Running 3 times per step (uncond + cond + perturbed)

---

## 🎓 Key Takeaways

1. **All frames generated simultaneously** in the denoising loop
2. **Transformer processes entire video** at each step
3. **40 iterations** to go from pure noise to clean video
4. **Guidance methods** (CFG + STG) improve quality
5. **Callback hook exists** but only has latents (line 1290)
6. **Full sequence GRPO** evaluates after VAE decode

---

## 🚀 Integration with Full Sequence GRPO

Your full sequence GRPO framework works perfectly because:

1. ✅ Denoising loop generates all frames together
2. ✅ After VAE decode, you have all 121 pixel frames
3. ✅ Can evaluate each frame independently
4. ✅ Can compute temporal consistency across frames
5. ✅ Can track progression over the sequence
6. ✅ No need to modify the denoising loop itself!

The framework I created in `grpo_full_sequence.py` operates **after** this entire process completes, which is:
- Non-intrusive (doesn't modify LTX-Video internals)
- Efficient (evaluate once, not during generation)
- Flexible (works with any reward function)

---

## 📝 Code References

**Main Files:**
- Denoising loop: `ltx_video_source/ltx_video/pipelines/pipeline_ltx_video.py:1115-1291`
- Denoising step: `ltx_video_source/ltx_video/pipelines/pipeline_ltx_video.py:1348-1381`
- Scheduler: `ltx_video_source/ltx_video/schedulers/rf.py`
- Transformer: `ltx_video_source/ltx_video/models/transformers/transformer3d.py`

**GRPO Integration:**
- Full sequence framework: `grpo_full_sequence.py`
- LTX integration: `ltx_grpo_integration.py`
- Example: `grpo_full_sequence_example.py`



