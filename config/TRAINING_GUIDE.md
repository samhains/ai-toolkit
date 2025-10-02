# Eagle Dataset Training Guide

## Progressive Training Approach for 6,700 Images on RTX 6000 (96GB VRAM)

This guide outlines a three-phase approach to training a Flux LoRA on your eagle dataset.

---

## Phase 1: Quick Validation (512px) - ~2-3 hours

**Config:** `eagle_flux_phase1_512.yaml`

**Purpose:**
- Fast iteration to validate dataset quality
- Test caption effectiveness
- Find optimal learning rate
- Identify any data issues early

**Settings:**
- Resolution: 512px
- Batch size: 8
- LoRA rank: 16
- Steps: 5,000 (~6 epochs)
- LR: 1e-4

**What to check:**
- Are samples showing eagle features by step 1000?
- Are captions being followed?
- Any obvious dataset issues (corrupted images, bad captions)?
- Quality progression across steps

**Run:**
```bash
python run.py config/eagle_flux_phase1_512.yaml
```

**Next steps:**
- If quality is poor → check captions, adjust learning rate
- If quality is good → proceed to Phase 2
- If overfitting early → increase caption dropout or reduce steps

---

## Phase 2: Refinement (768px) - ~5-6 hours

**Config:** `eagle_flux_phase2_768.yaml`

**Purpose:**
- Improve detail capture
- Higher resolution for feather textures
- Increased rank for better capacity

**Settings:**
- Resolution: 768px
- Batch size: 4
- LoRA rank: 32 (2x Phase 1)
- Steps: 5,000 (~3 epochs)
- LR: 8e-5 (slightly lower for stability)

**What to check:**
- Better detail in feathers, eyes, textures
- Maintained general eagle characteristics from Phase 1
- No degradation in composition quality

**Run:**
```bash
python run.py config/eagle_flux_phase2_768.yaml
```

**Next steps:**
- Compare samples against Phase 1
- If detail improvement is minimal → might not need Phase 3
- If quality is excellent → proceed to Phase 3 for production model

---

## Phase 3: Final Production (1024px) - ~12-15 hours

**Config:** `eagle_flux_phase3_1024.yaml`

**Purpose:**
- Maximum detail preservation
- Production-ready model
- Publication quality

**Settings:**
- Resolution: 1024px
- Batch size: 2 (gradient accumulation = 2, effective batch size 4)
- LoRA rank: 32
- Steps: 8,000 (~2.4 epochs)
- LR: 8e-5

**What to check:**
- Exceptional feather detail
- Sharp eye details
- Clean high-resolution generation
- No artifacts at high res

**Run:**
```bash
python run.py config/eagle_flux_phase3_1024.yaml
```

---

## Optimization Tips

### For your 96GB RTX 6000:

**If you want faster training:**
- Increase batch size (you have VRAM to spare)
- Phase 1: Try batch_size 12-16
- Phase 2: Try batch_size 6-8
- Phase 3: Try batch_size 4

**If you want better quality:**
- Increase rank to 64 (uses more VRAM)
- Reduce caption_dropout_rate to 0.02
- Add more sample prompts to monitor different scenarios

**If you want to save VRAM:**
- These configs are already conservative for your hardware
- You're unlikely to need VRAM optimization

### Learning Rate Adjustments:

If Phase 1 samples show:
- **Too slow learning** (step 1000 looks random): Increase LR to 2e-4
- **Too fast/unstable** (artifacts, chaos): Decrease LR to 5e-5
- **Just right** (gradual improvement): Keep 1e-4

### Caption Quality:

Monitor your first few samples. If captions aren't being followed:
1. Check your `.txt` files are properly formatted
2. Reduce `caption_dropout_rate` to 0
3. Consider adding a trigger word

---

## Dataset Sync Workflow

Before each training phase, sync your dataset:

```bash
# Initial download
python sync_eagle_dataset.py ./datasets/eagle

# Before Phase 2 or 3, resync to get any new images
python sync_eagle_dataset.py ./datasets/eagle
```

The sync script tracks what's downloaded and only updates changed/new images.

---

## Expected Timeline

| Phase | Resolution | Time | Purpose |
|-------|-----------|------|---------|
| Phase 1 | 512px | 2-3h | Validation |
| Phase 2 | 768px | 5-6h | Refinement |
| Phase 3 | 1024px | 12-15h | Production |
| **Total** | | **~20h** | Complete progression |

---

## Output Location

All models will be saved to:
```
output/eagle_flux_phase1_512/
output/eagle_flux_phase2_768/
output/eagle_flux_phase3_1024/
```

Each folder contains:
- `samples/` - Generated samples during training
- Model checkpoints every 500 steps
- Only the last 5-6 checkpoints are kept

---

## Customization

### Add a trigger word:
Uncomment this line in the config:
```yaml
trigger_word: "eagle_style"
```

### Change sample prompts:
Edit the `sample.prompts` section to match your use case

### Push to Hugging Face:
In Phase 3 config, uncomment:
```yaml
push_to_hub: true
hf_repo_id: "your-username/eagle-flux-lora"
```

---

## Monitoring Training

Watch the `samples/` folder during training to see:
- Step 0: Pre-training baseline (usually random)
- Step 500, 1000, etc: Progressive improvement
- Look for overfitting (samples become too similar)

Good signs:
- Gradual improvement in eagle features
- Variety in poses/angles maintained
- Captions are followed

Bad signs:
- Samples look identical
- Loss of diversity
- Artifacts or distortions

---

## Questions?

- **Training is too slow**: Increase batch size (you have 96GB VRAM!)
- **Not enough detail**: Move to next phase or increase rank
- **Overfitting**: Reduce steps or increase caption_dropout_rate
- **Bad samples**: Check dataset quality with `ls datasets/eagle/*.txt | head`
