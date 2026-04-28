# Flux.2 LoRA Training Guide

## Hardware

- **RTX 5090** (32GB) — cuda:0 — good for training
- **RTX PRO 6000** (96GB) — cuda:1 — can train at higher resolutions or run ComfyUI simultaneously

## Models Available

| Model | Path | Size |
|-------|------|------|
| FLUX.1-dev | HF cache (`black-forest-labs/FLUX.1-dev`) | 23GB |
| FLUX.1-dev (safetensors) | `~/Code/ComfyUI/models/diffusion_models/flux1-dev.safetensors` | 23GB |
| FLUX.1-dev fp8 | `~/Code/ComfyUI/models/checkpoints/flux1-dev-fp8.safetensors` | - |
| FLUX.1-Kontext-dev | HF cache | - |
| FLUX Redux | `~/Code/ComfyUI/models/style_models/flux1-redux-dev.safetensors` | - |

## Quick Start

```bash
cd ~/Code/ai-toolkit
source venv/bin/activate

# Run training
python run.py config/my_training_job.yaml
```

## Training Config Template

Save as `config/<project_name>.yaml`:

```yaml
---
job: extension
config:
  name: "my_style_lora_v1"
  process:
    - type: 'sd_trainer'
      training_folder: "output"
      device: cuda:0  # or cuda:1 for PRO 6000
      trigger_word: "mystyle"
      network:
        type: "lora"
        linear: 16        # rank — 8 for subtle, 16 standard, 32 for complex styles
        linear_alpha: 16   # usually same as linear
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
        push_to_hub: false
      datasets:
        - folder_path: "/path/to/captioned/images"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [512, 768, 1024]
      train:
        batch_size: 1
        steps: 2000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        ema_config:
          use_ema: true
          ema_decay: 0.99
        dtype: bf16
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true  # 8bit mixed precision — essential for ≤32GB
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        prompts:
          - "a woman with red hair, playing chess at the park, mystyle"
          - "a bear building a log cabin in the snow, mystyle"
          - "underwater city at sunset, mystyle"
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 20
meta:
  name: "[name]"
  version: '1.0'
```

## Key Parameters

### LoRA Rank (`linear`)
- **8** — lightweight, subtle style transfer, fast training
- **16** — standard, good balance (recommended starting point)
- **32** — high capacity, complex styles, slower training
- **64** — maximum detail, risk of overfitting with small datasets

### Steps
- **500-1000** — quick test, light style influence
- **1500-2500** — standard training (recommended)
- **3000-4000** — deep style embedding, risk of overfitting
- Rule of thumb: ~100 steps per image for small datasets

### Learning Rate
- **1e-4** — standard (recommended)
- **5e-5** — more conservative, less risk of overfitting
- **2e-4** — aggressive, faster convergence but watch for artifacts

### Resolution
- `[512, 768, 1024]` — multi-resolution bucketing (recommended)
- Higher resolutions need more VRAM
- On 32GB (5090): up to 1024 with quantize=true
- On 96GB (PRO 6000): can go higher, or skip quantization

### Device Selection
- `cuda:0` — RTX PRO 6000 (96GB) — more headroom
- `cuda:1` — RTX 5090 (32GB) — works fine with quantize=true
- Can train on one GPU while ComfyUI runs on the other

## Dataset Preparation

### Image Requirements
- **Format:** JPG, JPEG, PNG
- **Minimum:** 15-20 images, sweet spot 20-30
- **Resolution:** At least 1024x1024 recommended
- **Variety:** Different subjects with consistent style

### Captioning
Each image needs a matching `.txt` caption file:
```
images/
├── img_001.jpg
├── img_001.txt
├── img_002.png
├── img_002.txt
```

**Caption strategy for style LoRAs:**
- Describe CONTENT only (subjects, composition, lighting, colors)
- Do NOT describe artistic style — the LoRA learns that through the trigger word
- The trigger word gets prepended automatically if `trigger_word` is set

**Auto-captioning available via:**
- `scripts/style_lora_pipeline.py` — uses Qwen2.5-VL-7B locally
- `~/Code/data-intelligence/scripts/image_captioner_mcp.py` — uses Gemini via MCP

## Automated Pipeline

The full pipeline (`scripts/style_lora_pipeline.py`) accepts a JSON job:

```json
{
    "project_name": "client_style_v1",
    "trigger_word": "clientstyle",
    "image_urls": ["https://...", "https://..."],
    "options": {
        "steps": 2000,
        "lora_rank": 16,
        "device": "cuda:0"
    }
}
```

Runs: Download → Caption → Train → Sample outputs

```bash
cd ~/Code/ai-toolkit
source venv/bin/activate
python scripts/style_lora_pipeline.py job.json
```

## Output

Training produces:
- `output/<name>/<name>.safetensors` — final LoRA weights
- `output/<name>/<name>_step_*.safetensors` — intermediate checkpoints
- `output/<name>/samples/` — sample images generated during training
- `output/<name>/config.yaml` — training config used

### Using the LoRA
Copy the `.safetensors` file to `~/Code/ComfyUI/models/loras/` and use in workflows.

**Recommended LoRA weight:** 0.7-1.0

## Troubleshooting

### OOM on 32GB GPU
- Ensure `quantize: true` in model config
- Reduce resolution to `[512, 768]`
- Set `batch_size: 1`
- Enable `gradient_checkpointing: true`

### Overfitting
- Reduce steps
- Lower learning rate
- Add more training images
- Increase `caption_dropout_rate` to 0.1

### Style not transferring
- Increase rank to 32
- Train more steps
- Check captions aren't describing the style (should be content-only)
- Ensure trigger word is being used in sample prompts

## Previous Training Runs

Found in `output/`:
- `eagle_flux_phase1_512` — Eagle image dataset training
- `eagle_images_2`, `eagle_images_3` — iterations
- Various test runs (`test_3` through `test_7`)
