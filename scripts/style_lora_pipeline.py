#!/usr/bin/env python3
"""
Style LoRA Training Pipeline
============================
End-to-end pipeline: Client JSON (image URLs) -> Download -> Auto-caption -> Train LoRA

Usage:
    python scripts/style_lora_pipeline.py job.json
    python scripts/style_lora_pipeline.py job.json --caption-only
    python scripts/style_lora_pipeline.py job.json --dry-run

The client provides a JSON file like:
{
    "project_name": "my_style_v1",
    "model": "flux_dev",
    "trigger_word": "mystyle",
    "image_urls": ["https://...", "https://..."],
    "options": { "steps": 2000, "lora_rank": 16 }
}
"""

import argparse
import gc
import io
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import torch
import yaml
from PIL import Image, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("style_lora_pipeline")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = REPO_ROOT / "datasets"
CONFIG_GEN_DIR = REPO_ROOT / "config" / "generated"

# --- Caption prompt designed for style LoRAs ---
# Describes content only; the trigger word + LoRA learns the style implicitly.
STYLE_CAPTION_PROMPT = (
    "Describe the content of this image in detail: the subjects, their appearance, "
    "poses, the setting, composition, lighting, and colors. "
    "Do NOT describe the artistic style, medium, or aesthetic qualities. "
    "Focus only on what is depicted."
)

# Default sample prompts when none are provided
DEFAULT_SAMPLE_PROMPTS = [
    "woman with red hair, playing chess at the park, bomb going off in the background",
    "a woman holding a coffee cup, in a beanie, sitting at a cafe",
    "a horse is a DJ at a night club, fish eye lens, smoke machine, lazer lights, holding a martini",
    "a man showing off his cool new t shirt at the beach, a shark is jumping out of the water in the background",
    "a bear building a log cabin in the snow covered mountains",
    "woman playing the guitar, on stage, singing a song, laser lights, punk rocker",
    "hipster man with a beard, building a chair, in a wood shop",
    "photo of a man, white background, medium shot, modeling clothing, studio lighting, white backdrop",
]


# ============================================================================
# Image Downloader
# ============================================================================

def download_images(image_urls: List[str], output_dir: Path) -> List[Path]:
    """Download images from URLs, validate, and save as JPG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    for i, url in enumerate(image_urls, 1):
        try:
            logger.info(f"[{i}/{len(image_urls)}] Downloading: {url[:100]}...")
            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()

            img = Image.open(io.BytesIO(resp.content))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")

            # Validate minimum size
            w, h = img.size
            if w < 256 or h < 256:
                logger.warning(f"  Skipping: too small ({w}x{h}), minimum 256x256")
                continue

            # Resize if oversized (keep aspect ratio, max 2048px)
            if w > 2048 or h > 2048:
                scale = min(2048 / w, 2048 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.info(f"  Resized {w}x{h} -> {new_w}x{new_h}")

            # Save as JPG
            filename = f"img_{i:04d}.jpg"
            save_path = output_dir / filename
            img.save(save_path, "JPEG", quality=95)
            downloaded.append(save_path)
            logger.info(f"  Saved: {filename} ({img.size[0]}x{img.size[1]})")

        except Exception as e:
            logger.error(f"  Failed to download {url[:80]}: {e}")
            continue

    logger.info(f"Downloaded {len(downloaded)}/{len(image_urls)} images")
    return downloaded


# ============================================================================
# Auto-Captioner (Qwen2.5-VL-7B)
# ============================================================================

class StyleCaptioner:
    """Image captioner using Qwen2.5-VL-7B, adapted for style LoRA training."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:1",
        inference_timeout: int = 180,
        max_retries: int = 3,
    ):
        self.model_path = model_path
        self.device = device
        self.inference_timeout = inference_timeout
        self.max_retries = max_retries
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is not None:
            return
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        logger.info(f"Loading {self.model_path} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
            attn_implementation="eager",
        )
        logger.info("Captioning model loaded")

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Captioning model unloaded, VRAM freed")

    def _resize_for_inference(self, image: Image.Image) -> Image.Image:
        """Resize to max 1024x1024 for captioning inference."""
        w, h = image.size
        if w <= 1024 and h <= 1024:
            return image
        scale = min(1024 / w, 1024 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.LANCZOS)

    def caption_image(self, image_path: Path, prompt: str = STYLE_CAPTION_PROMPT) -> str:
        """Generate a caption for a single image with retry logic."""
        from qwen_vl_utils import process_vision_info

        self.load_model()

        img = Image.open(image_path).convert("RGB")
        img = self._resize_for_inference(img)

        # Save temp resized image for the model
        temp_path = image_path.parent / f".tmp_caption_{image_path.stem}.jpg"
        img.save(temp_path, "JPEG", quality=95)

        for attempt in range(self.max_retries):
            try:
                torch.cuda.empty_cache()

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": f"file://{temp_path}"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {
                    k: v.to(self.model.device) if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }

                with torch.inference_mode():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        temperature=0.7,
                        repetition_penalty=1.1,
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )

                generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
                caption = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                return caption

            except Exception as e:
                logger.error(f"Caption attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.info(f"Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
            finally:
                temp_path.unlink(missing_ok=True)

    def caption_dataset(
        self,
        image_paths: List[Path],
        trigger_word: str,
        force_recaption: bool = False,
    ) -> int:
        """Caption all images in the dataset, saving .txt files alongside images."""
        captioned = 0

        for i, img_path in enumerate(image_paths, 1):
            txt_path = img_path.with_suffix(".txt")

            if txt_path.exists() and not force_recaption:
                logger.info(f"[{i}/{len(image_paths)}] Already captioned: {img_path.name}")
                captioned += 1
                continue

            try:
                logger.info(f"[{i}/{len(image_paths)}] Captioning: {img_path.name}")
                caption = self.caption_image(img_path)

                # Prepend trigger word
                full_caption = f"{trigger_word}, {caption}"
                txt_path.write_text(full_caption, encoding="utf-8")
                logger.info(f"  Caption: {full_caption[:120]}...")
                captioned += 1

            except Exception as e:
                logger.error(f"  Failed to caption {img_path.name}: {e}")
                continue

        logger.info(f"Captioned {captioned}/{len(image_paths)} images")
        return captioned


# ============================================================================
# Training Config Generator
# ============================================================================

def generate_training_config(
    project_name: str,
    dataset_path: str,
    model_type: str,
    num_images: int,
    trigger_word: str,
    device_train: str = "cuda:0",
    steps: Optional[int] = None,
    lora_rank: int = 16,
    learning_rate: float = 1e-4,
    caption_dropout_rate: float = 0.05,
    sample_prompts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate a training YAML config for ai-toolkit."""

    # Smart step count: ~60 steps per image, min 1500, max 3000
    if steps is None:
        steps = max(1500, min(num_images * 60, 3000))
    save_every = min(250, steps // 4)
    sample_every = save_every

    # Add trigger word to sample prompts
    prompts = sample_prompts or DEFAULT_SAMPLE_PROMPTS
    prompts_with_trigger = []
    for p in prompts:
        if trigger_word not in p:
            prompts_with_trigger.append(f"{trigger_word}, {p}")
        else:
            prompts_with_trigger.append(p)

    # Base process config
    process = {
        "type": "sd_trainer",
        "training_folder": "output",
        "device": device_train,
        "trigger_word": trigger_word,
        "network": {
            "type": "lora",
            "linear": lora_rank,
            "linear_alpha": lora_rank,
        },
        "save": {
            "dtype": "float16",
            "save_every": save_every,
            "max_step_saves_to_keep": 4,
            "push_to_hub": False,
        },
        "datasets": [
            {
                "folder_path": dataset_path,
                "caption_ext": "txt",
                "caption_dropout_rate": caption_dropout_rate,
                "shuffle_tokens": False,
                "cache_latents_to_disk": True,
                "resolution": [512, 768, 1024],
            }
        ],
        "train": {
            "batch_size": 1,
            "steps": steps,
            "gradient_accumulation_steps": 1,
            "train_unet": True,
            "train_text_encoder": False,
            "gradient_checkpointing": True,
            "noise_scheduler": "flowmatch",
            "optimizer": "adamw8bit",
            "lr": learning_rate,
            "ema_config": {
                "use_ema": True,
                "ema_decay": 0.99,
            },
            "dtype": "bf16",
        },
        "sample": {
            "sampler": "flowmatch",
            "sample_every": sample_every,
            "width": 1024,
            "height": 1024,
            "prompts": prompts_with_trigger,
            "neg": "",
            "seed": 42,
            "walk_seed": True,
        },
    }

    # Model-specific settings
    if model_type == "flux_dev":
        process["model"] = {
            "name_or_path": "black-forest-labs/FLUX.1-dev",
            "is_flux": True,
            "quantize": True,
        }
        process["sample"]["guidance_scale"] = 4
        process["sample"]["sample_steps"] = 20

    elif model_type == "flux_schnell":
        process["model"] = {
            "name_or_path": "black-forest-labs/FLUX.1-schnell",
            "assistant_lora_path": "ostris/FLUX.1-schnell-training-adapter",
            "is_flux": True,
            "quantize": True,
        }
        process["sample"]["guidance_scale"] = 1
        process["sample"]["sample_steps"] = 4

    elif model_type == "qwen_image":
        process["model"] = {
            "name_or_path": "Qwen/Qwen-Image",
            "arch": "qwen_image",
            "quantize": True,
            "qtype": "uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors",
            "quantize_te": True,
            "qtype_te": "qfloat8",
            "low_vram": True,
        }
        process["train"]["cache_text_embeddings"] = True
        process["sample"]["guidance_scale"] = 3
        process["sample"]["sample_steps"] = 25

    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "Supported: flux_dev, flux_schnell, qwen_image"
        )

    config = {
        "job": "extension",
        "config": {
            "name": project_name,
            "process": [process],
        },
        "meta": {
            "name": "[name]",
            "version": "1.0",
        },
    }

    return config


# ============================================================================
# Main Pipeline
# ============================================================================

def load_job_file(job_path: str) -> Dict[str, Any]:
    """Load and validate the client job JSON."""
    path = Path(job_path)
    if not path.exists():
        raise FileNotFoundError(f"Job file not found: {job_path}")

    with open(path) as f:
        job = json.load(f)

    # Validate required fields
    required = ["project_name", "model", "trigger_word", "image_urls"]
    for field in required:
        if field not in job:
            raise ValueError(f"Missing required field: {field}")

    if not isinstance(job["image_urls"], list) or len(job["image_urls"]) == 0:
        raise ValueError("image_urls must be a non-empty list")

    valid_models = ["flux_dev", "flux_schnell", "qwen_image"]
    if job["model"] not in valid_models:
        raise ValueError(f"model must be one of {valid_models}, got: {job['model']}")

    return job


def run_pipeline(args):
    """Execute the full style LoRA pipeline."""
    # Load job
    job = load_job_file(args.job_file)
    project_name = job["project_name"]
    model_type = job["model"]
    trigger_word = job["trigger_word"]
    image_urls = job["image_urls"]
    options = job.get("options", {})

    logger.info("=" * 60)
    logger.info(f"Style LoRA Pipeline: {project_name}")
    logger.info(f"Model: {model_type} | Trigger: {trigger_word}")
    logger.info(f"Images: {len(image_urls)} URLs")
    logger.info("=" * 60)

    # Warnings based on best practices
    if len(image_urls) < 10:
        logger.warning(
            f"Only {len(image_urls)} images provided. "
            "Style LoRAs work best with 20-30 images. Consider adding more."
        )
    elif len(image_urls) > 50:
        logger.warning(
            f"{len(image_urls)} images is a lot. "
            "Style LoRAs can lose focus above ~40-50 images. Consider curating."
        )

    # --- Phase 1: Download ---
    dataset_dir = DATASETS_DIR / project_name
    logger.info(f"\n--- Phase 1: Download Images -> {dataset_dir} ---")
    image_paths = download_images(image_urls, dataset_dir)

    if len(image_paths) == 0:
        logger.error("No images downloaded successfully. Aborting.")
        sys.exit(1)

    # --- Phase 2: Caption ---
    logger.info(f"\n--- Phase 2: Auto-Caption ({args.caption_model}) ---")
    captioner = StyleCaptioner(
        model_path=args.caption_model,
        device=args.device_caption,
    )
    try:
        captioner.caption_dataset(
            image_paths,
            trigger_word=trigger_word,
            force_recaption=args.force_recaption,
        )
    finally:
        captioner.unload_model()

    if args.caption_only:
        logger.info("\n--caption-only flag set. Stopping after captioning.")
        logger.info(f"Dataset ready at: {dataset_dir}")
        return

    # --- Phase 3: Generate Training Config ---
    logger.info("\n--- Phase 3: Generate Training Config ---")
    CONFIG_GEN_DIR.mkdir(parents=True, exist_ok=True)
    config_path = CONFIG_GEN_DIR / f"{project_name}.yaml"

    training_config = generate_training_config(
        project_name=project_name,
        dataset_path=str(dataset_dir),
        model_type=model_type,
        num_images=len(image_paths),
        trigger_word=trigger_word,
        device_train=args.device_train,
        steps=options.get("steps"),
        lora_rank=options.get("lora_rank", 16),
        learning_rate=options.get("learning_rate", 1e-4),
        caption_dropout_rate=options.get("caption_dropout_rate", 0.05),
        sample_prompts=options.get("sample_prompts"),
    )

    with open(config_path, "w") as f:
        yaml.dump(training_config, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Training config saved: {config_path}")
    logger.info(f"  Steps: {training_config['config']['process'][0]['train']['steps']}")
    logger.info(f"  LoRA rank: {training_config['config']['process'][0]['network']['linear']}")
    logger.info(f"  LR: {training_config['config']['process'][0]['train']['lr']}")

    if args.dry_run:
        logger.info("\n--dry-run flag set. Config generated but training not started.")
        logger.info(f"To train manually: python run.py {config_path}")
        return

    # --- Phase 4: Launch Training ---
    logger.info("\n--- Phase 4: Launch LoRA Training ---")
    logger.info(f"Running: python run.py {config_path}")

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "run.py"), str(config_path)],
        cwd=str(REPO_ROOT),
    )

    if result.returncode == 0:
        logger.info("\nTraining completed successfully!")
        logger.info(f"Output: {REPO_ROOT / 'output' / project_name}")
    else:
        logger.error(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Style LoRA Training Pipeline: URLs -> Caption -> Train",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python scripts/style_lora_pipeline.py my_job.json

  # Caption only (no training)
  python scripts/style_lora_pipeline.py my_job.json --caption-only

  # Dry run (download, caption, generate config, but don't train)
  python scripts/style_lora_pipeline.py my_job.json --dry-run

  # Use specific GPUs
  python scripts/style_lora_pipeline.py my_job.json --device-caption cuda:1 --device-train cuda:0
        """,
    )
    parser.add_argument("job_file", help="Path to client job JSON file")
    parser.add_argument(
        "--caption-only",
        action="store_true",
        help="Download and caption images only, skip training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate everything but don't start training",
    )
    parser.add_argument(
        "--device-caption",
        default="cuda:1",
        help="GPU for captioning model (default: cuda:1 = RTX PRO 6000)",
    )
    parser.add_argument(
        "--device-train",
        default="cuda:0",
        help="GPU for training (default: cuda:0 = RTX 5090)",
    )
    parser.add_argument(
        "--caption-model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Captioning model (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--force-recaption",
        action="store_true",
        help="Re-caption images even if .txt files already exist",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
