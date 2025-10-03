#!/usr/bin/env python3
"""
Simple inference script for testing checkpoints
Direct implementation without using job system
"""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ['DISABLE_TELEMETRY'] = 'YES'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.getcwd())

import argparse
import yaml
import torch
import json
import random
import gc
from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.train_tools import get_torch_dtype
from toolkit.util.get_model import get_model_class

def sanitize_prompt_for_filename(prompt, max_length=100):
    """Convert prompt to a safe filename string"""
    # Remove or replace unsafe characters
    safe = prompt.lower()
    # Replace spaces and special chars with underscores
    safe = ''.join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in safe)
    # Replace multiple spaces/underscores with single underscore
    safe = '_'.join(safe.split())
    # Truncate to max length
    if len(safe) > max_length:
        safe = safe[:max_length]
    # Remove trailing underscores
    safe = safe.rstrip('_')
    return safe

def main():
    parser = argparse.ArgumentParser(description='Run inference on model checkpoints')
    parser.add_argument(
        'config',
        type=str,
        help='Path to inference config YAML file'
    )
    parser.add_argument(
        '--output-name',
        type=str,
        default=None,
        help='Override output folder name (will be placed in ./output/[output-name])'
    )

    args = parser.parse_args()

    # Load the config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Get process config (first process in the list)
    process_config = config['config']['process'][0]

    # Parse configuration
    device = process_config.get('device', 'cuda:0')
    dtype = get_torch_dtype(process_config.get('dtype', 'float16'))

    # Use --output-name if provided, otherwise use config value
    if args.output_name:
        base_output_folder = os.path.join('./output', args.output_name)
    else:
        base_output_folder = process_config['output_folder']

    model_config = ModelConfig(**process_config['model'])
    gen_config = process_config['generate']

    # Get LoRA paths from name and steps
    lora_name = process_config['model'].get('lora_name')
    lora_steps = process_config['model'].get('lora_steps', [])

    # Build paths from lora_name and steps
    lora_paths = []
    for step in lora_steps:
        # Format: ./output/{lora_name}/{lora_name}_{step:09d}.safetensors
        lora_path = f"./output/{lora_name}/{lora_name}_{int(step):09d}.safetensors"
        lora_paths.append(lora_path)

    # Validate all files exist before starting
    missing_files = [p for p in lora_paths if not os.path.exists(p)]
    if missing_files:
        print("Error: The following LoRA files were not found:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    # Get prompts once
    prompts = gen_config['prompts']
    if isinstance(prompts, str):
        if os.path.exists(prompts):
            # Check if it's a JSON file
            if prompts.endswith('.json'):
                with open(prompts, 'r') as f:
                    data = json.load(f)
                    # Support both direct array and object with 'prompts' key
                    if isinstance(data, list):
                        prompts = data
                    elif isinstance(data, dict) and 'prompts' in data:
                        prompts = data['prompts']
                    else:
                        raise ValueError(f"JSON file must contain an array or an object with a 'prompts' key")
            else:
                # Plain text file - one prompt per line
                with open(prompts, 'r') as f:
                    prompts = [p.strip() for p in f.readlines() if p.strip()]

    # Handle random prompt selection
    num_random_prompts = gen_config.get('num_random_prompts', None)
    if num_random_prompts is not None and num_random_prompts > 0:
        if len(prompts) > num_random_prompts:
            print(f"\nSelecting {num_random_prompts} random prompts from {len(prompts)} available prompts")
            prompts = random.sample(prompts, num_random_prompts)
        else:
            print(f"\nWarning: Requested {num_random_prompts} random prompts but only {len(prompts)} available. Using all prompts.")

    print(f"\nUsing {len(prompts)} prompts for inference:")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")

    # Generation settings
    width = gen_config.get('width', 1024)
    height = gen_config.get('height', 1024)
    guidance_scale = gen_config.get('guidance_scale', 4.0)
    seed = gen_config.get('seed', -1)
    randomize_seed = gen_config.get('randomize_seed', False)
    ext = gen_config.get('ext', 'png')
    sample_steps = gen_config.get('sample_steps', 25)

    # Process each LoRA checkpoint
    for lora_idx, lora_path in enumerate(lora_paths):
        print(f"\n{'='*60}")
        print(f"Processing LoRA {lora_idx+1}/{len(lora_paths)}")
        print(f"LoRA: {lora_path}")
        print(f"{'='*60}\n")

        # Extract LoRA step from filename for folder naming
        import re
        match = re.search(r'_(\d+)\.safetensors', lora_path)
        lora_step = match.group(1).lstrip('0') or '0' if match else "base"

        # Create output folder with LoRA step
        output_folder = os.path.join(base_output_folder, f'lora_{lora_step}')
        os.makedirs(output_folder, exist_ok=True)

        # Build model config with current LoRA
        model_dict = {k: v for k, v in process_config['model'].items()
                     if k not in ['lora_name', 'lora_steps']}
        model_dict['lora_path'] = lora_path
        current_model_config = ModelConfig(**model_dict)

        # Get model class and sampler
        ModelClass = get_model_class(current_model_config)

        if hasattr(ModelClass, 'get_train_scheduler'):
            sampler = ModelClass.get_train_scheduler()
        else:
            from toolkit.sampler import get_sampler
            arch = 'sd'
            if current_model_config.is_pixart:
                arch = 'pixart'
            if current_model_config.is_flux:
                arch = 'flux'
            if current_model_config.is_lumina2:
                arch = 'lumina2'
            sampler = get_sampler(
                gen_config.get('sampler', 'flowmatch'),
                {
                    "prediction_type": "v_prediction" if current_model_config.is_v_pred else "epsilon",
                },
                arch=arch,
            )

        # Initialize model
        sd = ModelClass(
            device=device,
            model_config=current_model_config,
            dtype=dtype,
            noise_scheduler=sampler,
        )

        print("Loading model with LoRA...")
        sd.load_model()

        # Generate images for all prompts
        for i, prompt in enumerate(prompts):
            print(f"\nGenerating image {i+1}/{len(prompts)}: {prompt}")

            # Determine seed to use
            if randomize_seed:
                current_seed = -1  # Random seed for each image
            else:
                current_seed = seed if seed >= 0 else -1

            # Create filename tail from prompt if enabled
            use_prompt_in_filename = gen_config.get('prompt_in_filename', True)
            output_tail = ''
            if use_prompt_in_filename:
                output_tail = '_' + sanitize_prompt_for_filename(prompt)

            # Create image config
            img_config = GenerateImageConfig(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=sample_steps,
                guidance_scale=guidance_scale,
                negative_prompt=gen_config.get('neg', ''),
                seed=current_seed,
                output_folder=output_folder,
                output_ext=ext,
                output_tail=output_tail,
            )

            # Generate
            sd.generate_images([img_config], sampler=gen_config.get('sampler', 'flowmatch'))

        # Clean up model after this LoRA to free memory for the next one
        print("\nCleaning up model...")

        # Synchronize first to ensure all CUDA operations are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        # Move all model components to CPU before deletion to avoid CUDA errors
        if hasattr(sd, 'pipeline') and sd.pipeline is not None:
            sd.pipeline.to('cpu')

        # Now delete everything
        del sd
        del sampler
        del ModelClass
        del current_model_config

        # Force multiple garbage collections
        gc.collect()
        gc.collect()
        gc.collect()

        # Clear CUDA cache thoroughly
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats(device)

        print(f"Memory freed. GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)

if __name__ == '__main__':
    main()
