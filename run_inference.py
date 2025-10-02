#!/usr/bin/env python3
"""
Simple inference script for testing checkpoints
Direct implementation without using job system
"""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ['DISABLE_TELEMETRY'] = 'YES'

import sys
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.getcwd())

import argparse
import yaml
import torch
from toolkit.config_modules import ModelConfig, GenerateImageConfig
from toolkit.train_tools import get_torch_dtype
from toolkit.util.get_model import get_model_class

def main():
    parser = argparse.ArgumentParser(description='Run inference on model checkpoints')
    parser.add_argument(
        'config',
        type=str,
        help='Path to inference config YAML file'
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
    output_folder = process_config['output_folder']
    model_config = ModelConfig(**process_config['model'])
    gen_config = process_config['generate']

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Get model class and initialize
    ModelClass = get_model_class(model_config)

    # Get sampler
    if hasattr(ModelClass, 'get_train_scheduler'):
        sampler = ModelClass.get_train_scheduler()
    else:
        from toolkit.sampler import get_sampler
        arch = 'sd'
        if model_config.is_pixart:
            arch = 'pixart'
        if model_config.is_flux:
            arch = 'flux'
        if model_config.is_lumina2:
            arch = 'lumina2'
        sampler = get_sampler(
            gen_config.get('sampler', 'flowmatch'),
            {
                "prediction_type": "v_prediction" if model_config.is_v_pred else "epsilon",
            },
            arch=arch,
        )

    # Initialize model
    sd = ModelClass(
        device=device,
        model_config=model_config,
        dtype=dtype,
        noise_scheduler=sampler,
    )

    print(f"Using device {device}")

    # Load model
    print("Loading model...")
    sd.load_model()

    # Get prompts
    prompts = gen_config['prompts']
    if isinstance(prompts, str):
        if os.path.exists(prompts):
            with open(prompts, 'r') as f:
                prompts = [p.strip() for p in f.readlines() if p.strip()]

    # Generate images
    width = gen_config.get('width', 1024)
    height = gen_config.get('height', 1024)
    sample_steps = gen_config.get('sample_steps', 25)
    guidance_scale = gen_config.get('guidance_scale', 4.0)
    seed = gen_config.get('seed', 42)
    ext = gen_config.get('ext', 'png')

    for i, prompt in enumerate(prompts):
        print(f"\nGenerating image {i+1}/{len(prompts)}: {prompt}")

        # Create image config
        img_config = GenerateImageConfig(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=sample_steps,
            guidance_scale=guidance_scale,
            negative_prompt=gen_config.get('neg', ''),
            seed=seed if seed < 0 else seed + i,
            output_folder=output_folder,
            output_ext=ext,
        )

        # Generate
        sd.generate_images([img_config], sampler=gen_config.get('sampler', 'flowmatch'))

    print("\nInference complete!")

if __name__ == '__main__':
    main()
