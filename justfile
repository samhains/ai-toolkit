# ai-toolkit justfile

# Run a training job from a config file
train config:
    python run.py {{config}}

# Style LoRA full pipeline: download, caption, train
style-lora job_file:
    python scripts/style_lora_pipeline.py {{job_file}}

# Style LoRA: download and caption only (no training)
caption-dataset job_file:
    python scripts/style_lora_pipeline.py {{job_file}} --caption-only

# Style LoRA: dry run (generates config but doesn't train)
style-lora-dry job_file:
    python scripts/style_lora_pipeline.py {{job_file}} --dry-run

# Show the example job JSON format
style-lora-example:
    cat config/examples/style_lora_job_example.json
