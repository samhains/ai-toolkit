# Run inference on checkpoints
inference config="config/inference_example.yaml":
    python run_inference.py {{config}}

# Serve generated images via HTTP on port 8000
serve port="8000":
    cd output/inference_samples && python -m http.server {{port}}
