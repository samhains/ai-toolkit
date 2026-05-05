# Character Identity Configs

Pilot full-FLUX.2-dev identity LoRA configs:

```bash
cd /home/excalibur/Code/ai-toolkit-updated
source venv/bin/activate
python run.py config/character_identity/sam_vornak_identity_flux2_dev_pilot_20260505.yaml
python run.py config/character_identity/harry_mirlon_identity_flux2_dev_pilot_20260505.yaml
```

These use the same Sam and Harry datasets/triggers as the Klein identity LoRAs, but switch to:

- `model.name_or_path: black-forest-labs/FLUX.2-dev`
- `model.arch: flux2`
- quantized transformer/text encoder with `low_vram: false` for the 2x 96 GB cards
- no convolution LoRA ranks, matching ai-toolkit's full FLUX.2 support

Keep quantization enabled for the first pass. If full FLUX.2-dev still OOMs on
the 96 GB cards, flip `low_vram` back to `true` before lowering rank or batch
size.
