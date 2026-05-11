# Dreambook Identity Edit Training Configs

This folder holds local training smoke configs for the Dreambook masked
identity-edit method.

Current config:

```text
qwen_image_edit_2509_nano_native_smoke_20260511.yaml
```

Preflight:

```bash
python3 config/dreambook_identity_edit/validate_smoke_readiness.py
```

This checks the dataset/control layout, sample control files, and whether the
configured model/adapters are already available locally. It does not import
torch, touch a GPU, start training, or download models.

Purpose:

- Test whether a local multi-control edit model can learn the shape of the
  Dreambook workflow: source scene, mask, identity reference, and edited target.
- Use the 9-row Nano Native export as teacher/comparator data.
- Keep this as a smoke config only until every target row is visually curated.

Dataset export:

```text
/home/excalibur/media/dreambook-assets/mine-dreams-paperback-visual-v2-20260504/identity-method-dev/training-exports/nano-native-qwen-edit-20260511
```

Mapping:

```yaml
folder_path: targets
control_path:
  - controls/source
  - controls/mask
  - controls/identity
caption_ext: txt
```

Do not treat this as proof that local training is the right production path.
It is a small, reproducible experiment to compare against the current
Nano-native-plus-Pabrix workflow.

2026-05-11 readiness result:

- Dataset export is complete: 9 targets, 9 captions, and 27 matched controls.
- The three configured sample controls exist.
- The smoke is **not locally ready** because `Qwen/Qwen-Image-Edit-2509` and
  `ostris/accuracy_recovery_adapters` were not found in the local Hugging Face,
  storage, or ComfyUI model caches.
- The next gate is an explicit model/adaptor fetch or a decision to test a
  different local edit architecture that is already present.
