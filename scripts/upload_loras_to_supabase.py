#!/usr/bin/env python3
"""
Upload LoRA checkpoints from output/ to Supabase shelf.

Usage:
    python scripts/upload_loras_to_supabase.py                    # upload all final loras
    python scripts/upload_loras_to_supabase.py style_mlhbhtp9     # upload specific run
"""

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load env from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))
from toolkit.supabase_upload import push_to_supabase


def get_model_arch(run_dir: Path) -> str:
    """Extract model arch from the saved config.yaml in the run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return "unknown"
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        processes = config.get("config", {}).get("process", [])
        if processes:
            model = processes[0].get("model", {})
            arch = model.get("arch", None)
            if arch:
                return arch
            # Fallback: derive from name_or_path
            name = model.get("name_or_path", "")
            if "flux" in name.lower():
                return "flux"
            if "wan" in name.lower():
                return "wan"
    except Exception as e:
        print(f"  Warning: could not parse config: {e}")
    return "unknown"


def get_trigger_word(run_dir: Path) -> str:
    """Extract trigger word from config."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        processes = config.get("config", {}).get("process", [])
        if processes:
            return processes[0].get("trigger_word", None)
    except Exception:
        pass
    return None


def get_train_steps(run_dir: Path) -> int:
    """Extract total training steps from config."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        processes = config.get("config", {}).get("process", [])
        if processes:
            return processes[0].get("train", {}).get("steps", None)
    except Exception:
        pass
    return None


def discover_loras(output_dir: Path, filter_name: str = None):
    """Find final (non-step) safetensors files in output subdirectories."""
    loras = []
    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if filter_name and name != filter_name:
            continue
        final = run_dir / f"{name}.safetensors"
        if final.exists():
            loras.append((name, run_dir, final))
    return loras


def main():
    output_dir = PROJECT_ROOT / "output"
    if not output_dir.exists():
        print("No output/ directory found.")
        return

    filter_name = sys.argv[1] if len(sys.argv) > 1 else None

    loras = discover_loras(output_dir, filter_name)
    if not loras:
        target = filter_name or "any"
        print(f"No final lora checkpoints found for: {target}")
        return

    print(f"Found {len(loras)} lora(s) to upload:\n")
    for name, run_dir, final in loras:
        size_mb = final.stat().st_size / 1024 / 1024
        arch = get_model_arch(run_dir)
        trigger = get_trigger_word(run_dir)
        steps = get_train_steps(run_dir)
        print(f"  {name}")
        print(f"    arch: {arch}  trigger: {trigger}  steps: {steps}  size: {size_mb:.1f} MB")
        print(f"    -> loras/{arch}/{name}/{name}.safetensors")
        print()

    for name, run_dir, final in loras:
        arch = get_model_arch(run_dir)
        trigger = get_trigger_word(run_dir)
        steps = get_train_steps(run_dir)

        # Build metadata from config (no training meta OrderedDict available offline)
        from collections import OrderedDict
        import datetime
        meta = OrderedDict({
            "job_name": name,
            "model_arch": arch,
            "trigger_word": trigger,
            "steps": steps,
            "file_size_mb": round(final.stat().st_size / 1024 / 1024, 1),
            "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

        url = push_to_supabase(
            safetensors_path=str(final),
            job_name=name,
            model_arch=arch,
            metadata=None,  # use minimal metadata, we build our own
            bucket="ai-creative-studio-shelf",
        )

        if url:
            # Also upload richer metadata manually
            import json
            from supabase import create_client
            client = create_client(
                os.environ["SUPABASE_URL"],
                os.environ["SUPABASE_KEY"],
            )
            storage = client.storage.from_("ai-creative-studio-shelf")
            meta_path = f"loras/{arch}/{name}/metadata.json"
            meta_json = json.dumps(dict(meta), indent=2)
            try:
                storage.upload(
                    path=meta_path,
                    file=meta_json.encode("utf-8"),
                    file_options={"content-type": "application/json", "upsert": "true"},
                )
            except Exception:
                pass  # already uploaded by push_to_supabase

    print("\nDone.")


if __name__ == "__main__":
    main()
