#!/usr/bin/env python3
"""
Upload LoRA checkpoints from output/ to Supabase shelf.

Usage:
    python scripts/upload_loras_to_supabase.py                    # upload all runs
    python scripts/upload_loras_to_supabase.py style_mlhbhtp9     # upload specific run
"""

import os
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT))
from toolkit.supabase_upload import upload_lora, list_existing


def parse_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        processes = config.get("config", {}).get("process", [])
        if processes:
            p = processes[0]
            return {
                "arch": p.get("model", {}).get("arch", "unknown"),
                "total_steps": p.get("train", {}).get("steps", None),
            }
    except Exception:
        pass
    return {"arch": "unknown", "total_steps": None}


def normalize_filename(cp: Path, run_name: str, total_steps: int) -> str:
    """Ensure every file has steps in the name: {run_name}_{steps_9digits}.safetensors"""
    name = cp.stem  # e.g. style_mlhbhtp9 or style_mlhbhtp9_000000600
    if name == run_name:
        # Final checkpoint with no step — add total_steps
        steps = total_steps or 0
        return f"{run_name}_{str(steps).zfill(9)}.safetensors"
    return cp.name


def main():
    output_dir = PROJECT_ROOT / "output"
    if not output_dir.exists():
        print("No output/ directory found.")
        return

    filter_name = sys.argv[1] if len(sys.argv) > 1 else None

    for run_dir in sorted(output_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        if filter_name and name != filter_name:
            continue

        checkpoints = sorted(run_dir.glob(f"{name}*.safetensors"))
        if not checkpoints:
            continue

        cfg = parse_config(run_dir)
        arch = cfg.get("arch", "unknown")
        total_steps = cfg.get("total_steps")

        # Build list of (local_path, normalized_upload_name)
        uploads = []
        existing = list_existing(name, arch)
        for cp in checkpoints:
            upload_name = normalize_filename(cp, name, total_steps)
            if upload_name not in existing:
                uploads.append((cp, upload_name))

        if not uploads:
            print(f"  {name}: all {len(checkpoints)} checkpoints already uploaded, skipping")
            continue

        print(f"\n{name} ({arch}) — {len(uploads)} new / {len(checkpoints)} total")

        for cp, upload_name in uploads:
            upload_lora(str(cp), name, arch, upload_name=upload_name)

    print("\nDone.")


if __name__ == "__main__":
    main()
