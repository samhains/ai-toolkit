#!/usr/bin/env python3
"""Preflight the Dreambook identity-edit training smoke config.

This script intentionally does not import torch or start training. It checks the
dataset/control layout, sample control files, and local model/cache availability
so the smoke config has an explicit gate before a long GPU job or model download.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - operator-facing path
    print("FAIL missing dependency: PyYAML is required to parse the config", file=sys.stderr)
    raise SystemExit(2) from exc


DEFAULT_CONFIG = Path(__file__).with_name(
    "qwen_image_edit_2509_nano_native_smoke_20260511.yaml"
)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
COMMON_MODEL_ROOTS = [
    Path.home() / ".cache/huggingface/hub",
    Path("/home/excalibur/.cache/huggingface/hub"),
    Path("/home/excalibur/Storage/models"),
    Path("/home/excalibur/Code/ComfyUI/models"),
]


def repo_cache_name(repo_id: str) -> str:
    return f"models--{repo_id.replace('/', '--')}"


def existing_paths(paths: Iterable[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def find_repo_cache(repo_id: str) -> list[Path]:
    if repo_id.startswith("/") or repo_id.startswith("."):
        return existing_paths([Path(repo_id).expanduser()])

    cache_name = repo_cache_name(repo_id)
    candidates = []
    for root in COMMON_MODEL_ROOTS:
        candidates.append(root / cache_name)
        candidates.append(root / repo_id)
    return existing_paths(candidates)


def image_files(path: Path) -> list[Path]:
    return sorted(
        child for child in path.iterdir() if child.is_file() and child.suffix.lower() in IMAGE_EXTS
    )


def status_line(ok: bool, label: str, detail: str = "") -> None:
    prefix = "PASS" if ok else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"{prefix} {label}{suffix}")


def warn_line(label: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"WARN {label}{suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to the ai-toolkit YAML config",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Do not fail if the model/adapters are absent from local cache",
    )
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    failures = 0

    if not config_path.exists():
        status_line(False, "config exists", str(config_path))
        return 1

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    process = config.get("config", {}).get("process", [])
    trainer = process[0] if process else {}
    datasets = trainer.get("datasets", [])
    model = trainer.get("model", {})
    sample = trainer.get("sample", {})

    status_line(True, "config parsed", str(config_path))

    if not datasets:
        status_line(False, "datasets configured")
        return 1

    dataset = datasets[0]
    target_dir = Path(dataset.get("folder_path", "")).expanduser()
    control_dirs = [Path(path).expanduser() for path in dataset.get("control_path", [])]
    caption_ext = dataset.get("caption_ext", "txt").lstrip(".")

    if not target_dir.is_dir():
        status_line(False, "target folder exists", str(target_dir))
        failures += 1
        targets = []
    else:
        targets = image_files(target_dir)
        status_line(bool(targets), "target images found", f"{len(targets)} in {target_dir}")
        failures += 0 if targets else 1

    for index, control_dir in enumerate(control_dirs, start=1):
        if not control_dir.is_dir():
            status_line(False, f"control folder {index} exists", str(control_dir))
            failures += 1
            continue
        controls = image_files(control_dir)
        status_line(bool(controls), f"control folder {index} images found", f"{len(controls)} in {control_dir}")
        failures += 0 if controls else 1

    missing_caption = []
    missing_control = []
    for target in targets:
        caption = target.with_suffix(f".{caption_ext}")
        if not caption.exists():
            missing_caption.append(caption)
        for control_dir in control_dirs:
            control = control_dir / target.name
            if not control.exists():
                missing_control.append(control)

    status_line(
        not missing_caption,
        "target captions match images",
        f"{len(targets) - len(missing_caption)}/{len(targets)}",
    )
    failures += 0 if not missing_caption else 1

    expected_controls = len(targets) * len(control_dirs)
    status_line(
        not missing_control,
        "control images match target basenames",
        f"{expected_controls - len(missing_control)}/{expected_controls}",
    )
    failures += 0 if not missing_control else 1

    samples = sample.get("samples", [])
    sample_missing = []
    for sample_entry in samples:
        for key in ("ctrl_img_1", "ctrl_img_2", "ctrl_img_3"):
            value = sample_entry.get(key)
            if value and not Path(value).expanduser().exists():
                sample_missing.append(Path(value))
    status_line(
        not sample_missing,
        "sample control files exist",
        f"{len(samples)} samples",
    )
    failures += 0 if not sample_missing else 1

    name_or_path = model.get("name_or_path")
    if name_or_path:
        model_hits = find_repo_cache(name_or_path)
        if model_hits:
            status_line(True, "model available locally", ", ".join(str(path) for path in model_hits))
        elif args.allow_download:
            warn_line("model not found locally", f"{name_or_path}; run may download")
        else:
            status_line(False, "model available locally", name_or_path)
            failures += 1
    else:
        status_line(False, "model name_or_path configured")
        failures += 1

    for key in ("qtype", "qtype_te"):
        value = model.get(key)
        if not isinstance(value, str) or "|" not in value:
            continue
        _, repo_file = value.split("|", 1)
        repo_id = "/".join(repo_file.split("/")[:2])
        file_name = "/".join(repo_file.split("/")[2:])
        adapter_hits = find_repo_cache(repo_id)
        if not adapter_hits:
            if args.allow_download:
                warn_line(f"{key} adapter repo not found locally", repo_file)
            else:
                status_line(False, f"{key} adapter repo available locally", repo_file)
                failures += 1
            continue
        if file_name:
            file_hits = [path for root in adapter_hits for path in root.rglob(Path(file_name).name)]
            if file_hits:
                status_line(True, f"{key} adapter file available locally", str(file_hits[0]))
            elif args.allow_download:
                warn_line(f"{key} adapter file not found locally", repo_file)
            else:
                status_line(False, f"{key} adapter file available locally", repo_file)
                failures += 1

    if failures:
        print(f"\nReadiness: NOT READY ({failures} failing gate(s))")
        return 1

    print("\nReadiness: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
