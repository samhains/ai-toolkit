"""
Supabase Storage upload for LoRA checkpoints.

Uploads .safetensors to the ai-creative-studio-shelf bucket
under loras/{model_arch}/{job_name}/.

All info (job name, model arch, step count) is encoded in the path
and filename — no metadata sidecars needed.

Environment variables:
  SUPABASE_URL  - Project URL
  SUPABASE_KEY  - Service role or anon key with storage write access
"""

import os
import traceback
from typing import Optional

SHELF_BUCKET = "ai-creative-studio-shelf"

_client = None


def _get_client():
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if not url or not key:
            return None
        from supabase import create_client
        _client = create_client(url, key)
    return _client


def upload_lora(
    safetensors_path: str,
    job_name: str,
    model_arch: str,
    upload_name: str = None,
    bucket: str = SHELF_BUCKET,
) -> Optional[str]:
    """
    Upload a single .safetensors file to Supabase Storage.

    Args:
        upload_name: Override the filename in storage (e.g. to normalize step naming).

    Returns public URL on success, None on failure. Never raises.
    """
    try:
        client = _get_client()
        if client is None:
            print("[supabase] SUPABASE_URL or SUPABASE_KEY not set, skipping")
            return None

        if not os.path.exists(safetensors_path):
            print(f"[supabase] File not found: {safetensors_path}")
            return None

        storage = client.storage.from_(bucket)
        filename = upload_name or os.path.basename(safetensors_path)
        file_size_mb = os.path.getsize(safetensors_path) / 1024 / 1024
        storage_path = f"loras/{model_arch}/{job_name}/{filename}"

        print(f"[supabase] Uploading {filename} ({file_size_mb:.1f} MB)...")

        with open(safetensors_path, "rb") as f:
            file_data = f.read()

        storage.upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": "application/octet-stream", "upsert": "true"},
        )

        public_url = storage.get_public_url(storage_path)
        print(f"[supabase] Done: {storage_path}")
        return public_url

    except Exception as e:
        print(f"[supabase] Upload failed (non-fatal): {e}")
        traceback.print_exc()
        return None


def list_existing(job_name: str, model_arch: str, bucket: str = SHELF_BUCKET) -> set:
    """Return set of filenames already uploaded for this run."""
    try:
        client = _get_client()
        if client is None:
            return set()
        storage = client.storage.from_(bucket)
        folder = f"loras/{model_arch}/{job_name}"
        result = storage.list(folder, {"limit": 10000})
        return {f["name"] for f in result if f.get("name")}
    except Exception:
        return set()
