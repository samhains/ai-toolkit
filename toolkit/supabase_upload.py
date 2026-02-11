"""
Supabase Storage upload for LoRA checkpoints.

Uploads .safetensors and a metadata.json sidecar to the
ai-creative-studio-shelf bucket under loras/{model_arch}/{job_name}/.

Environment variables:
  SUPABASE_URL  - Project URL (e.g., https://xxx.supabase.co)
  SUPABASE_KEY  - Service role or anon key with storage write access
"""

import json
import os
import traceback
from collections import OrderedDict
from typing import Optional

SHELF_BUCKET = "ai-creative-studio-shelf"


def push_to_supabase(
    safetensors_path: str,
    job_name: str,
    model_arch: str,
    metadata: Optional[OrderedDict] = None,
    bucket: str = SHELF_BUCKET,
) -> Optional[str]:
    """
    Upload a LoRA safetensors file and metadata sidecar to Supabase Storage.

    Args:
        safetensors_path: Absolute path to the .safetensors file on disk.
        job_name: The training job name, used for the subfolder.
        model_arch: Model architecture (e.g., flux2_klein_9b, wan, sd3.5).
        metadata: OrderedDict of training metadata to write as JSON sidecar.
        bucket: Supabase Storage bucket name.

    Returns:
        Public URL of the uploaded safetensors, or None on failure.
    """
    try:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")

        if not url or not key:
            print("[supabase_upload] SUPABASE_URL or SUPABASE_KEY not set, skipping upload")
            return None

        if not os.path.exists(safetensors_path):
            print(f"[supabase_upload] File not found: {safetensors_path}")
            return None

        from supabase import create_client
        client = create_client(url, key)
        storage = client.storage.from_(bucket)

        safetensors_name = os.path.basename(safetensors_path)
        file_size_mb = os.path.getsize(safetensors_path) / 1024 / 1024
        storage_path = f"loras/{model_arch}/{job_name}/{safetensors_name}"

        print(f"[supabase_upload] Uploading {safetensors_name} ({file_size_mb:.1f} MB) to {bucket}/{storage_path}...")

        with open(safetensors_path, "rb") as f:
            file_data = f.read()

        storage.upload(
            path=storage_path,
            file=file_data,
            file_options={"content-type": "application/octet-stream", "upsert": "true"},
        )

        public_url = storage.get_public_url(storage_path)
        print(f"[supabase_upload] Uploaded: {public_url}")

        # Upload metadata sidecar
        if metadata is not None:
            meta_dict = _prepare_metadata(metadata, job_name, model_arch, file_size_mb)
        else:
            meta_dict = _minimal_metadata(job_name, model_arch, file_size_mb)

        meta_storage_path = f"loras/{model_arch}/{job_name}/metadata.json"
        meta_json = json.dumps(meta_dict, indent=2)

        storage.upload(
            path=meta_storage_path,
            file=meta_json.encode("utf-8"),
            file_options={"content-type": "application/json", "upsert": "true"},
        )
        print(f"[supabase_upload] Metadata uploaded: {bucket}/{meta_storage_path}")

        return public_url

    except Exception as e:
        print(f"[supabase_upload] Upload failed (non-fatal): {e}")
        traceback.print_exc()
        return None


def _minimal_metadata(job_name: str, model_arch: str, file_size_mb: float) -> dict:
    import datetime
    return {
        "job_name": job_name,
        "model_arch": model_arch,
        "file_size_mb": round(file_size_mb, 1),
        "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def _prepare_metadata(meta: OrderedDict, job_name: str, model_arch: str, file_size_mb: float) -> dict:
    """Build a JSON-serializable metadata dict from training meta."""
    import datetime

    result = _minimal_metadata(job_name, model_arch, file_size_mb)

    # Extract training_info
    training_info = meta.get("training_info", {})
    if isinstance(training_info, str):
        try:
            training_info = json.loads(training_info)
        except (json.JSONDecodeError, TypeError):
            training_info = {}

    result["steps"] = training_info.get("step", None)
    result["epoch"] = training_info.get("epoch", None)
    result["base_model_version"] = meta.get("ss_base_model_version", None)
    result["output_name"] = meta.get("ss_output_name", job_name)

    # Extract trigger word from tag frequency
    tag_freq = meta.get("ss_tag_frequency", {})
    if isinstance(tag_freq, str):
        try:
            tag_freq = json.loads(tag_freq)
        except (json.JSONDecodeError, TypeError):
            tag_freq = {}
    if tag_freq:
        for key in tag_freq.keys():
            if key.startswith("1_"):
                result["trigger_word"] = key[2:]
                break

    return result
