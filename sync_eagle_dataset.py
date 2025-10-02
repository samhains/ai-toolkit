#!/usr/bin/env python3
"""
Sync dataset from Supabase eagle_images table.

This script intelligently syncs images from Supabase:
1. Tracks what's already been downloaded via .dataset_sync.json
2. Only downloads new or updated images
3. Updates captions when they change
4. Optionally removes images that were deleted from Supabase
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import requests
from tqdm import tqdm
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Storage bucket base URL for constructing image URLs from storage_path
STORAGE_BASE_URL = os.getenv('SUPABASE_STORAGE_BASE_URL',
                              'https://idyoveanwiuwcvgxijtz.supabase.co/storage/v1/object/public/eagle-images/')

SYNC_METADATA_FILE = '.dataset_sync.json'

def get_image_extension(url, fallback_ext=None):
    """Extract file extension from URL or storage path."""
    if fallback_ext:
        ext = fallback_ext
        if not ext.startswith('.'):
            ext = '.' + ext
        return ext

    parsed = urlparse(url)
    path = parsed.path
    _, ext = os.path.splitext(path)
    if not ext:
        ext = '.jpg'  # default
    return ext

def sanitize_filename(filename):
    """Remove or replace invalid filename characters."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

def download_image(url, save_path):
    """Download an image from URL to save_path."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def load_sync_metadata(output_dir):
    """Load the sync metadata file."""
    metadata_path = Path(output_dir) / SYNC_METADATA_FILE
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {
        'version': '1.0',
        'last_sync': None,
        'images': {}  # key: image_id, value: {filename, updated_at, caption_hash}
    }

def save_sync_metadata(output_dir, metadata):
    """Save the sync metadata file."""
    metadata_path = Path(output_dir) / SYNC_METADATA_FILE
    metadata['last_sync'] = datetime.utcnow().isoformat()
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_caption_hash(caption):
    """Get a simple hash of caption for change detection."""
    import hashlib
    return hashlib.md5(caption.encode('utf-8')).hexdigest()

def sync_dataset(
    output_dir,
    limit=None,
    caption_field='caption',
    fallback_caption_field='merged_text',
    filters=None,
    create_json=False,
    remove_deleted=False
):
    """
    Sync dataset from the eagle_images table.

    Args:
        output_dir: Directory to save images and captions
        limit: Maximum number of images to download (None for all)
        caption_field: Primary field to use for captions
        fallback_caption_field: Fallback field if primary is empty
        filters: Dict of column filters (e.g., {'tags': ['some_tag']})
        create_json: If True, create a JSON dataset file instead of .txt files
        remove_deleted: If True, remove local files for images deleted from Supabase
    """

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY environment variables")

    # Create Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load sync metadata
    metadata = load_sync_metadata(output_dir)
    print(f"Last sync: {metadata.get('last_sync', 'Never')}")
    print(f"Previously synced images: {len(metadata['images'])}")

    print(f"\nFetching images from eagle_images table...")

    # Fetch images in batches to avoid timeout
    images = []
    batch_size = 1000
    offset = 0

    while True:
        # Build query for this batch
        query = supabase.table('eagle_images').select(
            'id, eagle_id, image_url, storage_path, file_extension, updated_at, '
            f'{caption_field}, {fallback_caption_field}'
        )

        # Don't filter by image_url anymore - we'll construct from storage_path if needed
        # query = query.not_.is_('image_url', 'null')

        # Apply filters
        if filters:
            for column, value in filters.items():
                if isinstance(value, list):
                    # For array columns like tags
                    query = query.contains(column, value)
                else:
                    query = query.eq(column, value)

        # Order by id for consistent pagination
        query = query.order('id', desc=False)
        query = query.range(offset, offset + batch_size - 1)

        # Execute query
        try:
            response = query.execute()
            batch = response.data

            if not batch:
                break

            images.extend(batch)
            print(f"Fetched {len(images)} images so far...")

            # Check if we've reached the limit
            if limit and len(images) >= limit:
                images = images[:limit]
                break

            # Check if we got fewer results than batch_size (last page)
            if len(batch) < batch_size:
                break

            offset += batch_size

        except Exception as e:
            print(f"Error fetching batch at offset {offset}: {e}")
            if images:
                print(f"Continuing with {len(images)} images fetched so far...")
                break
            else:
                raise

    print(f"Found {len(images)} images in Supabase")

    if not images:
        print("No images found. Exiting.")
        return

    # Track current image IDs for cleanup
    current_image_ids = set()

    # Stats
    stats = {
        'new': 0,
        'updated': 0,
        'unchanged': 0,
        'failed': 0,
        'deleted': 0
    }

    # Dataset mapping for JSON format
    dataset_json = {}

    # Process images
    for img_data in tqdm(images, desc="Syncing images"):
        img_id = str(img_data.get('id'))
        current_image_ids.add(img_id)

        # Get image URL - use image_url if available, otherwise construct from storage_path
        image_url = img_data.get('image_url')
        storage_path = img_data.get('storage_path')

        if not image_url and storage_path:
            # Construct URL from storage_path
            image_url = STORAGE_BASE_URL + storage_path

        if not image_url:
            print(f"Skipping {img_id} - no image_url or storage_path")
            stats['failed'] += 1
            continue

        # Get caption
        caption = img_data.get(caption_field) or img_data.get(fallback_caption_field) or ''
        caption_hash = get_caption_hash(caption)

        # Get updated_at timestamp
        updated_at = img_data.get('updated_at')

        # Create filename using eagle_id or id
        base_name = img_data.get('eagle_id') or img_id
        base_name = sanitize_filename(base_name)

        # Get file extension
        ext = get_image_extension(image_url, img_data.get('file_extension'))

        # Paths
        image_filename = f"{base_name}{ext}"
        image_path = output_path / image_filename
        caption_path = output_path / f"{base_name}.txt"

        # Check if this is new or updated
        is_new = img_id not in metadata['images']
        is_updated = False

        if not is_new:
            old_data = metadata['images'][img_id]
            # Check if updated_at changed or caption changed
            if old_data.get('updated_at') != updated_at or old_data.get('caption_hash') != caption_hash:
                is_updated = True

        # Download if new or updated
        if is_new or is_updated:
            if download_image(image_url, image_path):
                # Save caption
                if create_json:
                    dataset_json[str(image_filename)] = caption
                else:
                    with open(caption_path, 'w', encoding='utf-8') as f:
                        f.write(caption)

                # Update metadata
                metadata['images'][img_id] = {
                    'filename': image_filename,
                    'updated_at': updated_at,
                    'caption_hash': caption_hash,
                    'eagle_id': img_data.get('eagle_id')
                }

                if is_new:
                    stats['new'] += 1
                else:
                    stats['updated'] += 1
            else:
                stats['failed'] += 1
        else:
            stats['unchanged'] += 1
            # Still add to JSON if we're creating it
            if create_json:
                dataset_json[str(image_filename)] = caption

    # Handle deleted images
    if remove_deleted:
        deleted_ids = set(metadata['images'].keys()) - current_image_ids
        for img_id in deleted_ids:
            old_data = metadata['images'][img_id]
            filename = old_data.get('filename')
            if filename:
                image_path = output_path / filename
                base_name = Path(filename).stem
                caption_path = output_path / f"{base_name}.txt"

                # Remove files
                if image_path.exists():
                    image_path.unlink()
                if caption_path.exists():
                    caption_path.unlink()

                print(f"Removed deleted image: {filename}")

            del metadata['images'][img_id]
            stats['deleted'] += 1

    # Save JSON dataset file if requested
    if create_json and dataset_json:
        json_path = output_path / 'dataset.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset_json, f, indent=2, ensure_ascii=False)
        print(f"Updated dataset JSON: {json_path}")

    # Save metadata
    save_sync_metadata(output_dir, metadata)

    # Print stats
    print(f"\n{'='*60}")
    print("Sync completed!")
    print(f"{'='*60}")
    print(f"New images:       {stats['new']}")
    print(f"Updated images:   {stats['updated']}")
    print(f"Unchanged images: {stats['unchanged']}")
    print(f"Failed downloads: {stats['failed']}")
    if remove_deleted:
        print(f"Deleted images:   {stats['deleted']}")
    print(f"\nTotal images in dataset: {len(metadata['images'])}")
    print(f"Output directory: {output_path.absolute()}")

    # Print example config on first run
    if metadata.get('last_sync') is None or stats['new'] > 0:
        print("\n" + "="*60)
        print("Example dataset configuration for your training config:")
        print("="*60)

        dataset_config = {
            "dataset_path": str(output_path.absolute()) if not create_json else str((output_path / 'dataset.json').absolute()),
            "caption_ext": ".txt" if not create_json else None,
            "resolution": 512,
            "buckets": True,
        }

        if create_json:
            del dataset_config["caption_ext"]

        print(json.dumps(dataset_config, indent=2))
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Sync dataset from Supabase eagle_images table')
    parser.add_argument('output_dir', help='Directory to save the dataset')
    parser.add_argument('--limit', type=int, help='Maximum number of images to fetch from Supabase')
    parser.add_argument('--caption-field', default='caption', help='Primary caption field (default: caption)')
    parser.add_argument('--fallback-caption', default='merged_text', help='Fallback caption field (default: merged_text)')
    parser.add_argument('--json', action='store_true', help='Create JSON dataset file instead of .txt files')
    parser.add_argument('--filter-tag', action='append', help='Filter by tag (can specify multiple times)')
    parser.add_argument('--remove-deleted', action='store_true', help='Remove local files for images deleted from Supabase')

    args = parser.parse_args()

    filters = {}
    if args.filter_tag:
        filters['tags'] = args.filter_tag

    sync_dataset(
        output_dir=args.output_dir,
        limit=args.limit,
        caption_field=args.caption_field,
        fallback_caption_field=args.fallback_caption,
        filters=filters if filters else None,
        create_json=args.json,
        remove_deleted=args.remove_deleted
    )

if __name__ == '__main__':
    main()
