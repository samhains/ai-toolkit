#!/usr/bin/env python3
"""
Sync dataset from Supabase eagle_images table with smart caption generation.

This script creates a training dataset with varied caption styles:
- 20% Cleaned long descriptions
- 20% Cluster + cleaned long descriptions
- 20% Cluster + alt text
- 15% Alt text only
- 15% Cluster + tags + short description
- 10% Tags only

Supports reusing images from existing datasets to avoid re-downloading.
"""

import os
import json
import argparse
import re
import shutil
import random
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


def clean_caption_text(text):
    """
    Clean academic/formal phrasing from captions to make them more natural.

    Removes:
    - Academic intros ("The image displays", "This image shows")
    - Hedge words ("appears to", "suggesting", "visible")
    - Imprecise quantifiers ("approximately", "numerous", "various")
    """
    if not text:
        return ""

    # Remove academic intros at start
    text = re.sub(r'^(The image |This image |The |This )(displays?|shows?|depicts?|presents?|features?|contains?)\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^(An? |The )(illustration|photograph|digital image|render|screenshot|photo|film_still)\s+(of|showing|depicting|featuring)\s+', '', text, flags=re.IGNORECASE)

    # Remove hedge phrases
    text = re.sub(r'\b(appears to be|seems to be|suggesting|is visible|are visible|can be seen)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(is positioned|are positioned|is located|are located|is situated|are situated)\b', '', text, flags=re.IGNORECASE)

    # Remove imprecise quantifiers
    text = re.sub(r'\b(approximately|roughly|about|various|numerous|multiple|several)\s+', '', text, flags=re.IGNORECASE)

    # Clean up extra spaces and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.])', r'\1', text)
    text = text.strip()

    # Capitalize first letter if needed
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


def shorten_description(long_text, max_tokens=100):
    """Create a shortened version of a long description by keeping key details."""
    if not long_text:
        return ""

    # Split into sentences
    sentences = re.split(r'[.!?]+', long_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Take first 1-2 sentences and clean
    short = '. '.join(sentences[:2])
    if short and not short.endswith('.'):
        short += '.'

    return clean_caption_text(short)


def generate_caption_variant(image_data, cluster_names, variant_type):
    """
    Generate a caption based on the variant type.

    Args:
        image_data: Dict with caption, alt_text, tags fields
        cluster_names: List of cluster names for this image
        variant_type: One of ['long', 'cluster_long', 'cluster_alt', 'alt', 'cluster_tags', 'tags']

    Returns:
        Generated caption string
    """
    caption = image_data.get('caption', '')
    alt_text = image_data.get('alt_text', '')
    tags = image_data.get('tags', [])

    # Get primary cluster name (first one if multiple)
    cluster_prefix = cluster_names[0] if cluster_names else None

    # Format tags (remove duplicates and clean)
    clean_tags = []
    seen = set()
    for tag in tags:
        tag_lower = tag.lower().strip()
        if tag_lower and tag_lower not in seen:
            clean_tags.append(tag_lower)
            seen.add(tag_lower)

    tag_string = ', '.join(clean_tags[:15])  # Limit to 15 tags

    # Generate based on variant type
    if variant_type == 'long':
        # 20% - Cleaned long description
        return clean_caption_text(caption)

    elif variant_type == 'cluster_long':
        # 20% - Cluster + cleaned long description (shortened)
        short_desc = shorten_description(caption, max_tokens=80)
        if cluster_prefix:
            return f"{cluster_prefix}: {short_desc}"
        return short_desc

    elif variant_type == 'cluster_alt':
        # 20% - Cluster + alt text
        if cluster_prefix and alt_text:
            return f"{cluster_prefix}: {alt_text}"
        return alt_text or clean_caption_text(caption)

    elif variant_type == 'alt':
        # 15% - Alt text only
        return alt_text or clean_caption_text(caption)

    elif variant_type == 'cluster_tags':
        # 15% - Cluster + tags + short description
        short_desc = shorten_description(alt_text or caption, max_tokens=50)
        if cluster_prefix and tag_string:
            return f"{cluster_prefix} | {tag_string} | {short_desc}"
        elif tag_string:
            return f"{tag_string} | {short_desc}"
        return short_desc

    elif variant_type == 'tags':
        # 10% - Tags only
        return tag_string or clean_caption_text(caption)

    else:
        # Fallback
        return clean_caption_text(caption)


def get_caption_variant_type(seed=None):
    """
    Randomly select a caption variant type based on the distribution:
    20% long, 20% cluster_long, 20% cluster_alt, 15% alt, 15% cluster_tags, 10% tags

    Args:
        seed: Optional seed for deterministic selection
    """
    if seed is not None:
        random.seed(seed)

    rand = random.random()

    if rand < 0.20:
        return 'long'
    elif rand < 0.40:
        return 'cluster_long'
    elif rand < 0.60:
        return 'cluster_alt'
    elif rand < 0.75:
        return 'alt'
    elif rand < 0.90:
        return 'cluster_tags'
    else:
        return 'tags'


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


def copy_image_from_source(source_dir, eagle_id, dest_path):
    """
    Copy image from source directory by matching eagle_id.

    Args:
        source_dir: Path to source dataset directory
        eagle_id: Eagle ID to match
        dest_path: Destination path for the image

    Returns:
        True if successful, False otherwise
    """
    source_path = Path(source_dir)

    # Try to find file matching eagle_id
    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']:
        potential_file = source_path / f"{eagle_id}{ext}"
        if potential_file.exists():
            try:
                shutil.copy2(potential_file, dest_path)
                return True
            except Exception as e:
                print(f"Error copying {potential_file}: {e}")
                return False

    return False


def load_sync_metadata(output_dir):
    """Load the sync metadata file."""
    metadata_path = Path(output_dir) / SYNC_METADATA_FILE
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {
        'version': '2.0',  # Updated version for smart captions
        'last_sync': None,
        'images': {}  # key: image_id, value: {filename, updated_at, caption_variant}
    }


def save_sync_metadata(output_dir, metadata):
    """Save the sync metadata file."""
    metadata_path = Path(output_dir) / SYNC_METADATA_FILE
    metadata['last_sync'] = datetime.utcnow().isoformat()
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def sync_dataset(
    output_dir,
    limit=None,
    filters=None,
    create_json=False,
    remove_deleted=False,
    reuse_images_from=None,
    deterministic=False
):
    """
    Sync dataset from the eagle_images table with smart caption generation.

    Args:
        output_dir: Directory to save images and captions
        limit: Maximum number of images to download (None for all)
        filters: Dict of column filters (e.g., {'tags': ['some_tag']})
        create_json: If True, create a JSON dataset file instead of .txt files
        remove_deleted: If True, remove local files for images deleted from Supabase
        reuse_images_from: Path to existing dataset to copy images from (instead of downloading)
        deterministic: If True, use deterministic caption variant selection based on image ID
    """

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Please set SUPABASE_URL and SUPABASE_KEY environment variables")

    # Create Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Create output directory (prepend ./datasets/ if not an absolute path)
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path('./datasets') / output_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Load sync metadata
    metadata = load_sync_metadata(output_path)
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
            'caption, alt_text, tags, cluster_ids'
        )

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

    # Fetch cluster information
    print("\nFetching cluster information...")
    cluster_map = {}
    try:
        clusters_response = supabase.table('eagle_clusters').select('id, cluster_name').execute()
        for cluster in clusters_response.data:
            cluster_map[cluster['id']] = cluster['cluster_name']
        print(f"Loaded {len(cluster_map)} clusters")
    except Exception as e:
        print(f"Warning: Could not fetch clusters: {e}")

    # Track current image IDs for cleanup
    current_image_ids = set()

    # Stats
    stats = {
        'new': 0,
        'updated': 0,
        'unchanged': 0,
        'failed': 0,
        'deleted': 0,
        'reused': 0,
        'variant_distribution': {
            'long': 0,
            'cluster_long': 0,
            'cluster_alt': 0,
            'alt': 0,
            'cluster_tags': 0,
            'tags': 0
        }
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

        # Get cluster names for this image
        cluster_ids = img_data.get('cluster_ids', [])
        cluster_names = [cluster_map.get(cid) for cid in cluster_ids if cid in cluster_map]
        cluster_names = [name for name in cluster_names if name]  # Remove None values

        # Get caption variant type
        if deterministic:
            # Use image ID as seed for deterministic selection
            variant_type = get_caption_variant_type(seed=int(img_id))
        else:
            variant_type = get_caption_variant_type()

        # Generate caption
        caption = generate_caption_variant(img_data, cluster_names, variant_type)
        stats['variant_distribution'][variant_type] += 1

        # Get updated_at timestamp
        updated_at = img_data.get('updated_at')

        # Create filename using eagle_id or id
        eagle_id = img_data.get('eagle_id')
        base_name = eagle_id or img_id
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
            # Check if updated_at changed
            if old_data.get('updated_at') != updated_at:
                is_updated = True

        # Download or copy image if new or updated
        if is_new or is_updated:
            success = False

            # Try to reuse image from source directory first
            if reuse_images_from and eagle_id:
                success = copy_image_from_source(reuse_images_from, eagle_id, image_path)
                if success:
                    stats['reused'] += 1

            # Fall back to downloading if copy failed or not using reuse
            if not success:
                success = download_image(image_url, image_path)

            if success:
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
                    'caption_variant': variant_type,
                    'eagle_id': eagle_id
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
    save_sync_metadata(output_path, metadata)

    # Print stats
    print(f"\n{'='*60}")
    print("Sync completed!")
    print(f"{'='*60}")
    print(f"New images:       {stats['new']}")
    print(f"Updated images:   {stats['updated']}")
    print(f"Unchanged images: {stats['unchanged']}")
    print(f"Failed downloads: {stats['failed']}")
    if reuse_images_from:
        print(f"Reused images:    {stats['reused']}")
    if remove_deleted:
        print(f"Deleted images:   {stats['deleted']}")

    print(f"\nCaption variant distribution:")
    for variant, count in stats['variant_distribution'].items():
        percentage = (count / len(images) * 100) if images else 0
        print(f"  {variant:15s}: {count:5d} ({percentage:5.1f}%)")

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
    parser = argparse.ArgumentParser(description='Sync dataset from Supabase eagle_images table with smart captions')
    parser.add_argument('output_dir', help='Directory to save the dataset')
    parser.add_argument('--limit', type=int, help='Maximum number of images to fetch from Supabase')
    parser.add_argument('--json', action='store_true', help='Create JSON dataset file instead of .txt files')
    parser.add_argument('--filter-tag', action='append', help='Filter by tag (can specify multiple times)')
    parser.add_argument('--remove-deleted', action='store_true', help='Remove local files for images deleted from Supabase')
    parser.add_argument('--reuse-images', type=str, help='Path to existing dataset directory to copy images from (e.g., eagle_images_2)')
    parser.add_argument('--deterministic', action='store_true', help='Use deterministic caption variant selection based on image ID')

    args = parser.parse_args()

    filters = {}
    if args.filter_tag:
        filters['tags'] = args.filter_tag

    sync_dataset(
        output_dir=args.output_dir,
        limit=args.limit,
        filters=filters if filters else None,
        create_json=args.json,
        remove_deleted=args.remove_deleted,
        reuse_images_from=args.reuse_images,
        deterministic=args.deterministic
    )


if __name__ == '__main__':
    main()
