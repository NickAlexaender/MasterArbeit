# Script to extract validation images and create binary masks from COCO dataset
# Extracts all 115 validation images to myThesis/image/car/1images (numbered 1.jpg, 2.jpg, ...)
# Creates binary masks (255 if pixel belongs to any class, 0 otherwise) to myThesis/image/car/2masken

import json
import os
import shutil
from PIL import Image, ImageDraw
import numpy as np

# Paths configuration
DATASET_ROOT = "/Users/nicklehmacher/Alles/MasterArbeit/ultralytics/datasets"
ANNOTATIONS_PATH = os.path.join(DATASET_ROOT, "annotations", "instances_val2017.json")
OUTPUT_IMAGES_DIR = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/1images"
OUTPUT_MASKS_DIR = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/2masken"

# Mapping file to keep track of original filenames
MAPPING_FILE = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/filename_mapping.json"

def load_coco_annotations(annotations_path):
    """Load COCO format annotations from JSON file"""
    print(f"📄 Loading annotations from: {annotations_path}")
    with open(annotations_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def create_binary_mask_from_annotations(annotations, image_width, image_height):
    """
    Create a binary mask from all annotation segmentations for an image.
    Returns a binary mask where 255 indicates the pixel belongs to any class.
    
    Handles all annotations and their polygon segmentations properly.
    """
    # Create empty mask
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    
    total_polygons = 0
    categories_found = set()
    
    for ann in annotations:
        category_id = ann.get('category_id', 'unknown')
        categories_found.add(category_id)
        
        segmentation = ann.get('segmentation', [])
        
        # Handle polygon format (list of lists)
        if isinstance(segmentation, list):
            for polygon in segmentation:
                if isinstance(polygon, list) and len(polygon) >= 6:
                    # Each polygon is a flat list of [x1, y1, x2, y2, ...]
                    # Convert to list of tuples [(x1, y1), (x2, y2), ...]
                    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    draw.polygon(points, fill=255)
                    total_polygons += 1
        
        # Handle RLE format (dict with 'counts' and 'size')
        elif isinstance(segmentation, dict) and 'counts' in segmentation:
            # Decode RLE - this would require pycocotools
            # For now, we skip RLE (our dataset uses polygons only)
            print(f"⚠️  RLE segmentation found (not supported yet)")
    
    return mask, total_polygons, categories_found

def clear_output_directories():
    """Clear existing files in output directories"""
    for directory in [OUTPUT_IMAGES_DIR, OUTPUT_MASKS_DIR]:
        if os.path.exists(directory):
            for f in os.listdir(directory):
                file_path = os.path.join(directory, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"🗑️  Cleared: {directory}")

def extract_images_and_masks():
    """
    Main function to extract validation images and create binary masks.
    Images are saved with numbered filenames (1.jpg, 2.jpg, ...).
    Masks are saved with corresponding names (1_mask.png, 2_mask.png, ...).
    """
    # Clear existing files
    clear_output_directories()
    
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)
    print(f"✅ Output directories ready:")
    print(f"   📁 Images: {OUTPUT_IMAGES_DIR}")
    print(f"   📁 Masks: {OUTPUT_MASKS_DIR}")
    
    # Load COCO annotations
    coco_data = load_coco_annotations(ANNOTATIONS_PATH)
    
    # Get images, annotations, and categories
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data.get('categories', [])
    
    print(f"📊 Found {len(images)} images in validation set")
    print(f"📊 Found {len(annotations)} annotations")
    print(f"📊 Found {len(categories)} categories:")
    for cat in categories:
        print(f"      {cat['id']}: {cat['name']}")
    
    # Create a mapping from image_id to annotations
    image_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Sort images by ID for consistent numbering
    images_sorted = sorted(images, key=lambda x: x['id'])
    
    # Keep track of filename mapping
    filename_mapping = {}
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    total_annotations_processed = 0
    all_categories = set()
    
    for idx, img_info in enumerate(images_sorted, start=1):
        img_id = img_info['id']
        file_name = img_info['file_name']
        width = img_info['width']
        height = img_info['height']
        
        # Source image path
        src_image_path = os.path.join(DATASET_ROOT, file_name)
        
        # Numbered destination paths
        numbered_image_name = f"{idx}.jpg"
        numbered_mask_name = f"{idx}_mask.png"
        
        dst_image_path = os.path.join(OUTPUT_IMAGES_DIR, numbered_image_name)
        dst_mask_path = os.path.join(OUTPUT_MASKS_DIR, numbered_mask_name)
        
        # Check if source image exists
        if not os.path.exists(src_image_path):
            print(f"⚠️  Image not found: {src_image_path}")
            skipped_count += 1
            continue
        
        # Copy image to destination (with numbered name)
        shutil.copy2(src_image_path, dst_image_path)
        
        # Store mapping
        filename_mapping[str(idx)] = {
            'original_filename': os.path.basename(file_name),
            'original_path': file_name,
            'image_id': img_id,
            'width': width,
            'height': height
        }
        
        # Get annotations for this image
        img_annotations = image_annotations.get(img_id, [])
        
        # Create binary mask from all annotations
        mask, num_polygons, categories_found = create_binary_mask_from_annotations(
            img_annotations, width, height
        )
        
        # Save mask
        mask.save(dst_mask_path)
        
        # Track statistics
        total_annotations_processed += len(img_annotations)
        all_categories.update(categories_found)
        
        processed_count += 1
        if processed_count % 20 == 0:
            print(f"   ✅ Processed {processed_count}/{len(images)} images...")
    
    # Save filename mapping
    with open(MAPPING_FILE, 'w') as f:
        json.dump(filename_mapping, f, indent=2)
    
    print(f"\n🎉 Processing complete!")
    print(f"   ✅ Successfully processed: {processed_count} images")
    print(f"   ⚠️  Skipped: {skipped_count} images")
    print(f"   📝 Total annotations processed: {total_annotations_processed}")
    print(f"   🏷️  Categories found: {sorted(all_categories)}")
    print(f"\n📁 Images saved to: {OUTPUT_IMAGES_DIR}")
    print(f"📁 Masks saved to: {OUTPUT_MASKS_DIR}")
    print(f"📁 Filename mapping saved to: {MAPPING_FILE}")

def verify_output():
    """Verify the output by checking the number of files and showing samples"""
    print("\n🔍 Verifying output...")
    
    # Count files
    images = sorted([f for f in os.listdir(OUTPUT_IMAGES_DIR) if f.endswith('.jpg')], 
                   key=lambda x: int(x.split('.')[0]))
    masks = sorted([f for f in os.listdir(OUTPUT_MASKS_DIR) if f.endswith('.png')],
                  key=lambda x: int(x.split('_')[0]))
    
    print(f"   📸 Images in output directory: {len(images)}")
    print(f"   🎭 Masks in output directory: {len(masks)}")
    
    # Show first few samples
    print(f"\n📋 Sample files:")
    for i in range(min(5, len(images))):
        img_name = images[i]
        mask_name = masks[i] if i < len(masks) else "N/A"
        print(f"   {img_name} → {mask_name}")
    
    # Analyze a few masks to verify coverage
    print(f"\n📊 Mask analysis (first 5 images):")
    for i in range(min(5, len(masks))):
        mask_path = os.path.join(OUTPUT_MASKS_DIR, masks[i])
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        coverage = np.sum(mask_array > 0) / mask_array.size * 100
        print(f"   {masks[i]}: {coverage:.1f}% coverage ({np.sum(mask_array > 0)} / {mask_array.size} pixels)")
    
    # Load and show mapping info
    if os.path.exists(MAPPING_FILE):
        with open(MAPPING_FILE, 'r') as f:
            mapping = json.load(f)
        print(f"\n📝 Filename mapping sample (first 5):")
        for key in list(mapping.keys())[:5]:
            print(f"   {key}.jpg → {mapping[key]['original_filename']}")

if __name__ == "__main__":
    print("🚗 Car Parts Validation Dataset Extractor")
    print("=" * 50)
    print(f"📁 Source dataset: {DATASET_ROOT}")
    print(f"📁 Annotations: {ANNOTATIONS_PATH}")
    print()
    
    extract_images_and_masks()
    verify_output()
