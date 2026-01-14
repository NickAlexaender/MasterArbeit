#!/usr/bin/env python3
"""
Leeds Butterfly Dataset to COCO Format Converter
Converts segmentation masks to COCO format for use with MaskDINO

Dataset structure:
- images/XXXNNNN.png (XXX = category 001-010, NNNN = sequence number)
- segmentations/XXXNNNN_seg0.png (foreground pixels: values 1 and 3)
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime
import random
from pycocotools import mask as mask_util

# Butterfly categories from README
BUTTERFLY_CATEGORIES = {
    "001": "Danaus plexippus",      # Monarch
    "002": "Heliconius charitonius", # Zebra Longwing
    "003": "Heliconius erato",       # Red Postman
    "004": "Junonia coenia",         # Common Buckeye
    "005": "Lycaena phlaeas",        # Small Copper
    "006": "Nymphalis antiopa",      # Mourning Cloak
    "007": "Papilio cresphontes",    # Giant Swallowtail
    "008": "Pieris rapae",           # Cabbage White
    "009": "Vanessa atalanta",       # Red Admiral
    "010": "Vanessa cardui",         # Painted Lady
}


def mask_to_polygons(mask):
    """
    Konvertiert eine binäre Maske zu Polygon-Koordinaten.
    Gibt eine Liste von Polygonen zurück (für den Fall von mehreren getrennten Regionen).
    """
    # Finde Konturen
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL,  # Nur äußere Konturen
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        # Ignoriere sehr kleine Konturen (Rauschen)
        if cv2.contourArea(contour) < 100:
            continue
        
        # Vereinfache Kontur leicht für bessere Performance
        epsilon = 0.002 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Konvertiere zu flacher Liste [x1, y1, x2, y2, ...]
        if len(contour) >= 3:  # Mindestens 3 Punkte für ein Polygon
            polygon = contour.flatten().tolist()
            polygons.append(polygon)
    
    return polygons


def mask_to_rle(mask):
    """
    Konvertiert eine binäre Maske zu RLE (Run-Length Encoding) für COCO.
    """
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def calculate_bbox_from_mask(mask):
    """
    Berechnet Bounding Box aus einer binären Maske.
    Gibt [x_min, y_min, width, height] zurück.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def calculate_area_from_mask(mask):
    """Berechnet die Fläche der Maske in Pixeln."""
    return int(np.sum(mask > 0))


def load_segmentation_mask(seg_path):
    """
    Lädt die Segmentierungsmaske und konvertiert zu binär.
    Die Masken haben Pixelwerte:
    - 0: Hintergrund
    - 255: Vordergrund (Schmetterling)
    - 76, 149: Randpixel (werden als Vordergrund behandelt)
    """
    mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    # Vordergrund sind alle Pixel mit Wert > 0 (oder alternativ nur 255)
    # Verwende > 128 um nur die deutlichen Vordergrund-Pixel zu nehmen
    binary_mask = (mask > 128).astype(np.uint8)
    return binary_mask


def analyze_dataset():
    """Analysiert das Leeds Butterfly Dataset."""
    
    dataset_root = Path(__file__).parent
    images_dir = dataset_root / "images"
    seg_dir = dataset_root / "segmentations"
    
    print("🦋 LEEDS BUTTERFLY DATASET ANALYZER")
    print("=" * 50)
    
    # Sammle alle Bilder
    images = sorted(list(images_dir.glob("*.png")))
    print(f"📊 Total images found: {len(images)}")
    
    # Zähle pro Kategorie
    category_counts = {}
    for img_path in images:
        cat_id = img_path.stem[:3]
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    print(f"\n📋 Images per category:")
    for cat_id in sorted(category_counts.keys()):
        cat_name = BUTTERFLY_CATEGORIES.get(cat_id, "Unknown")
        print(f"   {cat_id}: {category_counts[cat_id]:3d} images - {cat_name}")
    
    # Prüfe Segmentierungen
    matched = 0
    for img_path in images:
        seg_path = seg_dir / f"{img_path.stem}_seg0.png"
        if seg_path.exists():
            matched += 1
    
    print(f"\n✅ Matched image-segmentation pairs: {matched}/{len(images)}")
    
    # Sample-Analyse
    if images:
        sample_img = cv2.imread(str(images[0]))
        seg_path = seg_dir / f"{images[0].stem}_seg0.png"
        sample_seg = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        
        print(f"\n📐 Sample image dimensions: {sample_img.shape}")
        if sample_seg is not None:
            unique_vals = np.unique(sample_seg)
            print(f"📐 Sample segmentation unique values: {unique_vals}")
            
            # Test mask loading
            binary_mask = load_segmentation_mask(seg_path)
            foreground_pixels = np.sum(binary_mask > 0)
            print(f"📊 Foreground pixels in sample: {foreground_pixels}")
    
    return images, seg_dir


def resize_images_to_square(target_size=256):
    """
    Schneidet alle Bilder und Masken quadratisch zu (zentriert) und skaliert auf target_size x target_size.
    Überschreibt die originalen Dateien.
    
    Args:
        target_size: Zielgröße in Pixeln (Standard: 256)
    """
    dataset_root = Path(__file__).parent
    images_dir = dataset_root / "images"
    seg_dir = dataset_root / "segmentations"
    
    print(f"\n🔄 RESIZING IMAGES TO {target_size}x{target_size}")
    print("=" * 50)
    
    # Sammle alle Bilder
    images = sorted(list(images_dir.glob("*.png")))
    print(f"📊 Found {len(images)} images to process")
    
    processed = 0
    errors = 0
    
    for img_path in images:
        try:
            # Lade Bild
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"   ⚠️ Could not load: {img_path.name}")
                errors += 1
                continue
            
            # Lade zugehörige Segmentierungsmaske
            seg_path = seg_dir / f"{img_path.stem}_seg0.png"
            mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE) if seg_path.exists() else None
            
            h, w = img.shape[:2]
            
            # Quadratisch zuschneiden (zentriert)
            if h != w:
                min_dim = min(h, w)
                # Zentrierter Ausschnitt
                start_x = (w - min_dim) // 2
                start_y = (h - min_dim) // 2
                
                img = img[start_y:start_y + min_dim, start_x:start_x + min_dim]
                if mask is not None:
                    mask = mask[start_y:start_y + min_dim, start_x:start_x + min_dim]
            
            # Auf Zielgröße skalieren
            img_resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
            
            # Speichere Bild (überschreibt Original)
            cv2.imwrite(str(img_path), img_resized)
            
            # Speichere Maske (überschreibt Original)
            if mask is not None:
                mask_resized = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(seg_path), mask_resized)
            
            processed += 1
            
            if processed % 100 == 0:
                print(f"   ✅ Processed {processed}/{len(images)} images...")
                
        except Exception as e:
            print(f"   ❌ Error processing {img_path.name}: {e}")
            errors += 1
    
    print(f"\n✅ Resizing complete!")
    print(f"   Processed: {processed}")
    print(f"   Errors: {errors}")
    print(f"   New size: {target_size}x{target_size}")
    
    return processed > 0


def convert_to_coco(val_per_category=10, use_polygons=True):
    """
    Konvertiert Leeds Butterfly Dataset zu COCO Format.
    
    Args:
        val_per_category: Anzahl der Bilder pro Kategorie im Validation Set (Standard: 10)
        use_polygons: Wenn True, werden Polygone verwendet; sonst RLE
    """
    
    dataset_root = Path(__file__).parent
    images_dir = dataset_root / "images"
    seg_dir = dataset_root / "segmentations"
    output_dir = dataset_root / "annotations"
    
    print("\n🔄 CONVERTING TO COCO FORMAT")
    print("=" * 50)
    
    # Erstelle Output-Verzeichnis
    output_dir.mkdir(exist_ok=True)
    
    # Sammle alle Bild-Segmentierungs-Paare und gruppiere nach Kategorie
    pairs_by_category = {cat_id: [] for cat_id in BUTTERFLY_CATEGORIES.keys()}
    images = sorted(list(images_dir.glob("*.png")))
    
    for img_path in images:
        seg_path = seg_dir / f"{img_path.stem}_seg0.png"
        if seg_path.exists():
            cat_id = img_path.stem[:3]  # Kategorie aus Dateinamen (z.B. "001")
            if cat_id in pairs_by_category:
                pairs_by_category[cat_id].append((img_path, seg_path))
    
    total_pairs = sum(len(pairs) for pairs in pairs_by_category.values())
    print(f"📊 Found {total_pairs} valid image-segmentation pairs")
    
    if total_pairs == 0:
        print("❌ No valid pairs found!")
        return False
    
    # COCO Kategorien erstellen (1-basierte IDs)
    categories = []
    for cat_id, cat_name in BUTTERFLY_CATEGORIES.items():
        categories.append({
            "id": int(cat_id),  # 1-10
            "name": cat_name,
            "supercategory": "butterfly"
        })
    
    # Stratifizierter Split: val_per_category Bilder pro Kategorie für Validation
    random.seed(42)
    
    val_pairs = []
    train_pairs = []
    
    print(f"\n📊 Stratified split ({val_per_category} images per category for validation):")
    for cat_id in sorted(pairs_by_category.keys()):
        cat_pairs = pairs_by_category[cat_id]
        random.shuffle(cat_pairs)
        
        # Stelle sicher, dass genug Bilder vorhanden sind
        val_count_for_cat = min(val_per_category, len(cat_pairs))
        
        val_pairs.extend(cat_pairs[:val_count_for_cat])
        train_pairs.extend(cat_pairs[val_count_for_cat:])
        
        cat_name = BUTTERFLY_CATEGORIES[cat_id]
        print(f"   {cat_id} ({cat_name}): {val_count_for_cat} val, {len(cat_pairs) - val_count_for_cat} train")
    
    # Shuffle die finalen Listen für bessere Durchmischung
    random.shuffle(val_pairs)
    random.shuffle(train_pairs)
    
    splits = {
        "train": train_pairs,
        "val": val_pairs
    }
    
    print(f"📊 Split: {len(splits['train'])} train, {len(splits['val'])} val")
    
    # Konvertiere jeden Split
    for split_name, pairs in splits.items():
        if len(pairs) == 0:
            continue
        
        print(f"\n🔄 Processing {split_name} set ({len(pairs)} images)...")
        
        coco_data = {
            "info": {
                "description": f"Leeds Butterfly Dataset - {split_name} set",
                "url": "http://www.josiahwang.com/dataset/leedsbutterfly/",
                "version": "1.0",
                "year": 2009,
                "contributor": "Josiah Wang, Katja Markert, Mark Everingham",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Academic Use",
                    "url": "http://www.josiahwang.com/dataset/leedsbutterfly/"
                }
            ],
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        annotation_id = 1
        processed = 0
        skipped = 0
        
        for image_id, (img_path, seg_path) in enumerate(pairs, start=1):
            try:
                # Lade Bild für Dimensionen
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"   ⚠️  Could not load image: {img_path.name}")
                    skipped += 1
                    continue
                
                height, width = img.shape[:2]
                
                # Kategorie aus Dateinamen extrahieren (erste 3 Zeichen)
                category_id = int(img_path.stem[:3])
                
                # Füge Bild hinzu
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": img_path.name,
                    "width": width,
                    "height": height,
                    "license": 1
                })
                
                # Lade Segmentierungsmaske
                binary_mask = load_segmentation_mask(seg_path)
                if binary_mask is None:
                    print(f"   ⚠️  Could not load segmentation: {seg_path.name}")
                    skipped += 1
                    continue
                
                # Berechne Bounding Box und Fläche
                bbox = calculate_bbox_from_mask(binary_mask)
                if bbox is None:
                    print(f"   ⚠️  Empty mask: {seg_path.name}")
                    skipped += 1
                    continue
                
                area = calculate_area_from_mask(binary_mask)
                
                # Erstelle Annotation
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0
                }
                
                # Füge Segmentierung hinzu
                if use_polygons:
                    polygons = mask_to_polygons(binary_mask)
                    if polygons:
                        annotation["segmentation"] = polygons
                    else:
                        # Fallback auf RLE wenn keine Polygone gefunden
                        rle = mask_to_rle(binary_mask)
                        annotation["segmentation"] = rle
                else:
                    rle = mask_to_rle(binary_mask)
                    annotation["segmentation"] = rle
                
                coco_data["annotations"].append(annotation)
                annotation_id += 1
                processed += 1
                
                if processed % 100 == 0:
                    print(f"   Processed {processed} images...")
                
            except Exception as e:
                print(f"   ⚠️  Error processing {img_path.name}: {e}")
                skipped += 1
                continue
        
        # Speichere COCO-Datei
        output_file = output_dir / f"instances_{split_name}2017.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"   ✅ Saved: {output_file}")
        print(f"      📊 Images: {len(coco_data['images'])}")
        print(f"      📊 Annotations: {len(coco_data['annotations'])}")
        print(f"      📊 Categories: {len(coco_data['categories'])}")
        print(f"      ⚠️  Skipped: {skipped}")
    
    return True


def verify_conversion():
    """Überprüft die COCO-Konvertierung."""
    
    dataset_root = Path(__file__).parent
    annotations_dir = dataset_root / "annotations"
    images_dir = dataset_root / "images"
    
    print("\n🔍 VERIFYING COCO CONVERSION")
    print("=" * 50)
    
    if not annotations_dir.exists():
        print("❌ Annotations directory not found!")
        return False
    
    coco_files = list(annotations_dir.glob("instances_*.json"))
    
    if not coco_files:
        print("❌ No COCO annotation files found!")
        return False
    
    print(f"✅ Found {len(coco_files)} COCO files")
    
    total_images = 0
    total_annotations = 0
    success = True
    
    for coco_file in coco_files:
        print(f"\n📄 {coco_file.name}:")
        
        try:
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Validiere Struktur
            required_keys = ['images', 'annotations', 'categories']
            missing_keys = [key for key in required_keys if key not in coco_data]
            
            if missing_keys:
                print(f"   ❌ Missing keys: {missing_keys}")
                success = False
                continue
            
            images_count = len(coco_data['images'])
            annotations_count = len(coco_data['annotations'])
            categories_count = len(coco_data['categories'])
            
            print(f"   📊 Images: {images_count}")
            print(f"   📊 Annotations: {annotations_count}")
            print(f"   📊 Categories: {categories_count}")
            
            # Zeige Kategorien
            print(f"   🦋 Categories:")
            for cat in coco_data['categories']:
                print(f"      {cat['id']:2d}: {cat['name']}")
            
            # Beispiel-Annotation
            if coco_data['annotations']:
                sample_ann = coco_data['annotations'][0]
                print(f"   📋 Sample annotation:")
                print(f"      Image ID: {sample_ann['image_id']}")
                print(f"      Category ID: {sample_ann['category_id']}")
                print(f"      Bbox: [{sample_ann['bbox'][0]:.1f}, {sample_ann['bbox'][1]:.1f}, "
                      f"{sample_ann['bbox'][2]:.1f}, {sample_ann['bbox'][3]:.1f}]")
                print(f"      Area: {sample_ann['area']}")
                
                if 'segmentation' in sample_ann:
                    seg = sample_ann['segmentation']
                    if isinstance(seg, list):
                        total_points = sum(len(poly) // 2 for poly in seg)
                        print(f"      Segmentation: {len(seg)} polygon(s), {total_points} points total")
                    elif isinstance(seg, dict):
                        print(f"      Segmentation: RLE format")
            
            # Prüfe ob Bilder existieren
            missing_images = 0
            for img_info in coco_data['images'][:10]:  # Prüfe erste 10
                img_path = images_dir / img_info['file_name']
                if not img_path.exists():
                    missing_images += 1
            
            if missing_images > 0:
                print(f"   ⚠️  {missing_images} images not found in first 10 checked")
            else:
                print(f"   ✅ All checked images exist")
            
            total_images += images_count
            total_annotations += annotations_count
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            success = False
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total Images: {total_images}")
    print(f"   Total Annotations: {total_annotations}")
    
    if success:
        print(f"\n🎉 COCO conversion successful!")
        print(f"📁 Annotations saved in: {annotations_dir}")
        print(f"🚀 Ready for MaskDINO training!")
        print(f"\n💡 Usage hints for MaskDINO:")
        print(f"   - Set dataset root to: {dataset_root}")
        print(f"   - Images are in: images/")
        print(f"   - Annotations are in: annotations/")
    
    return success


def create_symlinks_for_maskdino():
    """
    Erstellt symbolische Links für die MaskDINO-Verzeichnisstruktur.
    MaskDINO erwartet: coco/train2017, coco/val2017, coco/annotations
    """
    dataset_root = Path(__file__).parent
    coco_root = dataset_root / "coco"
    
    print("\n🔗 CREATING MASKDINO-COMPATIBLE STRUCTURE")
    print("=" * 50)
    
    # Erstelle coco-Verzeichnis
    coco_root.mkdir(exist_ok=True)
    
    # Da alle Bilder in images/ sind, erstellen wir Symlinks
    # für train2017 und val2017 zum selben images-Verzeichnis
    images_dir = dataset_root / "images"
    annotations_dir = dataset_root / "annotations"
    
    links_to_create = [
        (images_dir, coco_root / "train2017"),
        (images_dir, coco_root / "val2017"),
        (annotations_dir, coco_root / "annotations"),
    ]
    
    for source, target in links_to_create:
        if target.exists():
            if target.is_symlink():
                target.unlink()
            else:
                print(f"   ⚠️  {target.name} exists and is not a symlink, skipping")
                continue
        
        try:
            target.symlink_to(source)
            print(f"   ✅ Created symlink: {target.name} -> {source.name}")
        except OSError as e:
            print(f"   ⚠️  Could not create symlink: {e}")
            print(f"      You may need to copy the files manually")
    
    print(f"\n📁 MaskDINO-compatible structure created in: {coco_root}")
    print(f"   Use this as your COCO dataset root for MaskDINO")


if __name__ == "__main__":
    print("🦋 LEEDS BUTTERFLY TO COCO CONVERTER")
    print("=" * 60)
    print()
    
    # Schritt 1: Bilder auf 256x256 quadratisch zuschneiden
    print("STEP 1: Resizing images to 256x256 square")
    print("-" * 40)
    resize_images_to_square(target_size=256)
    
    # Schritt 2: Analyse
    print("\n\nSTEP 2: Analyzing dataset")
    print("-" * 40)
    analyze_dataset()
    
    # Schritt 3: Konvertierung
    print("\n\nSTEP 3: Converting to COCO format")
    print("-" * 40)
    success = convert_to_coco(val_per_category=10, use_polygons=True)
    
    # Schritt 4: Verifikation
    print("\n\nSTEP 4: Verifying conversion")
    print("-" * 40)
    verify_conversion()
    
    # Schritt 5: MaskDINO-Struktur
    print("\n\nSTEP 5: Creating MaskDINO-compatible structure")
    print("-" * 40)
    create_symlinks_for_maskdino()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ All steps completed successfully!")
        print("\n📝 Next steps for MaskDINO training:")
        print("   1. Register the dataset in MaskDINO's dataset registry")
        print("   2. Update config to point to leedsbutterfly/coco/")
        print("   3. Set num_classes=10 in your config")
    else:
        print("❌ Some steps failed. Please check the output above.")
