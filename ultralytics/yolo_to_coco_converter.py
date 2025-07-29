#!/usr/bin/env python3
"""
YOLOv8 to COCO Format Converter
Converts YOLOv8 dataset format to COCO format for use with MaskDINO
"""

import json
import os
from pathlib import Path
import cv2
import yaml
from datetime import datetime
import random

def analyze_dataset_structure():
    """Analysiert die YOLOv8-Dataset-Struktur"""
    
    dataset_root = Path("/Users/nicklehmacher/Alles/MasterArbeit/ultralytics/datasets")
    print(f"🔍 Analyzing dataset structure in: {dataset_root}")
    print("=" * 60)
    
    # Überprüfe Hauptverzeichnisse
    for item in dataset_root.iterdir():
        if item.is_dir():
            print(f"📁 Directory: {item.name}")
            # Zeige Inhalt der ersten Ebene
            try:
                contents = list(item.iterdir())[:5]  # Erste 5 Dateien/Ordner
                for content in contents:
                    if content.is_file():
                        print(f"   📄 {content.name}")
                    else:
                        print(f"   📁 {content.name}/")
                if len(list(item.iterdir())) > 5:
                    print(f"   ... and {len(list(item.iterdir())) - 5} more items")
            except PermissionError:
                print("   ❌ Permission denied")
            print()
        elif item.is_file():
            print(f"📄 File: {item.name}")
    
    # Suche nach typischen YOLOv8-Dateien
    print("\n🔍 Looking for YOLOv8 specific files:")
    
    # data.yaml suchen
    yaml_files = list(dataset_root.rglob("*.yaml"))
    if yaml_files:
        print(f"📄 Found YAML files: {[f.name for f in yaml_files]}")
        
        # Lade erste YAML-Datei
        try:
            with open(yaml_files[0], 'r') as f:
                yaml_content = yaml.safe_load(f)
            print(f"📋 YAML content preview:")
            for key, value in yaml_content.items():
                print(f"   {key}: {value}")
        except Exception as e:
            print(f"   ❌ Error reading YAML: {e}")
    
    # Bildverzeichnisse finden
    print(f"\n🖼️  Looking for image directories:")
    image_dirs = []
    for path in dataset_root.rglob("*"):
        if path.is_dir():
            image_count = len([f for f in path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            if image_count > 0:
                print(f"   📁 {path.relative_to(dataset_root)}: {image_count} images")
                image_dirs.append(path)
    
    # Label-Verzeichnisse finden
    print(f"\n🏷️  Looking for label directories:")
    label_dirs = []
    for path in dataset_root.rglob("*"):
        if path.is_dir():
            label_count = len([f for f in path.iterdir() if f.suffix.lower() == '.txt'])
            if label_count > 0:
                print(f"   📁 {path.relative_to(dataset_root)}: {label_count} labels")
                label_dirs.append(path)
    
    # Beispiel-Label analysieren
    if label_dirs:
        print(f"\n📋 Analyzing sample label format:")
        sample_labels = list(label_dirs[0].iterdir())[:3]
        for label_file in sample_labels:
            if label_file.suffix == '.txt':
                print(f"   📄 {label_file.name}:")
                with open(label_file, 'r') as f:
                    lines = f.readlines()[:2]  # Erste 2 Zeilen
                    for i, line in enumerate(lines):
                        parts = line.strip().split()
                        if len(parts) > 5:
                            # Segmentation format
                            print(f"      Line {i+1}: class_id={parts[0]}, {len(parts[1:])//2} polygon points")
                        elif len(parts) == 5:
                            # Bounding box format
                            print(f"      Line {i+1}: class_id={parts[0]}, bbox=[{', '.join(parts[1:5])}]")
                        else:
                            print(f"      Line {i+1}: {line.strip()}")
                break
    
    return image_dirs, label_dirs

def yolo_to_coco_converter():
    """Konvertiert YOLOv8-Format zu COCO-Format"""
    
    dataset_root = Path("/Users/nicklehmacher/Alles/MasterArbeit/ultralytics/datasets")
    
    print("\n🔄 Starting YOLOv8 to COCO conversion...")
    print("=" * 50)
    
    # Suche nach data.yaml für Klassendefinitionen
    classes = []
    yaml_files = list(dataset_root.rglob("*.yaml"))
    
    if yaml_files:
        try:
            with open(yaml_files[0], 'r') as f:
                yaml_content = yaml.safe_load(f)
            
            if 'names' in yaml_content:
                if isinstance(yaml_content['names'], dict):
                    classes = [yaml_content['names'][i] for i in sorted(yaml_content['names'].keys())]
                elif isinstance(yaml_content['names'], list):
                    classes = yaml_content['names']
                print(f"📋 Found {len(classes)} classes: {classes[:5]}..." if len(classes) > 5 else f"📋 Found {len(classes)} classes: {classes}")
            else:
                print("⚠️  No 'names' found in YAML, analyzing labels for classes...")
                # Analysiere Label-Dateien für Klassen-IDs
                all_labels = list(dataset_root.rglob("*.txt"))
                class_ids = set()
                for label_file in all_labels[:100]:  # Prüfe erste 100 Label-Dateien
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_ids.add(int(parts[0]))
                    except:
                        continue
                
                max_class_id = max(class_ids) if class_ids else 0
                classes = [f"class_{i}" for i in range(max_class_id + 1)]
                print(f"📋 Detected {len(classes)} classes from labels: {classes}")
                
        except Exception as e:
            print(f"⚠️  Error reading YAML: {e}")
            classes = ['carpart']
    else:
        print("⚠️  No YAML file found, using default class")
        classes = ['carpart']
    
    # COCO-Kategorien erstellen (1-basierte IDs für COCO)
    categories = []
    for i, class_name in enumerate(classes):
        categories.append({
            "id": i + 1,  # COCO verwendet 1-basierte IDs
            "name": class_name,
            "supercategory": "object"
        })
    
    # Alle Bilder und Labels finden
    print("🔍 Finding all images and labels...")
    all_images = []
    
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        all_images.extend(dataset_root.rglob(f"*{ext}"))
    
    print(f"📊 Found {len(all_images)} images")
    
    # Matche Bilder mit Labels
    matched_pairs = []
    for img_path in all_images:
        # Versuche verschiedene Label-Pfade
        possible_label_paths = [
            img_path.parent / 'labels' / f"{img_path.stem}.txt",  # images/train -> labels/train
            img_path.parent.parent / 'labels' / img_path.parent.name / f"{img_path.stem}.txt",  # ../labels/train/
            dataset_root / 'labels' / img_path.parent.name / f"{img_path.stem}.txt",  # root/labels/train/
            dataset_root / 'labels' / f"{img_path.stem}.txt",  # root/labels/
        ]
        
        for label_path in possible_label_paths:
            if label_path.exists():
                matched_pairs.append((img_path, label_path))
                break
    
    print(f"📊 Matched {len(matched_pairs)} image-label pairs")
    
    if len(matched_pairs) == 0:
        print("❌ No matching image-label pairs found!")
        return
    
    # Verwende existierende train/val Aufteilung falls vorhanden
    train_pairs = []
    val_pairs = []
    
    for img_path, label_path in matched_pairs:
        if 'train' in str(img_path):
            train_pairs.append((img_path, label_path))
        elif 'val' in str(img_path):
            val_pairs.append((img_path, label_path))
        else:
            # Falls keine klare Aufteilung, füge zu train hinzu
            train_pairs.append((img_path, label_path))
    
    # Falls keine val-Daten gefunden, erstelle 80/20 Split
    if len(val_pairs) == 0:
        print("📊 No existing train/val split found, creating 80/20 split...")
        random.seed(42)
        random.shuffle(matched_pairs)
        train_split = int(0.8 * len(matched_pairs))
        train_pairs = matched_pairs[:train_split]
        val_pairs = matched_pairs[train_split:]
    
    splits = {
        'train': train_pairs,
        'val': val_pairs
    }
    
    print(f"📊 Final split: {len(train_pairs)} train, {len(val_pairs)} val")
    
    # Erstelle output directory
    output_dir = dataset_root / "annotations"
    output_dir.mkdir(exist_ok=True)
    
    # Konvertiere jeden Split
    for split_name, pairs in splits.items():
        if len(pairs) == 0:
            continue
            
        print(f"\n🔄 Converting {split_name} set...")
        
        # COCO-Datenstruktur initialisieren
        coco_data = {
            "info": {
                "description": f"Converted from YOLOv8 format - {split_name} set",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "YOLOv8 to COCO Converter",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Custom License",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": categories
        }
        
        annotation_id = 1
        image_id = 1
        
        print(f"   Processing {len(pairs)} image-label pairs...")
        
        processed_count = 0
        for img_path, label_path in pairs:
            try:
                # Lade Bild für Dimensionen
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"   ⚠️  Skipping {img_path.name}: Could not load image")
                    continue
                
                height, width = img.shape[:2]
                
                # Bestimme relativen Pfad für file_name
                # Finde den relativen Pfad vom images-Verzeichnis aus
                images_root = dataset_root / "images"
                try:
                    relative_path = img_path.relative_to(images_root)
                    file_name = str(relative_path).replace("\\", "/")  # Für Windows-Kompatibilität
                except ValueError:
                    # Falls Bild nicht im images-Verzeichnis, verwende nur Dateinamen
                    file_name = img_path.name
                
                # Füge Bild zur COCO-Struktur hinzu
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "license": 1
                })
                
                # Lade und konvertiere YOLO-Annotationen
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5:  # Mindestens class_id + 4 Koordinaten
                        continue
                    
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    
                    if len(coords) == 4:
                        # Bounding Box Format: x_center y_center width height (normalisiert)
                        x_center, y_center, bbox_width, bbox_height = coords
                        
                        # Konvertiere normalisierte Koordinaten zu Pixeln
                        x_center *= width
                        y_center *= height
                        bbox_width *= width
                        bbox_height *= height
                        
                        # Konvertiere zu x_min, y_min, width, height
                        x_min = x_center - bbox_width / 2
                        y_min = y_center - bbox_height / 2
                        
                        area = bbox_width * bbox_height
                        
                        # Füge Annotation hinzu
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + 1,  # Konvertiere zu 1-basiert
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": area,
                            "iscrowd": 0
                        })
                        
                    else:
                        # Segmentation Format: x1 y1 x2 y2 ... xn yn (normalisiert)
                        if len(coords) % 2 != 0:
                            print(f"   ⚠️  Odd number of coordinates in {label_path.name}")
                            continue
                        
                        # Konvertiere normalisierte Koordinaten zu Pixeln
                        pixel_coords = []
                        for i in range(0, len(coords), 2):
                            x = coords[i] * width
                            y = coords[i+1] * height
                            pixel_coords.extend([x, y])
                        
                        # Berechne Bounding Box aus Polygon
                        x_coords = pixel_coords[::2]
                        y_coords = pixel_coords[1::2]
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min
                        
                        # Berechne Polygon-Fläche (Shoelace-Formel)
                        area = 0
                        n = len(x_coords)
                        for i in range(n):
                            j = (i + 1) % n
                            area += x_coords[i] * y_coords[j]
                            area -= x_coords[j] * y_coords[i]
                        area = abs(area) / 2
                        
                        # Füge Segmentation-Annotation hinzu
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": class_id + 1,  # Konvertiere zu 1-basiert
                            "bbox": [x_min, y_min, bbox_width, bbox_height],
                            "area": area,
                            "segmentation": [pixel_coords],
                            "iscrowd": 0
                        })
                    
                    annotation_id += 1
                
                image_id += 1
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"     Processed {processed_count} images...")
                
            except Exception as e:
                print(f"   ⚠️  Error processing {img_path.name}: {e}")
                continue
        
        # Speichere COCO-Datei
        output_file = output_dir / f"instances_{split_name}2017.json"
        with open(output_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"   ✅ Saved {output_file}")
        print(f"      📊 Images: {len(coco_data['images'])}")
        print(f"      📊 Annotations: {len(coco_data['annotations'])}")
        print(f"      📊 Categories: {len(coco_data['categories'])}")

def verify_coco_conversion():
    """Überprüft die erfolgreiche COCO-Konvertierung"""
    
    dataset_root = Path("/Users/nicklehmacher/Alles/MasterArbeit/ultralytics/datasets")
    annotations_dir = dataset_root / "annotations"
    
    print("\n🔍 Verifying COCO conversion...")
    print("=" * 40)
    
    if not annotations_dir.exists():
        print("❌ Annotations directory not found!")
        return False
    
    # Überprüfe COCO-Dateien
    coco_files = list(annotations_dir.glob("instances_*.json"))
    
    if not coco_files:
        print("❌ No COCO annotation files found!")
        return False
    
    print(f"✅ Found {len(coco_files)} COCO files:")
    
    total_images = 0
    total_annotations = 0
    success = True
    
    for coco_file in coco_files:
        print(f"\n📄 {coco_file.name}:")
        
        try:
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Validiere COCO-Struktur
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
            print(f"   🏷️  Categories:")
            for cat in coco_data['categories'][:5]:  # Erste 5 Kategorien
                print(f"      {cat['id']}: {cat['name']}")
            if len(coco_data['categories']) > 5:
                print(f"      ... and {len(coco_data['categories']) - 5} more")
            
            # Beispiel-Annotation
            if coco_data['annotations']:
                sample_ann = coco_data['annotations'][0]
                print(f"   📋 Sample annotation:")
                print(f"      Image ID: {sample_ann['image_id']}")
                print(f"      Category ID: {sample_ann['category_id']}")
                print(f"      Bbox: [{sample_ann['bbox'][0]:.1f}, {sample_ann['bbox'][1]:.1f}, {sample_ann['bbox'][2]:.1f}, {sample_ann['bbox'][3]:.1f}]")
                if 'segmentation' in sample_ann and sample_ann['segmentation']:
                    seg_points = len(sample_ann['segmentation'][0]) // 2
                    print(f"      Segmentation: {seg_points} points")
                print(f"      Area: {sample_ann['area']:.1f}")
            
            total_images += images_count
            total_annotations += annotations_count
            
            print(f"   ✅ Valid COCO format")
            
        except Exception as e:
            print(f"   ❌ Error reading {coco_file.name}: {e}")
            success = False
            continue
    
    print(f"\n📊 CONVERSION SUMMARY:")
    print(f"   Total Images: {total_images}")
    print(f"   Total Annotations: {total_annotations}")
    print(f"   Average Annotations per Image: {total_annotations/total_images:.2f}" if total_images > 0 else "   No images found")
    
    # Überprüfe ob Bilder erreichbar sind
    if coco_files and total_images > 0:
        print(f"\n🔍 Checking image accessibility...")
        sample_coco_file = coco_files[0]
        with open(sample_coco_file, 'r') as f:
            sample_data = json.load(f)
        
        if sample_data['images']:
            sample_image = sample_data['images'][0]
            image_name = sample_image['file_name']
            
            # Suche Bild in verschiedenen Verzeichnissen
            possible_image_paths = [
                dataset_root / 'images' / 'train' / image_name,
                dataset_root / 'images' / 'val' / image_name,
                dataset_root / 'images' / image_name,
            ]
            
            image_found = False
            for img_path in possible_image_paths:
                if img_path.exists():
                    print(f"   ✅ Sample image found: {img_path.relative_to(dataset_root)}")
                    image_found = True
                    break
            
            if not image_found:
                print(f"   ⚠️  Sample image not found: {image_name}")
                print(f"   💡 Make sure image paths in your training script match the file locations")
    
    if success and total_images > 0:
        print(f"\n🎉 COCO conversion successful!")
        print(f"   📁 Annotations saved in: {annotations_dir}")
        print(f"   🚀 Ready for MaskDINO training!")
    else:
        print(f"\n❌ COCO conversion failed or incomplete!")
    
    return success and total_images > 0

if __name__ == "__main__":
    # Schritt 1: Analysiere Dataset-Struktur
    print("STEP 1: Analyzing YOLOv8 dataset structure")
    print("=" * 60)
    image_dirs, label_dirs = analyze_dataset_structure()
    
    # Schritt 2: Konvertiere zu COCO
    print("\n\nSTEP 2: Converting to COCO format")
    print("=" * 60)
    yolo_to_coco_converter()
    
    # Schritt 3: Überprüfe Konvertierung
    print("\n\nSTEP 3: Verifying conversion")
    print("=" * 60)
    success = verify_coco_conversion()
    
    if success:
        print(f"\n✅ All steps completed successfully!")
    else:
        print(f"\n❌ Some steps failed. Please check the output above.")
