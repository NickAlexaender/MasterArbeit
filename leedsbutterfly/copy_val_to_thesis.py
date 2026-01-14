"""
Kopiert alle Bilder aus dem COCO Validation Set in den myThesis/image/butterfly/1images Ordner.
Verwendet die instances_val2017.json um die korrekten Validation-Bilder zu identifizieren.
Erstellt zusätzlich binäre Masken für jedes Bild in myThesis/image/butterfly/2masken.
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np


def create_binary_mask_from_annotations(annotations, image_width, image_height):
    """
    Erstellt eine binäre Maske aus allen Annotation-Segmentierungen für ein Bild.
    Gibt eine binäre Maske zurück, bei der 255 bedeutet, dass das Pixel zu einer Klasse gehört.
    """
    # Erstelle leere Maske
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    
    for ann in annotations:
        segmentation = ann.get('segmentation', [])
        
        # Handle Polygon-Format (Liste von Listen)
        if isinstance(segmentation, list):
            for polygon in segmentation:
                if isinstance(polygon, list) and len(polygon) >= 6:
                    # Jedes Polygon ist eine flache Liste von [x1, y1, x2, y2, ...]
                    # Konvertiere zu Liste von Tupeln [(x1, y1), (x2, y2), ...]
                    points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    draw.polygon(points, fill=255)
    
    return mask


def copy_validation_images_to_thesis():
    """
    Kopiert alle Bilder aus dem COCO Validation Set (basierend auf instances_val2017.json) 
    in den myThesis/image/butterfly/1images Ordner.
    Erstellt zusätzlich binäre Masken in myThesis/image/butterfly/2masken.
    """
    # Pfade definieren
    base_dir = Path(__file__).parent.parent  # MasterArbeit Ordner
    
    annotations_file = base_dir / "leedsbutterfly" / "coco" / "annotations" / "instances_val2017.json"
    source_dir = base_dir / "leedsbutterfly" / "coco" / "val2017"
    target_dir = base_dir / "myThesis" / "image" / "butterfly" / "1images"
    masks_dir = base_dir / "myThesis" / "image" / "butterfly" / "2masken"
    
    # Prüfe ob Annotations-Datei existiert
    if not annotations_file.exists():
        print(f"FEHLER: Annotations-Datei nicht gefunden: {annotations_file}")
        return
    
    # Prüfe ob Quellordner existiert
    if not source_dir.exists():
        print(f"FEHLER: Quellordner nicht gefunden: {source_dir}")
        return
    
    # Lade die Validation-Annotations
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Extrahiere die Dateinamen aus der JSON
    validation_images = {img['file_name']: img for img in coco_data['images']}
    
    # Erstelle eine Zuordnung von image_id zu Annotations
    image_id_to_annotations = {}
    for ann in coco_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id not in image_id_to_annotations:
            image_id_to_annotations[img_id] = []
        image_id_to_annotations[img_id].append(ann)
    
    # Erstelle Zielordner falls nicht vorhanden
    target_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Kopiere Validation Images und erstelle Masken ===")
    print(f"Annotations-Datei: {annotations_file}")
    print(f"Quellordner: {source_dir}")
    print(f"Zielordner Bilder: {target_dir}")
    print(f"Zielordner Masken: {masks_dir}")
    print(f"Bilder im Validation-Set (laut JSON): {len(validation_images)}")
    print()
    
    # Kopiere die Bilder und erstelle Masken
    copied_count = 0
    skipped_count = 0
    not_found_count = 0
    masks_created = 0
    
    for image_name in sorted(validation_images.keys()):
        img_info = validation_images[image_name]
        source_path = source_dir / image_name
        target_path = target_dir / image_name
        
        # Masken-Pfad: gleicher Name, aber mit _mask Suffix und .png
        mask_name = Path(image_name).stem + "_mask.png"
        mask_path = masks_dir / mask_name
        
        # Prüfe ob Quelldatei existiert
        if not source_path.exists():
            print(f"  ⚠ Nicht gefunden: {image_name}")
            not_found_count += 1
            continue
        
        # Kopiere Bild falls nicht vorhanden
        if not target_path.exists():
            try:
                shutil.copy2(source_path, target_path)
                print(f"  ✓ Kopiert: {image_name}")
                copied_count += 1
            except Exception as e:
                print(f"  ✗ FEHLER beim Kopieren von {image_name}: {e}")
                continue
        else:
            print(f"  Übersprungen (existiert bereits): {image_name}")
            skipped_count += 1
        
        # Erstelle binäre Maske
        if not mask_path.exists():
            try:
                # Hole Annotationen für dieses Bild
                image_id = img_info['id']
                annotations = image_id_to_annotations.get(image_id, [])
                
                # Erstelle Maske
                width = img_info.get('width', 256)
                height = img_info.get('height', 256)
                mask = create_binary_mask_from_annotations(annotations, width, height)
                
                # Speichere Maske
                mask.save(mask_path)
                print(f"  ✓ Maske erstellt: {mask_name}")
                masks_created += 1
            except Exception as e:
                print(f"  ✗ FEHLER beim Erstellen der Maske für {image_name}: {e}")
        else:
            print(f"  Maske existiert bereits: {mask_name}")
    
    print()
    print("=== ZUSAMMENFASSUNG ===")
    print(f"Erfolgreich kopiert: {copied_count} Bilder")
    print(f"Übersprungen (existierten bereits): {skipped_count} Bilder")
    print(f"Nicht gefunden: {not_found_count} Bilder")
    print(f"Masken erstellt: {masks_created}")
    print(f"Gesamt Bilder im Zielordner: {len(list(target_dir.glob('*')))}")
    print(f"Gesamt Masken im Zielordner: {len(list(masks_dir.glob('*.png')))}")
    print(f"\nBilder gespeichert in: {target_dir}")
    print(f"Masken gespeichert in: {masks_dir}")


def get_validation_image_count():
    """
    Gibt die Anzahl der Bilder im Validation Set zurück.
    """
    base_dir = Path(__file__).parent.parent
    source_dir = base_dir / "leedsbutterfly" / "coco" / "val2017"
    
    if not source_dir.exists():
        return 0
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    return len([f for f in source_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions])


if __name__ == "__main__":
    copy_validation_images_to_thesis()
