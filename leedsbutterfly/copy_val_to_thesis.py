"""
Kopiert alle Bilder aus dem COCO Validation Set in den myThesis/image/butterfly/1images Ordner.
Verwendet die instances_val2017.json um die korrekten Validation-Bilder zu identifizieren.
"""

import json
import os
import shutil
from pathlib import Path


def copy_validation_images_to_thesis():
    """
    Kopiert alle Bilder aus dem COCO Validation Set (basierend auf instances_val2017.json) 
    in den myThesis/image/butterfly/1images Ordner.
    """
    # Pfade definieren
    base_dir = Path(__file__).parent.parent  # MasterArbeit Ordner
    
    annotations_file = base_dir / "leedsbutterfly" / "coco" / "annotations" / "instances_val2017.json"
    source_dir = base_dir / "leedsbutterfly" / "coco" / "val2017"
    target_dir = base_dir / "myThesis" / "image" / "butterfly" / "1images"
    
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
    validation_images = {img['file_name'] for img in coco_data['images']}
    
    # Erstelle Zielordner falls nicht vorhanden
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== Kopiere Validation Images nach myThesis ===")
    print(f"Annotations-Datei: {annotations_file}")
    print(f"Quellordner: {source_dir}")
    print(f"Zielordner: {target_dir}")
    print(f"Bilder im Validation-Set (laut JSON): {len(validation_images)}")
    print()
    
    # Kopiere nur die Bilder, die im Validation-Set sind
    copied_count = 0
    skipped_count = 0
    not_found_count = 0
    
    for image_name in sorted(validation_images):
        source_path = source_dir / image_name
        target_path = target_dir / image_name
        
        # Prüfe ob Quelldatei existiert
        if not source_path.exists():
            print(f"  ⚠ Nicht gefunden: {image_name}")
            not_found_count += 1
            continue
        
        # Prüfe ob Datei bereits existiert
        if target_path.exists():
            print(f"  Übersprungen (existiert bereits): {image_name}")
            skipped_count += 1
            continue
        
        try:
            shutil.copy2(source_path, target_path)
            print(f"  ✓ Kopiert: {image_name}")
            copied_count += 1
        except Exception as e:
            print(f"  ✗ FEHLER beim Kopieren von {image_name}: {e}")
    
    print()
    print("=== ZUSAMMENFASSUNG ===")
    print(f"Erfolgreich kopiert: {copied_count} Bilder")
    print(f"Übersprungen (existierten bereits): {skipped_count} Bilder")
    print(f"Nicht gefunden: {not_found_count} Bilder")
    print(f"Gesamt im Zielordner: {len(list(target_dir.glob('*')))}")
    print(f"\nBilder gespeichert in: {target_dir}")


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
