"""
Skript zur Analyse der Farbverteilung in maskierten Bildbereichen.

Berechnet für jedes Bild, wie viele Pixel innerhalb der Maske welcher der 12 Farben
zugeordnet werden. Erstellt CSV-Dateien mit Übersicht über die Farbverteilung.
"""

import cv2
import numpy as np
import os
import csv
from collections import defaultdict


def get_color_ranges():
    """
    Definiert HSV-Farbbereiche für die 12 Farben.
    
    Returns:
        dict: Dictionary mit Farbnamen als Schlüssel und HSV-Bereichen als Werten
    """
    color_ranges = {
        'rot': [
            ([0, 50, 50], [10, 255, 255]),      # Unterer roter Bereich
            ([170, 50, 50], [180, 255, 255])    # Oberer roter Bereich
        ],
        'gruen': [
            ([40, 50, 50], [80, 255, 255])      # Grüner Bereich
        ],
        'blau': [
            ([100, 50, 50], [130, 255, 255])    # Blauer Bereich
        ],
        'gelb': [
            ([20, 50, 50], [35, 255, 255])      # Gelber Bereich
        ],
        'orange': [
            ([10, 50, 50], [20, 255, 255])      # Oranger Bereich
        ],
        'lila': [
            ([130, 50, 50], [170, 255, 255])    # Lila/Violett Bereich
        ],
        'cyan': [
            ([80, 50, 50], [100, 255, 255])     # Cyan/Türkis Bereich
        ],
        'pink': [
            ([140, 50, 50], [180, 255, 255]),   # Pink Bereich (ähnlich wie Magenta)
            ([0, 50, 50], [10, 255, 200])       # Rosa Töne
        ],
        'braun': [
            ([8, 50, 20], [20, 255, 200])       # Brauner Bereich (niedriger Wert)
        ],
        'schwarz': [],  # Wird separat behandelt
        'weiss': [],    # Wird separat behandelt
        'grau': []      # Wird separat behandelt
    }
    return color_ranges


def classify_pixel_color(hsv_pixel, color_ranges):
    """
    Ordnet einen einzelnen HSV-Pixel einer der 12 Farben zu.
    Jeder Pixel wird genau einer Farbe zugeordnet (Prioritätsbasiert).
    
    Args:
        hsv_pixel: HSV-Werte des Pixels [H, S, V]
        color_ranges: Dictionary mit Farbbereichen
        
    Returns:
        str: Name der zugeordneten Farbe
    """
    h, s, v = hsv_pixel
    
    # Priorität 1: Schwarz (niedriger V-Wert)
    if v < 50:
        return 'schwarz'
    
    # Priorität 2: Weiß (hoher V-Wert, niedrige Sättigung)
    if v >= 200 and s < 30:
        return 'weiss'
    
    # Priorität 3: Grau (mittlerer V-Wert, niedrige Sättigung)
    if s < 30 and 50 <= v < 200:
        return 'grau'
    
    # Priorität 4: Chromatic colors basierend auf Hue
    # Rot (umfasst beide Enden des Hue-Spektrums)
    if (0 <= h <= 10) or (170 <= h <= 180):
        # Unterscheidung zwischen Rot, Orange und Pink basierend auf S und V
        if s >= 50 and v >= 50:
            if 10 < h <= 20:
                return 'orange'
            return 'rot'
    
    # Orange
    if 10 < h <= 20 and s >= 50 and v >= 50:
        return 'orange'
    
    # Gelb
    if 20 < h <= 35 and s >= 50 and v >= 50:
        return 'gelb'
    
    # Grün
    if 35 < h <= 80 and s >= 50 and v >= 50:
        return 'gruen'
    
    # Cyan
    if 80 < h <= 100 and s >= 50 and v >= 50:
        return 'cyan'
    
    # Blau
    if 100 < h <= 130 and s >= 50 and v >= 50:
        return 'blau'
    
    # Lila/Violett
    if 130 < h <= 150 and s >= 50 and v >= 50:
        return 'lila'
    
    # Pink/Magenta
    if 150 < h <= 170 and s >= 50 and v >= 50:
        return 'pink'
    
    # Braun (niedrigere Sättigung und Value im orange-roten Bereich)
    if 8 <= h <= 20 and 50 <= s <= 200 and 20 <= v <= 200:
        return 'braun'
    
    # Fallback: Basierend auf dominantem Hue-Bereich
    if h <= 10 or h >= 170:
        return 'rot'
    elif h <= 20:
        return 'orange'
    elif h <= 35:
        return 'gelb'
    elif h <= 80:
        return 'gruen'
    elif h <= 100:
        return 'cyan'
    elif h <= 130:
        return 'blau'
    elif h <= 150:
        return 'lila'
    else:
        return 'pink'


def analyze_masked_colors(image_path, mask_path):
    """
    Analysiert die Farbverteilung innerhalb der maskierten Bereiche eines Bildes.
    
    Args:
        image_path: Pfad zum Originalbild
        mask_path: Pfad zur Maske
        
    Returns:
        dict: Dictionary mit Farbzählungen {farbe: anzahl_pixel}
    """
    # Bild und Maske laden
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warnung: Konnte Bild nicht laden: {image_path}")
        return None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warnung: Konnte Maske nicht laden: {mask_path}")
        return None
    
    # Sicherstellen, dass Maske und Bild die gleiche Größe haben
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Zu HSV konvertieren
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Farbbereiche laden
    color_ranges = get_color_ranges()
    
    # Initialisiere Zähler für alle 12 Farben
    color_counts = {
        'rot': 0,
        'orange': 0,
        'gelb': 0,
        'gruen': 0,
        'cyan': 0,
        'blau': 0,
        'lila': 0,
        'pink': 0,
        'braun': 0,
        'schwarz': 0,
        'weiss': 0,
        'grau': 0
    }
    
    # Alle Pixel durchgehen, die in der Maske sind (Maske > 0)
    mask_indices = np.where(mask > 0)
    
    for y, x in zip(mask_indices[0], mask_indices[1]):
        hsv_pixel = hsv[y, x]
        color = classify_pixel_color(hsv_pixel, color_ranges)
        color_counts[color] += 1
    
    return color_counts


def process_dataset(images_dir, masks_dir, output_csv_path, image_extension='.png'):
    """
    Verarbeitet einen kompletten Datensatz und erstellt eine CSV mit der Farbverteilung.
    
    Args:
        images_dir: Verzeichnis mit den Originalbildern
        masks_dir: Verzeichnis mit den Masken
        output_csv_path: Pfad für die Ausgabe-CSV
        image_extension: Dateiendung der Bilder (.png oder .jpg)
    """
    # Alle Bilder im Verzeichnis finden
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
    
    if not image_files:
        print(f"Keine Bilder gefunden in: {images_dir}")
        return
    
    print(f"Verarbeite {len(image_files)} Bilder aus {images_dir}...")
    
    # Farben in gewünschter Reihenfolge
    color_order = ['rot', 'orange', 'gelb', 'gruen', 'cyan', 'blau', 
                   'lila', 'pink', 'braun', 'schwarz', 'weiss', 'grau']
    
    # Gesamtzählung über alle Bilder
    total_counts = {color: 0 for color in color_order}
    
    # Ergebnisse pro Bild
    results = []
    
    for image_file in sorted(image_files):
        # Bildnamen ohne Extension
        base_name = os.path.splitext(image_file)[0]
        
        # Maskendatei finden
        mask_file = f"{base_name}_mask.png"
        
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, mask_file)
        
        if not os.path.exists(mask_path):
            print(f"  Warnung: Keine Maske gefunden für {image_file}")
            continue
        
        # Farbverteilung analysieren
        color_counts = analyze_masked_colors(image_path, mask_path)
        
        if color_counts is None:
            continue
        
        # Gesamtanzahl Pixel in der Maske
        total_pixels = sum(color_counts.values())
        
        if total_pixels == 0:
            print(f"  Warnung: Keine Pixel in Maske für {image_file}")
            continue
        
        # Ergebnis für dieses Bild speichern
        result = {'bildname': image_file, 'gesamt_pixel': total_pixels}
        for color in color_order:
            result[f'{color}_count'] = color_counts[color]
            result[f'{color}_prozent'] = round(color_counts[color] / total_pixels * 100, 2)
        
        results.append(result)
        
        # Zur Gesamtzählung addieren
        for color in color_order:
            total_counts[color] += color_counts[color]
    
    # CSV schreiben
    if not results:
        print("Keine Ergebnisse zum Speichern.")
        return
    
    # Gesamtanzahl aller Pixel
    total_all_pixels = sum(total_counts.values())
    
    # Einfache CSV mit Farben als Zeilen und Anzahl/Prozent als Spalten
    header = ['farbe', 'anzahl_pixel', 'prozent']
    
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        
        # Eine Zeile pro Farbe
        for color in color_order:
            count = total_counts[color]
            percentage = round(count / total_all_pixels * 100, 2) if total_all_pixels > 0 else 0.0
            writer.writerow({
                'farbe': color,
                'anzahl_pixel': count,
                'prozent': percentage
            })
        
        # Gesamtzeile
        writer.writerow({
            'farbe': 'GESAMT',
            'anzahl_pixel': total_all_pixels,
            'prozent': 100.0
        })
    
    print(f"CSV gespeichert unter: {output_csv_path}")
    print(f"\nZusammenfassung:")
    print(f"  Verarbeitete Bilder: {len(results)}")
    print(f"  Gesamte maskierte Pixel: {total_all_pixels:,}")
    print(f"\nFarbverteilung (Gesamt):")
    for color in color_order:
        count = total_counts[color]
        percentage = count / total_all_pixels * 100 if total_all_pixels > 0 else 0
        print(f"  {color:10s}: {count:10,} Pixel ({percentage:6.2f}%)")


def main():
    """
    Hauptfunktion: Verarbeitet beide Datensätze (Butterfly und Car).
    """
    # Basispfad
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Butterfly Dataset
    print("=" * 60)
    print("BUTTERFLY DATASET")
    print("=" * 60)
    butterfly_images = os.path.join(base_path, 'image', 'butterfly', '1images')
    butterfly_masks = os.path.join(base_path, 'image', 'butterfly', '2masken')
    butterfly_output = os.path.join(base_path, 'image', 'butterfly', 'color_distribution.csv')
    
    process_dataset(butterfly_images, butterfly_masks, butterfly_output, image_extension='.png')
    
    print("\n")
    
    # Car Dataset
    print("=" * 60)
    print("CAR DATASET")
    print("=" * 60)
    car_images = os.path.join(base_path, 'image', 'car', '1images')
    car_masks = os.path.join(base_path, 'image', 'car', '2masken')
    car_output = os.path.join(base_path, 'image', 'car', 'color_distribution.csv')
    
    process_dataset(car_images, car_masks, car_output, image_extension='.jpg')
    
    print("\n" + "=" * 60)
    print("Fertig!")
    print("=" * 60)


if __name__ == "__main__":
    main()
