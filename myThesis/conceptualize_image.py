import cv2
import numpy as np
import os
import shutil
from PIL import Image

def get_color_ranges():
    """
    Definiert HSV-Farbbereiche für verschiedene Farben.
    
    Returns:
        dict: Dictionary mit Farbnamen als Schlüssel und HSV-Bereichen als Werten
    """
    color_ranges = {
        'rot': [
            ([0, 50, 50], [10, 255, 255]),      # Unterer roter Bereich
            ([170, 50, 50], [180, 255, 255])   # Oberer roter Bereich
        ],
        'gruen': [
            ([40, 50, 50], [80, 255, 255])     # Grüner Bereich
        ],
        'blau': [
            ([100, 50, 50], [130, 255, 255])   # Blauer Bereich
        ],
        'gelb': [
            ([20, 50, 50], [35, 255, 255])     # Gelber Bereich
        ],
        'orange': [
            ([10, 50, 50], [20, 255, 255])     # Oranger Bereich
        ],
        'lila': [
            ([130, 50, 50], [170, 255, 255])   # Lila/Violett Bereich
        ],
        'cyan': [
            ([80, 50, 50], [100, 255, 255])    # Cyan/Türkis Bereich
        ],
        'pink': [
            ([140, 50, 50], [180, 255, 255]),  # Pink Bereich (ähnlich wie Magenta)
            ([0, 50, 50], [10, 255, 200])      # Rosa Töne
        ],
        'braun': [
            ([8, 50, 20], [20, 255, 200])      # Brauner Bereich (niedriger Wert)
        ],
        'schwarz': [],  # Wird separat behandelt
        'weiss': [],    # Wird separat behandelt
        'grau': []      # Wird separat behandelt
    }
    return color_ranges

def extract_color_mask(image_path, color_name, output_path):
    """
    Extrahiert eine Farbmaske aus einem Bild für Network Dissection.
    
    Args:
        image_path (str): Pfad zum Eingangsbild
        color_name (str): Name der zu extrahierenden Farbe
        output_path (str): Pfad zum Speichern der Farbmaske
    """
    # Bild laden
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Konnte Bild nicht laden: {image_path}")
    
    # Von BGR zu HSV konvertieren für bessere Farbeextraktion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    color_ranges = get_color_ranges()
    color_name = color_name.lower()
    
    if color_name == 'schwarz':
        # Schwarz basierend auf niedrigem Wert (V-Kanal)
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    elif color_name == 'weiss':
        # Weiß basierend auf hohem Wert und niedriger Sättigung
        mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    elif color_name == 'grau':
        # Grau basierend auf mittlerem Wert und niedriger Sättigung
        mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 30, 200]))
    else:
        # Für alle anderen Farben HSV-Bereiche verwenden
        if color_name not in color_ranges:
            raise ValueError(f"Unbekannte Farbe: {color_name}")
        
        ranges = color_ranges[color_name]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Alle Bereiche für diese Farbe kombinieren
        for lower, upper in ranges:
            temp_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, temp_mask)
    
    # Morphologische Operationen zur Bereinigung der Maske
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Maske speichern
    cv2.imwrite(output_path, mask)
    print(f"{color_name.capitalize()} Maske gespeichert unter: {output_path}")
    
    # Statistiken anzeigen
    total_pixels = mask.shape[0] * mask.shape[1]
    color_pixels = np.sum(mask > 0)
    percentage = (color_pixels / total_pixels) * 100
    print(f"{color_name.capitalize()} Pixel: {color_pixels} von {total_pixels} ({percentage:.2f}%)")
    
    return mask

def extract_all_color_masks_exclusive(image_path, output_dir):
    """
    Extrahiert alle Farbmasken aus einem Bild mit exklusiver Zuordnung.
    Jeder Pixel gehört zu genau einem Farbkonzept basierend auf Prioritätsreihenfolge.
    Nicht zugeordnete Pixel werden der nächstgelegenen Farbe zugewiesen.
    
    Args:
        image_path (str): Pfad zum Eingangsbild
        output_dir (str): Ordner zum Speichern der Masken
    """
    # Prioritätsreihenfolge (höhere Position = höhere Priorität bei Überschneidungen)
    priority_colors = [
        'schwarz',    # Höchste Priorität
        'weiss', 
        'grau',
        'rot',
        'gruen', 
        'blau',
        'gelb',
        'orange',
        'braun',
        'pink',
        'lila',
        'cyan'        # Niedrigste Priorität
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Schritt 1: Alle ursprünglichen Masken erstellen (in temporärem Speicher)
    temp_masks = {}
    print("Erstelle temporäre Farbmasken...")
    for color in priority_colors:
        try:
            temp_mask = extract_single_color_mask(image_path, color)
            temp_masks[color] = temp_mask
            color_pixels = np.sum(temp_mask > 0)
            total_pixels = temp_mask.shape[0] * temp_mask.shape[1]
            percentage = (color_pixels / total_pixels) * 100
            print(f"  {color.capitalize()} (vor Exklusivität): {color_pixels} Pixel ({percentage:.2f}%)")
        except Exception as e:
            print(f"  Fehler bei {color}: {e}")
    
    # Schritt 2: Exklusive Zuordnung durch Priorität
    print("\nErstelle exklusive Farbmasken...")
    combined_mask = None  # Alle bereits zugewiesenen Pixel
    final_masks = {}
    
    for color in priority_colors:
        if color not in temp_masks:
            continue
            
        current_mask = temp_masks[color].copy()
        
        # Entferne bereits zugewiesene Pixel (höhere Priorität)
        if combined_mask is not None:
            current_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(combined_mask))
        
        # Aktualisiere die Gesamtmaske
        if combined_mask is None:
            combined_mask = current_mask.copy()
        else:
            combined_mask = cv2.bitwise_or(combined_mask, current_mask)
        
        final_masks[color] = current_mask
        
        # Statistiken für finale Maske
        color_pixels = np.sum(current_mask > 0)
        total_pixels = current_mask.shape[0] * current_mask.shape[1]
        percentage = (color_pixels / total_pixels) * 100
        print(f"  {color.capitalize()} (exklusiv): {color_pixels} Pixel ({percentage:.2f}%)")
    
    # Schritt 3: Nicht zugeordnete Pixel der nächstgelegenen Farbe zuweisen
    print("\nWeise nicht zugeordnete Pixel zu...")
    unassigned_mask = cv2.bitwise_not(combined_mask) if combined_mask is not None else None
    
    if unassigned_mask is not None:
        unassigned_pixels = np.sum(unassigned_mask > 0)
        total_pixels = unassigned_mask.shape[0] * unassigned_mask.shape[1]
        print(f"Nicht zugeordnete Pixel: {unassigned_pixels} ({(unassigned_pixels/total_pixels)*100:.2f}%)")
        
        if unassigned_pixels > 0:
            assign_unassigned_pixels(image_path, unassigned_mask, temp_masks, final_masks)
    
    # Schritt 4: Finale Masken speichern
    print("\nSpeichere finale Masken...")
    final_paths = {}
    for color in priority_colors:
        if color in final_masks:
            output_path = os.path.join(output_dir, f"{color}.png")
            cv2.imwrite(output_path, final_masks[color])
            final_paths[color] = output_path
            
            # Finale Statistiken
            color_pixels = np.sum(final_masks[color] > 0)
            total_pixels = final_masks[color].shape[0] * final_masks[color].shape[1]
            percentage = (color_pixels / total_pixels) * 100
            print(f"  {color.capitalize()} (final): {color_pixels} Pixel ({percentage:.2f}%)")
    
    # Verifikation der vollständigen Zuordnung
    verify_complete_assignment(final_paths, output_dir)
    
    return final_paths

def assign_unassigned_pixels(image_path, unassigned_mask, temp_masks, final_masks):
    """
    Weist nicht zugeordnete Pixel der nächstgelegenen Farbe zu.
    
    Args:
        image_path (str): Pfad zum Bild
        unassigned_mask (np.ndarray): Maske der nicht zugeordneten Pixel
        temp_masks (dict): Temporäre Masken aller Farben
        final_masks (dict): Finale Masken (werden modifiziert)
    """
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Finde alle nicht zugeordneten Pixel-Positionen
    unassigned_coords = np.where(unassigned_mask > 0)
    
    print(f"Verarbeite {len(unassigned_coords[0])} nicht zugeordnete Pixel...")
    
    # Für jeden nicht zugeordneten Pixel
    for i in range(len(unassigned_coords[0])):
        y, x = unassigned_coords[0][i], unassigned_coords[1][i]
        pixel_hsv = hsv[y, x]
        
        # Finde die beste Farbe basierend auf HSV-Distanz
        best_color = find_best_color_for_pixel(pixel_hsv)
        
        # Füge Pixel zur entsprechenden finalen Maske hinzu
        if best_color in final_masks:
            final_masks[best_color][y, x] = 255
    
    print("Alle Pixel erfolgreich zugeordnet!")

def find_best_color_for_pixel(pixel_hsv):
    """
    Findet die beste Farbe für einen gegebenen HSV-Pixel basierend auf Farbdistanz.
    
    Args:
        pixel_hsv (np.ndarray): HSV-Werte des Pixels
        
    Returns:
        str: Name der besten Farbe
    """
    h, s, v = pixel_hsv[0], pixel_hsv[1], pixel_hsv[2]
    
    # Spezialbehandlung für achromatische Farben (niedrige Sättigung)
    if s < 30:
        if v < 50:
            return 'schwarz'
        elif v > 200:
            return 'weiss'
        else:
            return 'grau'
    
    # Für chromatische Farben basierend auf Hue
    color_hue_ranges = {
        'rot': [(0, 10), (170, 180)],
        'orange': [(10, 20)],
        'gelb': [(20, 35)],
        'gruen': [(40, 80)],
        'cyan': [(80, 100)],
        'blau': [(100, 130)],
        'lila': [(130, 170)],
        'pink': [(140, 180)],  # Überschneidung mit Lila
        'braun': [(8, 20)]     # Niedrige Sättigung/Helligkeit
    }
    
    best_color = 'grau'  # Fallback
    min_distance = float('inf')
    
    for color, hue_ranges in color_hue_ranges.items():
        for hue_min, hue_max in hue_ranges:
            # Berechne zirkuläre Distanz im Hue-Raum
            if hue_min <= h <= hue_max:
                return color  # Direkter Treffer
            
            # Minimale Distanz zu diesem Bereich
            dist1 = min(abs(h - hue_min), abs(h - hue_max))
            dist2 = 180 - dist1  # Zirkuläre Distanz
            distance = min(dist1, dist2)
            
            if distance < min_distance:
                min_distance = distance
                best_color = color
    
    return best_color

def extract_single_color_mask(image_path, color_name):
    """
    Extrahiert eine einzelne Farbmaske ohne sie zu speichern.
    
    Args:
        image_path (str): Pfad zum Eingangsbild
        color_name (str): Name der zu extrahierenden Farbe
        
    Returns:
        np.ndarray: Die Farbmaske
    """
    # Bild laden
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Konnte Bild nicht laden: {image_path}")
    
    # Von BGR zu HSV konvertieren für bessere Farbeextraktion
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    color_ranges = get_color_ranges()
    color_name = color_name.lower()
    
    if color_name == 'schwarz':
        # Schwarz basierend auf niedrigem Wert (V-Kanal)
        mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    elif color_name == 'weiss':
        # Weiß basierend auf hohem Wert und niedriger Sättigung
        mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    elif color_name == 'grau':
        # Grau basierend auf mittlerem Wert und niedriger Sättigung
        mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 30, 200]))
    else:
        # Für alle anderen Farben HSV-Bereiche verwenden
        if color_name not in color_ranges:
            raise ValueError(f"Unbekannte Farbe: {color_name}")
        
        ranges = color_ranges[color_name]
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        # Alle Bereiche für diese Farbe kombinieren
        for lower, upper in ranges:
            temp_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask = cv2.bitwise_or(mask, temp_mask)
    
    # Morphologische Operationen zur Bereinigung der Maske
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

def verify_complete_assignment(masks_dict, output_dir):
    """
    Verifiziert, dass alle Pixel einem Farbkonzept zugeordnet sind.
    
    Args:
        masks_dict (dict): Dictionary mit Farbnamen und Maskenpfaden
        output_dir (str): Ausgabeordner für Statistiken
    """
    print(f"\n=== Vollständige Zuordnungs-Verifikation ===")
    
    # Lade alle Masken
    masks = {}
    total_pixels = 0
    assigned_pixels = 0
    
    for color, mask_path in masks_dict.items():
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks[color] = mask
            
            if total_pixels == 0:
                total_pixels = mask.shape[0] * mask.shape[1]
            
            color_pixels = np.sum(mask > 0)
            assigned_pixels += color_pixels
    
    print(f"Gesamtpixel: {total_pixels}")
    print(f"Zugewiesene Pixel: {assigned_pixels}")
    print(f"Nicht zugewiesene Pixel: {total_pixels - assigned_pixels}")
    print(f"Zuweisungsrate: {(assigned_pixels / total_pixels) * 100:.2f}%")
    
    # Prüfe auf Überschneidungen
    overlaps_found = False
    for color1 in masks:
        for color2 in masks:
            if color1 >= color2:  # Vermeide doppelte Prüfungen
                continue
                
            intersection = cv2.bitwise_and(masks[color1], masks[color2])
            overlap_pixels = np.sum(intersection > 0)
            
            if overlap_pixels > 0:
                print(f"WARNUNG: Überschneidung zwischen {color1} und {color2}: {overlap_pixels} Pixel")
                overlaps_found = True
    
    if not overlaps_found:
        print("✓ Alle Masken sind exklusiv - keine Überschneidungen gefunden!")
    
    # Prüfe vollständige Abdeckung
    if assigned_pixels == total_pixels:
        print("✓ Vollständige Pixelabdeckung erreicht - jeder Pixel ist zugeordnet!")
    else:
        print(f"⚠️  Unvollständige Abdeckung: {total_pixels - assigned_pixels} Pixel fehlen")
        
    # Erstelle eine Gesamtübersicht
    create_assignment_overview(masks, os.path.join(output_dir, "pixel_assignment_overview.png"))

def create_assignment_overview(masks, output_path):
    """
    Erstellt eine Visualisierung der Pixelzuweisungen.
    """
    if not masks:
        return
        
    # Erstelle eine farbkodierte Übersicht
    first_mask = list(masks.values())[0]
    h, w = first_mask.shape
    
    # Farbkodierung für jede Farbe
    color_codes = {
        'schwarz': [0, 0, 0],
        'weiss': [255, 255, 255],
        'grau': [128, 128, 128],
        'rot': [0, 0, 255],      # BGR Format
        'gruen': [0, 255, 0],
        'blau': [255, 0, 0],
        'gelb': [0, 255, 255],
        'orange': [0, 165, 255],
        'braun': [42, 42, 165],
        'pink': [203, 192, 255],
        'lila': [128, 0, 128],
        'cyan': [255, 255, 0]
    }
    
    assignment_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for color, mask in masks.items():
        if color in color_codes:
            color_bgr = color_codes[color]
            assignment_image[mask > 0] = color_bgr
    
    cv2.imwrite(output_path, assignment_image)
    print(f"Pixelzuweisungs-Übersicht gespeichert unter: {output_path}")

def create_color_visualization_grid(original_image_path, masks_dict, output_path):
    """
    Erstellt eine Gitter-Visualisierung aller extrahierten Farbmasken.
    
    Args:
        original_image_path (str): Pfad zum ursprünglichen Bild
        masks_dict (dict): Dictionary mit Farbnamen und Maskenpfaden
        output_path (str): Pfad zum Speichern der Visualisierung
    """
    # Original Bild laden
    original = cv2.imread(original_image_path)
    h, w = original.shape[:2]
    
    # 4x3 Grid für 12 Farben + Original
    grid_h, grid_w = 4, 3
    total_h = h * grid_h
    total_w = w * grid_w
    
    visualization = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    
    # Original in die Mitte setzen
    visualization[:h, :w] = original
    
    # Text hinzufügen für Original
    cv2.putText(visualization, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    colors = list(masks_dict.keys())
    pos_idx = 1  # Start nach dem Original
    
    for color in colors:
        if pos_idx >= grid_h * grid_w:
            break
            
        row = pos_idx // grid_w
        col = pos_idx % grid_w
        
        y_start = row * h
        y_end = (row + 1) * h
        x_start = col * w
        x_end = (col + 1) * w
        
        # Maske laden und auf Original anwenden
        mask_path = masks_dict[color]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is not None:
            # Maskiertes Bild erstellen
            masked_image = original.copy()
            masked_image[mask == 0] = [0, 0, 0]
            
            visualization[y_start:y_end, x_start:x_end] = masked_image
            
            # Text hinzufügen
            cv2.putText(visualization, color.capitalize(), 
                       (x_start + 10, y_start + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        pos_idx += 1
    
    cv2.imwrite(output_path, visualization)
    print(f"Farbgitter-Visualisierung gespeichert unter: {output_path}")

def process_dataset(dataset_name, base_path):
    """
    Verarbeitet alle Bilder eines Datasets und speichert die Farbmasken
    in die entsprechenden Farbordner.
    
    Args:
        dataset_name (str): Name des Datasets (z.B. 'car', 'butterfly')
        base_path (str): Basispfad zum Dataset-Ordner
    """
    # Pfade definieren
    input_dir = os.path.join(base_path, "1images")
    base_output_dir = base_path
    
    # Nur diese Farben werden als Masken ausgegeben
    output_colors = ['grau', 'orange', 'schwarz', 'blau']
    
    # Erstelle nur die benötigten Ausgabeordner
    for color_folder in output_colors:
        os.makedirs(os.path.join(base_output_dir, color_folder), exist_ok=True)
    
    # Finde alle Bilder im Input-Ordner
    if not os.path.exists(input_dir):
        print(f"FEHLER: Ordner nicht gefunden: {input_dir}")
        return 0
    
    image_files = sorted([f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not image_files:
        print(f"Keine Bilder gefunden in: {input_dir}")
        return 0
    
    print(f"\n{'#'*70}")
    print(f"### Dataset: {dataset_name.upper()}")
    print(f"{'#'*70}")
    print(f"Gefundene Bilder: {len(image_files)}")
    print(f"Input-Ordner: {input_dir}")
    print(f"Output-Basisordner: {base_output_dir}")
    
    # Verarbeite jedes Bild
    total_processed = 0
    for idx, image_file in enumerate(image_files, 1):
        input_image_path = os.path.join(input_dir, image_file)
        
        print(f"\n{'='*70}")
        print(f"[{dataset_name}] Verarbeite Bild {idx}/{len(image_files)}: {image_file}")
        print(f"{'='*70}")
        
        # Temporärer Ordner für die Masken dieses Bildes
        temp_output_dir = os.path.join(base_output_dir, "temp_masks")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        try:
            # Alle Farbmasken extrahieren (exklusiv)
            extracted_masks = extract_all_color_masks_exclusive(input_image_path, temp_output_dir)
            
            # Verschiebe nur ausgewählte Masken in den entsprechenden Farbordner
            # Mapping von internen Farbnamen zu Ordnernamen (nur für gewünschte Farben)
            color_mapping = {
                'orange': 'orange',
                'schwarz': 'schwarz',
                'grau': 'grau',
                'blau': 'blau'
            }
            
            for color, temp_mask_path in extracted_masks.items():
                if color in color_mapping:
                    target_folder = color_mapping[color]
                    target_path = os.path.join(base_output_dir, target_folder, image_file)
                    
                    # Kopiere Maske in Zielordner
                    if os.path.exists(temp_mask_path):
                        shutil.copy2(temp_mask_path, target_path)
                        print(f"  ✓ {color.capitalize()} -> {target_folder}/{image_file}")
            
            total_processed += 1
            print(f"\n✓ Bild erfolgreich verarbeitet: {image_file}")
            
        except Exception as e:
            print(f"\n✗ FEHLER beim Verarbeiten von {image_file}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Lösche temporäre Masken
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
    
    print(f"\n[{dataset_name}] Erfolgreich verarbeitet: {total_processed}/{len(image_files)} Bilder")
    return total_processed

def process_all_images():
    """
    Verarbeitet alle Bilder aus allen Datasets (butterfly und car) und speichert 
    die Farbmasken in die entsprechenden Farbordner.
    """
    # Basispfad für alle Datasets
    thesis_image_dir = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image"
    
    # Liste aller zu verarbeitenden Datasets
    datasets = [
        ("butterfly", os.path.join(thesis_image_dir, "butterfly")),
        ("car", os.path.join(thesis_image_dir, "car"))
    ]
    
    print("="*70)
    print("=== Farbmasken-Extraktion für Network Dissection ===")
    print("="*70)
    print(f"Zu verarbeitende Datasets: {[d[0] for d in datasets]}")
    
    # Statistiken
    total_results = {}
    
    # Verarbeite jedes Dataset
    for dataset_name, dataset_path in datasets:
        if os.path.exists(dataset_path):
            processed = process_dataset(dataset_name, dataset_path)
            total_results[dataset_name] = processed
        else:
            print(f"\nWARNUNG: Dataset-Ordner nicht gefunden: {dataset_path}")
            total_results[dataset_name] = 0
    
    # Gesamtzusammenfassung
    print(f"\n{'#'*70}")
    print(f"### GESAMTZUSAMMENFASSUNG ###")
    print(f"{'#'*70}")
    
    total_images = 0
    for dataset_name, count in total_results.items():
        print(f"  {dataset_name}: {count} Bilder verarbeitet")
        total_images += count
    
    print(f"\nGesamt: {total_images} Bilder verarbeitet")
    print(f"Farbmasken gespeichert in: {thesis_image_dir}/[dataset]/[farbname]/")
    print("\nFarbextraktion für alle Datasets abgeschlossen!")

if __name__ == "__main__":
    process_all_images()

