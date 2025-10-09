"""
Network Dissection für MaskDINO Transformer Decoder.

Basiert auf Decoder-spezifischen Daten:
- Iteriert über output/decoder/layer*/Query.csv
- Extrahiert Layer-, Bild- und Query-Index sowie Query-Features (pro Zeile)  
- Lädt passende Pixel-Embeddings aus output/decoder/pixel_embeddings/
- Bereitet die Masken aus myThesis/image/rot/ in entsprechender Größe vor
- Verarbeitet mehrere Bilder aus myThesis/image/1images/

Ergebnis: Generator, der iou_core_decoder mit allen benötigten Inputs versorgt.
"""

from __future__ import annotations

import os
import re
import csv
import json
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2  # Für Bild-/Maskenverarbeitung
except Exception:  # pragma: no cover
    cv2 = None


# -----------------------------
# Hilfsfunktionen Pfade/IO
# -----------------------------

def _project_root() -> str:
    # Dieses File liegt in myThesis/decoder -> eine Ebene hoch ist myThesis
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _decoder_out_dir() -> str:
    return os.path.join(_project_root(), "output", "decoder")


def _mask_dir() -> str:
    """Gibt den Ordner mit den Ground-Truth-Masken zurück."""
    return os.path.join(_project_root(), "image", "rot")


def _find_layer_csvs() -> List[Tuple[int, str]]:
    """Findet alle layer*/Query.csv Dateien und extrahiert Layer-Indices."""
    base = _decoder_out_dir()
    if not os.path.isdir(base):
        return []
    
    layer_csvs: List[Tuple[int, str]] = []
    for name in os.listdir(base):
        if not name.startswith("layer"):
            continue
        m = re.match(r"layer(\d+)$", name)
        if not m:
            continue
        lidx = int(m.group(1))
        csv_path = os.path.join(base, name, "Query.csv")
        if os.path.isfile(csv_path):
            layer_csvs.append((lidx, csv_path))
    
    layer_csvs.sort(key=lambda x: x[0])
    return layer_csvs


def _load_all_pixel_embeddings() -> Dict[str, Dict]:
    """
    Lädt alle verfügbaren metadata und pixel_embeddings.
    
    Returns:
        { embedding_id: {"metadata": metadata_dict, "embedding": np.ndarray} }
    """
    pixel_embed_dir = os.path.join(_decoder_out_dir(), "pixel_embeddings")
    out: Dict[str, Dict] = {}
    
    if not os.path.isdir(pixel_embed_dir):
        return out
    
    # Sammle alle metadata files
    metadata_files = [f for f in os.listdir(pixel_embed_dir) if f.startswith("metadata_") and f.endswith(".json")]
    
    for metadata_file in metadata_files:
        metadata_path = os.path.join(pixel_embed_dir, metadata_file)
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            continue
        
        # Lade entsprechende .npy Datei
        npy_file = metadata.get("npy_file")
        if not npy_file:
            continue
        
        npy_path = os.path.join(pixel_embed_dir, npy_file)
        if not os.path.isfile(npy_path):
            continue
        
        try:
            embedding = np.load(npy_path)
        except Exception:
            continue
        
        # Nutze embedding_id als Schlüssel (sollte in metadata vorhanden sein)
        embedding_id = metadata.get("embedding_id")
        if not embedding_id:
            # Fallback: Generiere aus image_id
            image_id = metadata.get("image_id")
            if image_id:
                embedding_id = f"embed_{image_id}"
            else:
                # Letzter Fallback: verwende Dateinamen
                embedding_id = os.path.splitext(metadata_file)[0].replace("metadata_", "embed_")
        
        out[embedding_id] = {
            "metadata": metadata,
            "embedding": embedding
        }
    
    return out


def _select_pixel_embedding_for(image_id: str, all_embeddings: Dict[str, Dict]) -> Optional[Dict]:
    """
    Wählt passende Pixel-Embedding für die gegebene image_id.
    
    Args:
        image_id: Eindeutige Bild-ID (z.B. "image_1")
        all_embeddings: Dict mit allen verfügbaren Embeddings (key = embedding_id)
    
    Strategie:
    1) Exaktes Match über embedding_id = f"embed_{image_id}"
    2) Match über image_id in metadata
    """
    if not all_embeddings:
        return None
    
    # 1) Exaktes Match über embedding_id
    expected_embedding_id = f"embed_{image_id}"
    if expected_embedding_id in all_embeddings:
        return all_embeddings[expected_embedding_id]
    
    # 2) Suche über image_id in metadata
    for embedding_id, embedding_data in all_embeddings.items():
        metadata_image_id = embedding_data["metadata"].get("image_id")
        if metadata_image_id == image_id:
            return embedding_data
    
    # Kein Match gefunden
    print(f"⚠️  Kein Pixel-Embedding gefunden für image_id='{image_id}'")
    return None


# -----------------------------
# Maskenaufbereitung 
# -----------------------------

def _prepare_mask_binary(mask_bgr: np.ndarray) -> np.ndarray:
    """
    Wandelt farbige Maske in binäre (bool) um. Erwartet BGR oder RGB.
    
    Heuristik für 'rot': R hoch, G/B niedrig. Fallback: alles != schwarz.
    Rückgabe: bool-Array [H, W].
    """
    if mask_bgr.ndim == 2:
        # Bereits Graustufen -> threshold > 0
        return (mask_bgr > 0)
    
    # Falls Bild in RGB statt BGR vorliegt, spielt es für die Heuristik kaum Rolle
    b = mask_bgr[..., 0].astype(np.int16)
    g = mask_bgr[..., 1].astype(np.int16)
    r = mask_bgr[..., 2].astype(np.int16)
    
    red_dominant = (r > 150) & (g < 100) & (b < 100)
    if np.count_nonzero(red_dominant) == 0:
        # Fallback: alle nicht-schwarzen Pixel
        non_black = (r > 0) | (g > 0) | (b > 0)
        return non_black
    return red_dominant


def _get_input_size_from_embedding(embedding_data: Dict) -> Tuple[int, int]:
    """
    Bestimmt Input-Größe basierend auf Pixel-Embedding-Metadaten.
    
    Priorität:
    1) Nutze exakte input_h/input_w aus Metadata (falls vorhanden)
    2) Fallback: berechne über stride (falls vorhanden)
    3) Fallback: Standard 32x Upsampling
    """
    metadata = embedding_data["metadata"]
    
    # 1) Exakte Input-Größe aus Metadata
    if "input_h" in metadata and "input_w" in metadata:
        input_h = int(metadata["input_h"])
        input_w = int(metadata["input_w"])
        return (input_h, input_w)
    
    # 2) Berechnung über stride
    if "stride" in metadata:
        embed_h = int(metadata.get("embed_h", metadata.get("height", 25)))
        embed_w = int(metadata.get("embed_w", metadata.get("width", 25)))
        stride = int(metadata["stride"])
        input_h = embed_h * stride
        input_w = embed_w * stride
        return (input_h, input_w)
    
    # 3) Fallback: Standard-Heuristik 32x Upsampling
    embed_h = int(metadata.get("embed_h", metadata.get("height", 25)))
    embed_w = int(metadata.get("embed_w", metadata.get("width", 25)))
    input_h = embed_h * 32  # Standard: 25 * 32 = 800
    input_w = embed_w * 32  # Standard: 25 * 32 = 800
    
    return (input_h, input_w)


def _validate_input_size(input_size: Tuple[int, int], metadata: Dict) -> Tuple[int, int]:
    """
    Validiert und korrigiert Input-Größe falls nötig.
    
    Args:
        input_size: (H, W) aus _get_input_size_from_embedding
        metadata: Metadata-Dictionary
    
    Returns:
        validated_size: Validierte (H, W) Tupel
    """
    input_h, input_w = input_size
    
    # Mindestgröße prüfen (sollte größer als Embedding sein)
    embed_h = int(metadata.get("embed_h", metadata.get("height", 25)))
    embed_w = int(metadata.get("embed_w", metadata.get("width", 25)))
    
    if input_h < embed_h or input_w < embed_w:
        print(f"Warning: Input-Größe ({input_h}×{input_w}) kleiner als Embedding ({embed_h}×{embed_w}). Korrigiere auf Minimum.")
        input_h = max(input_h, embed_h)
        input_w = max(input_w, embed_w)
    
    # Maximale Größe prüfen (Speicher-Schutz)
    max_size = 2048
    if input_h > max_size or input_w > max_size:
        print(f"Warning: Input-Größe ({input_h}×{input_w}) sehr groß. Begrenze auf {max_size}×{max_size}.")
        input_h = min(input_h, max_size)
        input_w = min(input_w, max_size)
    
    return (input_h, input_w)


def _load_mask_for_image(image_id: str, input_size: Tuple[int, int]) -> np.ndarray:
    """
    Lädt die Maske für das gegebene Bild und liefert sie als bool-Array in input-Größe (H_in, W_in).
    
    Args:
        image_id: Eindeutige Bild-ID (z.B. "image_1")
        input_size: Zielgröße (H, W)
    
    Returns:
        mask_input: bool-Array mit Shape (H_in, W_in)
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) wird benötigt, ist aber nicht verfügbar.")
    
    mask_dir = _mask_dir()
    
    # Konvertiere image_id zurück zu möglichen Dateinamen
    # "image_1" -> versuche "image 1.jpg", "image 1.png", etc.
    # Ersetze _ durch Leerzeichen für die Suche
    search_name = image_id.replace("_", " ")
    
    # Versuche verschiedene Dateinamen-Konventionen
    mask_file = None
    for ext in ['.jpg', '.png', '.jpeg']:
        # 1) Mit Leerzeichen: "image 1.jpg"
        candidate = os.path.join(mask_dir, f"{search_name}{ext}")
        if os.path.isfile(candidate):
            mask_file = candidate
            break
        # 2) Ohne Leerzeichen: "image1.jpg"
        candidate = os.path.join(mask_dir, f"{image_id}{ext}")
        if os.path.isfile(candidate):
            mask_file = candidate
            break
    
    if mask_file is None or not os.path.isfile(mask_file):
        raise FileNotFoundError(
            f"Maske für {image_id} nicht gefunden. "
            f"Gesucht in: {mask_dir} mit Muster: {search_name}.[jpg|png] oder {image_id}.[jpg|png]"
        )
    
    m = cv2.imread(mask_file, cv2.IMREAD_COLOR)
    if m is None:
        raise FileNotFoundError(f"Maske konnte nicht geladen werden: {mask_file}")
    
    # In bool umwandeln und direkt auf input-Größe skalieren (nearest für binär)
    mask_bin = _prepare_mask_binary(m).astype(np.uint8)
    Hin, Win = int(input_size[0]), int(input_size[1])
    mask_input = cv2.resize(mask_bin, (Win, Hin), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask_input


# -----------------------------
# CSV-Iteration und Paketierung
# -----------------------------

# Regex für das neue CSV-Format mit image_id: "image_1, Query1"
_NAME_RE = re.compile(r"^(.+),\s*Query(\d+)$")


def _iter_csv_rows(csv_path: str) -> Iterable[Tuple[str, int, np.ndarray]]:
    """
    Iteriert Zeilen einer Query.csv und liefert (image_id, query_idx, query_features).
    
    query_features ist ein 1D np.ndarray[256] float32.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            name = row[0].strip().strip('"')  # Entferne führende/nachgestellte Anführungszeichen
            m = _NAME_RE.match(name)
            if not m:
                continue
            image_id = m.group(1).strip()  # z.B. "image_1"
            query_idx = int(m.group(2))
            try:
                values = [float(x) for x in row[1:]]
            except ValueError:
                # Überspringe fehlerhafte Zeilen
                continue
            query_features = np.asarray(values, dtype=np.float32)
            yield image_id, query_idx, query_features


def iter_decoder_iou_inputs():
    """
    Haupt-Iterator: liefert pro CSV-Zeile ein DecoderIoUInput-Paket.
    - Erkennt Layer-Index aus Ordnernamen
    - Mappt image_id auf passende Pixel-Embeddings
    - Bereitet Maske für Input-Size vor (gecacht pro input_size)
    """
    # Lazy-Import von iou_core_decoder für Kompatibilität
    try:
        from .iou_core_decoder import DecoderIoUInput  # type: ignore
    except Exception:
        import os as _os, sys as _sys
        _sys.path.append(_os.path.dirname(__file__))
        from iou_core_decoder import DecoderIoUInput  # type: ignore
    
    layer_csvs = _find_layer_csvs()
    all_embeddings = _load_all_pixel_embeddings()
    
    # Cache: Maske je (image_id, input_size)
    mask_cache: Dict[Tuple[str, Tuple[int, int]], np.ndarray] = {}
    
    for lidx, csv_path in layer_csvs:
        for image_id, query_idx, query_features in _iter_csv_rows(csv_path):
            embedding_data = _select_pixel_embedding_for(image_id, all_embeddings)
            if embedding_data is None:
                # Ohne Pixel-Embeddings ist IoU-Berechnung nicht möglich
                print(f"⚠️  Kein Pixel-Embedding für {image_id} gefunden. Überspringe.")
                continue
            
            input_size = _get_input_size_from_embedding(embedding_data)
            input_size = _validate_input_size(input_size, embedding_data["metadata"])
            
            # Maske beschaffen (gecacht nach (image_id, input_size))
            cache_key = (image_id, input_size)
            if cache_key in mask_cache:
                mask_input = mask_cache[cache_key]
            else:
                try:
                    mask_input = _load_mask_for_image(image_id, input_size)
                    mask_cache[cache_key] = mask_input
                except FileNotFoundError as e:
                    print(f"⚠️  {e}. Überspringe {image_id}.")
                    continue
            
            yield DecoderIoUInput(
                layer_idx=lidx,
                image_id=image_id,  # Geändert: image_id statt image_idx
                query_idx=query_idx,
                query_features=query_features,
                pixel_embedding=embedding_data["embedding"],
                input_size=input_size,
                mask_input=mask_input,
            )


# -----------------------------
# Network Dissection: per-Query Thresholding
# -----------------------------

def compute_per_query_thresholds(percentile: float = 90.0) -> Dict[Tuple[int, int], float]:
    """
    Berechnet für jede Query den Threshold über alle Bilder.
    
    Args:
        percentile: Perzentil-Wert für Threshold (Default: 99.5)
    
    Returns:
        Dict[(layer_idx, query_idx)] -> threshold_value
    
    Strategie:
    - Sammle alle Aktivationswerte (Response-Map-Werte) für jede Query über alle Bilder
    - Berechne das X-Perzentil (z.B. 99.5) über alle gesammelten Werte
    - Speichere den Threshold pro (layer_idx, query_idx)
    """
    # Lazy-Import
    try:
        from .iou_core_decoder import _compute_query_response_map  # type: ignore
    except Exception:
        import os as _os, sys as _sys
        _sys.path.append(_os.path.dirname(__file__))
        from iou_core_decoder import _compute_query_response_map  # type: ignore
    
    print(f"🔍 Berechne per-Query Thresholds (Perzentil: {percentile})...")
    
    # Sammle alle Aktivationswerte pro Query: (layer_idx, query_idx) -> List[float]
    activation_values: Dict[Tuple[int, int], List[float]] = {}
    
    for item in iter_decoder_iou_inputs():
        key = (item.layer_idx, item.query_idx)
        
        # Berechne Response-Map für diese Query/Bild-Kombination
        response_map = _compute_query_response_map(
            item.query_features,
            item.pixel_embedding
        )
        
        # Sammle alle Werte der Response-Map
        values = response_map.flatten().tolist()
        
        if key not in activation_values:
            activation_values[key] = []
        activation_values[key].extend(values)
    
    # Berechne Threshold pro Query
    thresholds: Dict[Tuple[int, int], float] = {}
    for key, values in activation_values.items():
        if len(values) > 0:
            threshold = float(np.percentile(values, percentile))
            thresholds[key] = threshold
        else:
            thresholds[key] = 0.0
    
    print(f"✅ {len(thresholds)} Query-Thresholds berechnet")
    return thresholds


def compute_mean_iou_per_query(
    query_thresholds: Dict[Tuple[int, int], float]
) -> Dict[int, List[Dict[str, object]]]:
    """
    Berechnet den durchschnittlichen IoU über alle Bilder für jede Query.
    
    Args:
        query_thresholds: Dict[(layer_idx, query_idx)] -> threshold_value
    
    Returns:
        Dict[layer_idx] -> List[{"query_idx": int, "mean_iou": float, "num_images": int}]
    """
    # Lazy-Imports
    try:
        from .iou_core_decoder import _compute_query_response_map, apply_per_query_binarization, _scale_to_input_size, _compute_iou  # type: ignore
    except Exception:
        import os as _os, sys as _sys
        _sys.path.append(_os.path.dirname(__file__))
        from iou_core_decoder import _compute_query_response_map, apply_per_query_binarization, _scale_to_input_size, _compute_iou  # type: ignore
    
    print(f"📈 Berechne mIoU pro Query...")
    
    # Sammle IoU-Werte pro Query: (layer_idx, query_idx) -> List[float]
    query_ious: Dict[Tuple[int, int], List[float]] = {}
    
    for item in iter_decoder_iou_inputs():
        key = (item.layer_idx, item.query_idx)
        
        # Hole vorberechneten Threshold
        threshold = query_thresholds.get(key)
        if threshold is None:
            continue
        
        # Berechne Response-Map
        response_map = _compute_query_response_map(
            item.query_features,
            item.pixel_embedding
        )
        
        # Skaliere auf Input-Größe
        heatmap_scaled = _scale_to_input_size(response_map, item.input_size)
        
        # Binarisiere mit per-Query Threshold
        binary_map = apply_per_query_binarization(heatmap_scaled, threshold)
        
        # Berechne IoU
        iou = _compute_iou(binary_map, item.mask_input)
        
        # Sammle IoU-Wert
        if key not in query_ious:
            query_ious[key] = []
        query_ious[key].append(iou)
    
    # Berechne mIoU pro Query und gruppiere nach Layer
    per_layer_results: Dict[int, List[Dict[str, object]]] = {}
    
    for (layer_idx, query_idx), ious in query_ious.items():
        if len(ious) > 0:
            mean_iou = float(np.mean(ious))
        else:
            mean_iou = 0.0
        
        result = {
            "query_idx": query_idx,
            "mean_iou": mean_iou,
            "num_images": len(ious),
        }
        
        if layer_idx not in per_layer_results:
            per_layer_results[layer_idx] = []
        per_layer_results[layer_idx].append(result)
    
    # Sortiere pro Layer nach mIoU (absteigend)
    for layer_idx in per_layer_results:
        per_layer_results[layer_idx].sort(key=lambda x: x["mean_iou"], reverse=True)
    
    print(f"✅ mIoU berechnet für {len(query_ious)} Queries")
    return per_layer_results


def export_mean_iou_csv(
    mean_iou_results: Dict[int, List[Dict[str, object]]],
    percentile: float = 90.0
) -> None:
    """Exportiert mIoU-Ergebnisse als CSV pro Layer (wird von main_network_dissection_per_query genutzt)."""
    export_root = _export_root()
    print(f"📊 Exportiere mIoU-Ergebnisse...")
    for layer_idx, results in mean_iou_results.items():
        layer_dir = os.path.join(export_root, f"layer{layer_idx}")
        _ensure_dir(layer_dir)
        csv_path = os.path.join(layer_dir, "mIoU_per_Query.csv")
        fieldnames = ["query_idx", "mean_iou", "num_images"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  ✅ Layer {layer_idx}: {len(results)} Queries -> {csv_path}")
    print(f"✅ mIoU Export abgeschlossen")


def create_best_query_visualizations(
    mean_iou_results: Dict[int, List[Dict[str, object]]],
    query_thresholds: Dict[Tuple[int, int], float]
) -> None:
    """
    Erstellt für jede Layer eine Visualisierung der besten Query.
    
    Farbcodierung:
    - Rot (BGR=0,0,255): Ground Truth Maske (nur Maske)
    - Blau (BGR=255,0,0): Überschneidung (Maske ∧ Binär-Heatmap)
    - Gelb (BGR=0,255,255): Binär-Heatmap nur (ohne Maske)
    - Schwarz: Rest
    
    Args:
        mean_iou_results: Dict[layer_idx] -> List[{"query_idx", "mean_iou", ...}]
        query_thresholds: Dict[(layer_idx, query_idx)] -> threshold_value
    """
    if cv2 is None:
        print("⚠️  OpenCV nicht verfügbar. Überspringe Visualisierungen.")
        return
    
    # Lazy-Imports
    try:
        from .iou_core_decoder import _compute_query_response_map, apply_per_query_binarization, _scale_to_input_size  # type: ignore
    except Exception:
        import os as _os, sys as _sys
        _sys.path.append(_os.path.dirname(__file__))
        from iou_core_decoder import _compute_query_response_map, apply_per_query_binarization, _scale_to_input_size  # type: ignore
    
    print(f"🎨 Erstelle Visualisierungen für beste Queries pro Layer...")
    
    export_root = _export_root()
    
    # Für jede Layer die beste Query finden
    for layer_idx, results in mean_iou_results.items():
        if not results:
            continue
        
        # Beste Query (bereits nach mIoU sortiert, erste ist beste)
        best_query = results[0]
        best_query_idx = int(best_query["query_idx"])
        best_miou = float(best_query["mean_iou"])
        
        print(f"\n  Layer {layer_idx}: Beste Query = {best_query_idx} (mIoU = {best_miou:.4f})")
        
        # Hole Threshold für diese Query
        key = (layer_idx, best_query_idx)
        threshold = query_thresholds.get(key)
        if threshold is None:
            print(f"    ⚠️  Kein Threshold gefunden. Überspringe.")
            continue
        
        # Erzeuge Visualisierungen für ALLE Bilder dieser besten Query
        saved_count = 0
        for item in iter_decoder_iou_inputs():
            if item.layer_idx == layer_idx and item.query_idx == best_query_idx:
                # Berechne Response-Map
                response_map = _compute_query_response_map(
                    item.query_features,
                    item.pixel_embedding
                )
                
                # Skaliere auf Input-Größe
                heatmap_scaled = _scale_to_input_size(response_map, item.input_size)
                
                # Binarisiere
                binary_map = apply_per_query_binarization(heatmap_scaled, threshold)
                
                # Erstelle Visualisierung
                mask = item.mask_input.astype(bool)
                bin_hm = binary_map.astype(bool)
                
                # Berechne Bereiche
                inter = np.logical_and(mask, bin_hm)  # Überschneidung
                mask_only = np.logical_and(mask, np.logical_not(bin_hm))  # Nur Maske
                hm_only = np.logical_and(bin_hm, np.logical_not(mask))  # Nur Heatmap
                
                # Erstelle BGR-Bild
                H, W = mask.shape
                img = np.zeros((H, W, 3), dtype=np.uint8)
                
                # Blau für Überschneidung
                img[inter, 0] = 255  # B
                # Rot für Maske-only
                img[mask_only, 2] = 255  # R
                # Gelb für Heatmap-only (R+G)
                img[hm_only, 1] = 255  # G
                img[hm_only, 2] = 255  # R
                
                # Speichere Visualisierung
                layer_dir = os.path.join(export_root, f"layer{layer_idx}")
                _ensure_dir(layer_dir)
                vis_dir = os.path.join(layer_dir, "visualizations")
                _ensure_dir(vis_dir)
                
                vis_filename = f"best_query_{best_query_idx}_{item.image_id}.png"
                vis_path = os.path.join(vis_dir, vis_filename)
                
                cv2.imwrite(vis_path, img)
                saved_count += 1
                if saved_count % 25 == 0:
                    print(f"    … {saved_count} Visualisierungen gespeichert …")

        if saved_count == 0:
            print(f"    ⚠️  Keine Daten für Visualisierung gefunden.")
        else:
            print(f"    ✅ {saved_count} Visualisierungen gespeichert.")
    
    print(f"\n✅ Visualisierungen abgeschlossen")


# -----------------------------
# Export-Funktionen
# -----------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _export_root() -> str:
    # Sammelordner für IoU-Ergebnisse
    return os.path.join(_decoder_out_dir(), "iou_results")


# Hilfsfunktion _write_csv wird nicht mehr benötigt (zugehörige Exportfunktion entfernt)


# main_export_decoder_iou wurde entfernt (wird nicht über main aufgerufen)


# main_print_all wurde entfernt (wird nicht über main aufgerufen)


def main_network_dissection_per_query(percentile: float = 99.5) -> None:
    """
    Hauptfunktion für Network Dissection mit per-Query Thresholding.
    
    Args:
        percentile: Perzentil-Wert für Query-Threshold (Default: 99.5)
    
    Workflow:
    1. Berechne per-Query Thresholds über alle Bilder
    2. Berechne mIoU pro Query über alle Bilder
    3. Exportiere mIoU_per_Query.csv (sortiert nach mIoU)
    4. Erstelle Visualisierungen der besten Queries pro Layer
    
    Ausgabe:
    - myThesis/output/decoder/iou_results/layer<X>/mIoU_per_Query.csv
    - myThesis/output/decoder/iou_results/layer<X>/visualizations/best_query_*.png
    """
    print("=" * 80)
    print(f"🚀 Network Dissection mit per-Query Thresholding (Perzentil: {percentile})")
    print("=" * 80)
    
    # Schritt 1: Berechne per-Query Thresholds
    print("\n📌 Schritt 1/4: Threshold-Berechnung")
    query_thresholds = compute_per_query_thresholds(percentile=percentile)
    
    if not query_thresholds:
        print("❌ Keine Thresholds berechnet. Abbruch.")
        return
    
    # Schritt 2: Berechne mIoU pro Query
    print("\n📌 Schritt 2/4: mIoU-Berechnung")
    mean_iou_results = compute_mean_iou_per_query(query_thresholds)
    
    # Schritt 3: Exportiere mIoU_per_Query.csv (sortiert nach mIoU)
    print("\n📌 Schritt 3/4: mIoU-Export")
    export_mean_iou_csv(mean_iou_results, percentile=percentile)
    
    # Schritt 4: Erstelle Visualisierungen
    print("\n📌 Schritt 4/4: Visualisierungen erstellen")
    create_best_query_visualizations(mean_iou_results, query_thresholds)
    
    # Zusammenfassung
    print("\n" + "=" * 80)
    print("✅ Network Dissection abgeschlossen!")
    print("=" * 80)
    
    # Statistiken ausgeben
    total_queries = len(query_thresholds)
    total_layers = len(mean_iou_results)
    
    print(f"\n📊 Statistiken:")
    print(f"  • Verarbeitete Queries: {total_queries}")
    print(f"  • Verarbeitete Layer: {total_layers}")
    print(f"  • Perzentil: {percentile}")
    
    # Beste Queries pro Layer
    print(f"\n🏆 Top-Queries pro Layer (nach mIoU):")
    for layer_idx in sorted(mean_iou_results.keys()):
        results = mean_iou_results[layer_idx]
        if results:
            # Sortiere nach mIoU absteigend
            top_queries = sorted(results, key=lambda x: x["mean_iou"], reverse=True)[:3]
            print(f"\n  Layer {layer_idx}:")
            for i, q in enumerate(top_queries, 1):
                print(f"    {i}. Query {q['query_idx']}: mIoU = {q['mean_iou']:.4f} ({q['num_images']} Bilder)")
    
    export_root = _export_root()
    print(f"\n📁 Ausgabe-Verzeichnis: {export_root}")
    print("=" * 80)


if __name__ == "__main__":
    # Network Dissection mit per-Query Thresholding
    # Konfigurierbar: Perzentil (Default: 99.5)
    PERCENTILE = 90.0  # Konfigurierbare Variable
    main_network_dissection_per_query(percentile=PERCENTILE)


# -----------------------------
# Schema-Dokumentation für erweiterte Metadata
# -----------------------------
"""
Erweitertes Metadata-Schema für exakte Skalierung:

{
    "image_index": 0,                                    # Original: 0-basiert
    "layer_name": "sem_seg_head.pixel_decoder.input_proj.0.0",  # Original
    "shape": [256, 25, 25],                             # Original: [C, H, W]
    "channels": 256,                                     # Original
    "height": 25,                                        # Original: Embedding H
    "width": 25,                                         # Original: Embedding W
    "npy_file": "pixel_embed_Bild0000.npy",            # Original
    
    # Neue Felder für exakte Skalierung:
    "embed_h": 25,                                       # Explizit: Embedding-Höhe  
    "embed_w": 25,                                       # Explizit: Embedding-Breite
    "input_h": 800,                                      # Exakt: Input-Bildgröße H
    "input_w": 800,                                      # Exakt: Input-Bildgröße W
    "stride": 32,                                        # Optional: Skalierungsfaktor
}

Prioritätsreihenfolge beim Laden:
1. input_h/input_w (falls vorhanden) -> direkte Verwendung
2. stride (falls vorhanden) -> embed_h/w * stride
3. Fallback -> embed_h/w * 32 (Standard-Heuristik)
"""
