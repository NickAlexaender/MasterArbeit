"""
Network Dissection für MaskDINO Transformer Decoder.

Basiert auf Decoder-spezifischen Daten:
- Iteriert über output/decoder/layer*/Query.csv
- Extrahiert Layer-, Bild- und Query-Index sowie Query-Features (pro Zeile)  
- Lädt passende Pixel-Embeddings aus output/decoder/pixel_embeddings/
- Bereitet die Maske aus myThesis/image/colours/rot.png in entsprechender Größe vor

Ergebnis: Generator, der iou_core_decoder mit allen benötigten Inputs versorgt.
"""

from __future__ import annotations

import os
import re
import csv
import json
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple

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


def _mask_path() -> str:
    return os.path.join(_project_root(), "image", "colours", "rot.png")


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
        { image_id: {"metadata": metadata_dict, "embedding": np.ndarray} }
    """
    pixel_embed_dir = os.path.join(_decoder_out_dir(), "pixel_embeddings")
    out: Dict[str, Dict] = {}
    
    if not os.path.isdir(pixel_embed_dir):
        return out
    
    # Sammle alle metadata files
    metadata_files = [f for f in os.listdir(pixel_embed_dir) if f.startswith("metadata_") and f.endswith(".json")]
    
    for metadata_file in metadata_files:
        # Extrahiere image_id aus metadata_Bild0000.json
        m = re.match(r"metadata_(Bild\d+)\.json$", metadata_file)
        if not m:
            continue
        image_id = m.group(1)
        
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
        
        out[image_id] = {
            "metadata": metadata,
            "embedding": embedding
        }
    
    return out


def _select_pixel_embedding_for(image_idx: int, all_embeddings: Dict[str, Dict]) -> Optional[Dict]:
    """
    Wählt passende Pixel-Embedding für den gegebenen Bild-Index.
    
    Strategie:
    1) Wenn es exakt eine Embedding gibt -> nimm diese
    2) Versuche Match über Bild-Index im metadata
    3) Fallback: alphabetisch sortieren und 1-basiert zuordnen
    """
    if not all_embeddings:
        return None
    
    if len(all_embeddings) == 1:
        return next(iter(all_embeddings.values()))
    
    # 2) Match über image_index in metadata
    matches = [v for v in all_embeddings.values() 
              if int(v["metadata"].get("image_index", -1)) == int(image_idx - 1)]  # CSV ist 1-basiert, metadata 0-basiert
    
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Nicht eindeutig – wähle deterministisch erstes nach image_id
        items = sorted(((k, v) for k, v in all_embeddings.items() if v in matches), 
                      key=lambda kv: kv[0])
        return items[0][1]
    
    # 3) Fallback: 1-basierte Reihenfolge über alphabetische Sortierung
    items_sorted = sorted(all_embeddings.items(), key=lambda kv: kv[0])
    idx0 = max(0, min(len(items_sorted) - 1, image_idx - 1))
    return items_sorted[idx0][1]


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


def _load_mask_for_input(input_size: Tuple[int, int]) -> np.ndarray:
    """Lädt die Maske und liefert sie als bool-Array in input-Größe (H_in, W_in)."""
    mask_file = _mask_path()
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) wird benötigt, ist aber nicht verfügbar.")
    
    m = cv2.imread(mask_file, cv2.IMREAD_COLOR)
    if m is None:
        raise FileNotFoundError(f"Maske nicht gefunden: {mask_file}")
    
    # In bool umwandeln und direkt auf input-Größe skalieren (nearest für binär)
    mask_bin = _prepare_mask_binary(m).astype(np.uint8)
    Hin, Win = int(input_size[0]), int(input_size[1])
    mask_input = cv2.resize(mask_bin, (Win, Hin), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask_input


# -----------------------------
# CSV-Iteration und Paketierung
# -----------------------------

_NAME_RE = re.compile(r"^Bild(\d+),\s*Query(\d+)$")


def _iter_csv_rows(csv_path: str) -> Iterable[Tuple[int, int, np.ndarray]]:
    """
    Iteriert Zeilen einer Query.csv und liefert (image_idx, query_idx, query_features).
    
    query_features ist ein 1D np.ndarray[256] float32.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            m = _NAME_RE.match(name)
            if not m:
                continue
            img_idx = int(m.group(1))
            query_idx = int(m.group(2))
            try:
                values = [float(x) for x in row[1:]]
            except ValueError:
                # Überspringe fehlerhafte Zeilen
                continue
            query_features = np.asarray(values, dtype=np.float32)
            yield img_idx, query_idx, query_features


def iter_decoder_iou_inputs():
    """
    Haupt-Iterator: liefert pro CSV-Zeile ein DecoderIoUInput-Paket.
    - Erkennt Layer-Index aus Ordnernamen
    - Mappt Bild-Index auf passende Pixel-Embeddings
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
    
    # Cache: Maske je input_size
    mask_cache_input: Dict[Tuple[int, int], np.ndarray] = {}
    
    for lidx, csv_path in layer_csvs:
        for img_idx, query_idx, query_features in _iter_csv_rows(csv_path):
            embedding_data = _select_pixel_embedding_for(img_idx, all_embeddings)
            if embedding_data is None:
                # Ohne Pixel-Embeddings ist IoU-Berechnung nicht möglich
                continue
            
            input_size = _get_input_size_from_embedding(embedding_data)
            input_size = _validate_input_size(input_size, embedding_data["metadata"])
            
            # Maske beschaffen (gecacht nach input-Größe)
            if input_size in mask_cache_input:
                mask_input = mask_cache_input[input_size]
            else:
                mask_input = _load_mask_for_input(input_size)
                mask_cache_input[input_size] = mask_input
            
            yield DecoderIoUInput(
                layer_idx=lidx,
                image_idx=img_idx,
                query_idx=query_idx,
                query_features=query_features,
                pixel_embedding=embedding_data["embedding"],
                input_size=input_size,
                mask_input=mask_input,
            )


# -----------------------------
# Export-Funktionen
# -----------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _export_root() -> str:
    # Sammelordner für IoU-Ergebnisse
    return os.path.join(_decoder_out_dir(), "iou_results")


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    
    fieldnames = [
        "layer_idx",
        "image_idx", 
        "query_idx",
        "iou",
        "threshold",
        "positives",
        "heatmap_path",
        "overlay_path",
    ]
    
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main_export_decoder_iou() -> None:
    """
    Berechnet IoUs für alle Decoder-Queries und exportiert Ergebnisse.
    
    Ausgabe-Struktur:
    - myThesis/output/decoder/iou_results/layer<L>/iou_sorted.csv
    - myThesis/output/decoder/iou_results/layer<L>/heatmaps/Bild<I>_Query<Q>.png
    - myThesis/output/decoder/iou_results/layer<L>/comparisons/best_Bild<I>_Query<Q>.png
    """
    # Lazy-Import von iou_core_decoder
    try:
        from .iou_core_decoder import compute_iou_decoder, save_heatmap_png, save_overlay_comparison  # type: ignore
    except Exception:
        import os as _os, sys as _sys
        _sys.path.append(_os.path.dirname(__file__))
        from iou_core_decoder import compute_iou_decoder, save_heatmap_png, save_overlay_comparison  # type: ignore
    
    export_root = _export_root()
    _ensure_dir(export_root)
    
    # Sammle Ergebnisse pro Layer
    per_layer: Dict[int, List[Dict[str, object]]] = {}
    # Tracking der besten Einträge pro Layer
    per_layer_best: Dict[int, Dict[str, object]] = {}
    
    count = 0
    for item in iter_decoder_iou_inputs():
        res = compute_iou_decoder(
            item,
            threshold_method="percentile",
            threshold_value=80.0,
            threshold_absolute=None,
            return_heatmap=True,
        )
        
        layer_dir = os.path.join(export_root, f"layer{res.layer_idx}")
        heat_dir = os.path.join(layer_dir, "heatmaps")
        _ensure_dir(heat_dir)
        heat_name = f"Bild{res.image_idx}_Query{res.query_idx}.png"
        heat_path = os.path.join(heat_dir, heat_name)
        
        if res.heatmap is not None:
            save_heatmap_png(heat_path, res.heatmap)
        else:
            heat_path = ""
        
        row = {
            "layer_idx": res.layer_idx,
            "image_idx": res.image_idx,
            "query_idx": res.query_idx,
            "iou": float(res.iou),
            "threshold": float(res.threshold),
            "positives": int(res.positives),
            "heatmap_path": os.path.relpath(heat_path, start=export_root) if heat_path else "",
            "overlay_path": "",
        }
        per_layer.setdefault(res.layer_idx, []).append(row)
        
        # Bestleistung pro Layer aktualisieren
        best = per_layer_best.get(res.layer_idx)
        if best is None:
            per_layer_best[res.layer_idx] = {
                "best_iou": float(res.iou),
                "items": [
                    {
                        "image_idx": res.image_idx,
                        "query_idx": res.query_idx,
                        "threshold": float(res.threshold),
                        "heatmap": res.heatmap,
                        "mask_input": item.mask_input,
                    }
                ],
            }
        else:
            cur_best = float(best["best_iou"])  # type: ignore
            if float(res.iou) > cur_best + 1e-12:
                best["best_iou"] = float(res.iou)
                best["items"] = [
                    {
                        "image_idx": res.image_idx,
                        "query_idx": res.query_idx,
                        "threshold": float(res.threshold),
                        "heatmap": res.heatmap,
                        "mask_input": item.mask_input,
                    }
                ]
            elif abs(float(res.iou) - cur_best) <= 1e-12:
                best.setdefault("items", []).append(
                    {
                        "image_idx": res.image_idx,
                        "query_idx": res.query_idx,
                        "threshold": float(res.threshold),
                        "heatmap": res.heatmap,
                        "mask_input": item.mask_input,
                    }
                )
        
        count += 1
    
    # Erzeuge Overlays der besten Queries und schreibe pro Layer CSV
    for lidx, rows in per_layer.items():
        # Overlays für beste Einträge
        best = per_layer_best.get(lidx)
        overlay_map: Dict[Tuple[int, int], str] = {}
        if best is not None:
            items = best.get("items", [])  # type: ignore
            layer_dir = os.path.join(export_root, f"layer{lidx}")
            cmp_dir = os.path.join(layer_dir, "comparisons")
            _ensure_dir(cmp_dir)
            for it in items:  # type: ignore
                img_idx = int(it["image_idx"])  # type: ignore
                query_idx = int(it["query_idx"])  # type: ignore
                thr = float(it["threshold"])  # type: ignore
                hm = it["heatmap"]  # type: ignore
                msk = it["mask_input"]  # type: ignore
                if hm is None:
                    continue
                cmp_name = f"best_Bild{img_idx}_Query{query_idx}.png"
                cmp_path = os.path.join(cmp_dir, cmp_name)
                save_overlay_comparison(cmp_path, msk, hm, thr)
                overlay_map[(img_idx, query_idx)] = os.path.relpath(cmp_path, start=export_root)
        
        rows_sorted = sorted(rows, key=lambda r: r.get("iou", 0.0), reverse=True)
        # Füge overlay_path für Best-Items ein
        for r in rows_sorted:
            key = (int(r["image_idx"]), int(r["query_idx"]))
            if key in overlay_map:
                r["overlay_path"] = overlay_map[key]
        
        layer_dir = os.path.join(export_root, f"layer{lidx}")
        _ensure_dir(layer_dir)
        csv_path = os.path.join(layer_dir, "iou_sorted.csv")
        _write_csv(csv_path, rows_sorted)
    
    if count == 0:
        print("Keine Daten gefunden. Bitte zuvor die Decoder-Extraktion ausführen.")
    else:
        print(f"Decoder IoU Export abgeschlossen. Root: {export_root}")
        print(f"Verarbeitet: {count} Query-Embeddings")


def main_print_all() -> None:
    """
    Berechnet und druckt alle IoUs für Decoder-Queries.
    
    Ausgabe in der Form:
    Layer=<L> Bild=<B> Query=<Q> IoU=<IOU> thr=<T> pos=<N>
    """
    # Import hier durchführen
    try:
        from .iou_core_decoder import compute_iou_decoder  # type: ignore
    except Exception:
        import os as _os, sys as _sys
        _sys.path.append(_os.path.dirname(__file__))
        from iou_core_decoder import compute_iou_decoder  # type: ignore
    
    count = 0
    for item in iter_decoder_iou_inputs():
        result = compute_iou_decoder(
            item,
            threshold_method="percentile",
            threshold_value=80.0,
            threshold_absolute=None,
        )
        print(
            f"Layer={result.layer_idx} Bild={result.image_idx} Query={result.query_idx} "
            f"IoU={result.iou:.6f} thr={result.threshold:.4f} pos={result.positives}"
        )
        count += 1
    
    if count == 0:
        print("Keine Daten gefunden. Bitte zuvor die Decoder-Extraktion ausführen.")


if __name__ == "__main__":
    # Exportiere IoU-Ergebnisse für Decoder-Queries
    main_export_decoder_iou()


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
