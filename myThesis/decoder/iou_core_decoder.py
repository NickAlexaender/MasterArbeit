"""
IoU-Kernlogik für MaskDINO Decoder Network Dissection.

Eingabe (pro Query):
- layer_idx: int
- image_idx: int  
- query_idx: int
- query_features: np.ndarray, Shape (256,) – Query-Features für eine Query
- pixel_embedding: np.ndarray, Shape (256, H, W) – Pixel-Embeddings vom Decoder
- mask_input: np.ndarray[bool], Shape (H_in, W_in) – Ground-Truth Maske in Input-Größe

Kernschritte:
- Dot-Product zwischen Query-Features und Pixel-Embeddings berechnen
- Auf Input-Größe skalieren 
- Binarisieren (Schwellenwertstrategie konfigurierbar)
- IoU mit mask_input berechnen
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# -----------------------------
# Datenstrukturen
# -----------------------------

@dataclass
class IoUResultDecoder:
    layer_idx: int
    image_idx: int
    query_idx: int
    iou: float
    threshold: float
    positives: int
    heatmap: Optional[np.ndarray] = None


@dataclass 
class DecoderIoUInput:
    layer_idx: int
    image_idx: int
    query_idx: int
    query_features: np.ndarray  # Shape: (256,)
    pixel_embedding: np.ndarray  # Shape: (256, H, W) 
    input_size: Tuple[int, int]  # (H_in, W_in)
    mask_input: np.ndarray  # bool, Shape: (H_in, W_in)


# -----------------------------
# Kernfunktionen
# -----------------------------

def _compute_query_response_map(
    query_features: np.ndarray, 
    pixel_embedding: np.ndarray
) -> np.ndarray:
    """
    Berechnet Response-Map durch Dot-Product zwischen Query und Pixel-Embeddings.
    
    Args:
        query_features: Shape (256,)
        pixel_embedding: Shape (256, H, W)
    
    Returns:
        response_map: Shape (H, W) - Aktivierungsstärke pro Pixel
    """
    # Dot-Product: (256,) @ (256, H*W) -> (H*W) -> (H, W)
    _, H, W = pixel_embedding.shape
    pixel_flat = pixel_embedding.reshape(pixel_embedding.shape[0], -1)  # (256, H*W)
    response_flat = np.dot(query_features, pixel_flat)  # (H*W,)
    response_map = response_flat.reshape(H, W)  # (H, W)
    
    return response_map


def _scale_to_input_size(
    response_map: np.ndarray, 
    target_size: Tuple[int, int]
) -> np.ndarray:
    """
    Skaliert Response-Map auf Input-Größe mittels bilinearer Interpolation.
    
    Args:
        response_map: Shape (H, W)
        target_size: (H_target, W_target)
    
    Returns:
        scaled_map: Shape (H_target, W_target)
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) wird benötigt, ist aber nicht verfügbar.")
    
    H_target, W_target = target_size
    scaled = cv2.resize(
        response_map.astype(np.float32), 
        (W_target, H_target), 
        interpolation=cv2.INTER_LINEAR
    )
    return scaled


def _apply_threshold(
    heatmap: np.ndarray,
    threshold_method: str = "percentile",
    threshold_value: float = 80.0,
    threshold_absolute: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """
    Binarisiert Heatmap basierend auf Schwellenwert-Strategie.
    
    Args:
        heatmap: Shape (H, W)
        threshold_method: "percentile", "mean", "median", "absolute"
        threshold_value: Wert für percentile-Methode (0-100)
        threshold_absolute: Fester Schwellenwert für "absolute"-Methode
    
    Returns:
        binary_map: bool array, Shape (H, W)
        actual_threshold: verwendeter Schwellenwert
    """
    if threshold_method == "absolute" and threshold_absolute is not None:
        threshold = float(threshold_absolute)
    elif threshold_method == "percentile":
        threshold = float(np.percentile(heatmap, threshold_value))
    elif threshold_method == "mean":
        threshold = float(np.mean(heatmap))
    elif threshold_method == "median":
        threshold = float(np.median(heatmap))
    else:
        raise ValueError(f"Unbekannte threshold_method: {threshold_method}")
    
    binary_map = heatmap >= threshold
    return binary_map, threshold


def _compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Berechnet Intersection over Union zwischen zwei binären Masken.
    
    Args:
        mask1, mask2: bool arrays gleicher Größe
    
    Returns:
        iou: IoU-Wert zwischen 0 und 1
    """
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    
    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)
    
    if union_count == 0:
        return 0.0
    
    return float(intersection_count) / float(union_count)


# -----------------------------
# Haupt-IoU-Funktion
# -----------------------------

def compute_iou_decoder(
    item: DecoderIoUInput,
    threshold_method: str = "percentile",
    threshold_value: float = 80.0,
    threshold_absolute: Optional[float] = None,
    return_heatmap: bool = False,
) -> IoUResultDecoder:
    """
    Berechnet IoU für eine Decoder-Query.
    
    Args:
        item: Eingabedaten für eine Query
        threshold_method: Schwellenwert-Strategie
        threshold_value: Parameter für Schwellenwert
        threshold_absolute: Fester Schwellenwert (falls method="absolute")
        return_heatmap: Ob skalierte Heatmap zurückgegeben werden soll
    
    Returns:
        IoUResultDecoder mit berechneten Werten
    """
    # 1. Query-Response-Map berechnen
    response_map = _compute_query_response_map(
        item.query_features, 
        item.pixel_embedding
    )
    
    # 2. Auf Input-Größe skalieren
    heatmap_scaled = _scale_to_input_size(response_map, item.input_size)
    
    # 3. Binarisieren
    binary_map, threshold = _apply_threshold(
        heatmap_scaled,
        threshold_method=threshold_method,
        threshold_value=threshold_value,
        threshold_absolute=threshold_absolute,
    )
    
    # 4. IoU berechnen
    iou = _compute_iou(binary_map, item.mask_input)
    positives = int(np.count_nonzero(binary_map))
    
    # 5. Ergebnis zusammenstellen
    result = IoUResultDecoder(
        layer_idx=item.layer_idx,
        image_idx=item.image_idx,
        query_idx=item.query_idx,
        iou=iou,
        threshold=threshold,
        positives=positives,
        heatmap=heatmap_scaled if return_heatmap else None,
    )
    
    return result


# -----------------------------
# Hilfsfunktionen für Export/Visualisierung 
# -----------------------------

def save_heatmap_png(dest_path: str, heatmap: np.ndarray) -> None:
    """Speichert eine float-Heatmap als PNG (0-255 skaliert)."""
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) wird benötigt, ist aber nicht verfügbar.")
    
    h = heatmap.astype(np.float32, copy=False)
    vmin = float(h.min())
    vmax = float(h.max())
    
    if vmax <= vmin + 1e-8:
        img = np.zeros_like(h, dtype=np.uint8)
    else:
        hn = (h - vmin) / (vmax - vmin)
        img = np.clip(hn * 255.0, 0, 255).astype(np.uint8)
    
    import os
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cv2.imwrite(dest_path, img)


def save_overlay_comparison(
    dest_path: str, 
    mask_input: np.ndarray, 
    heatmap: np.ndarray, 
    threshold: float
) -> None:
    """
    Erstellt und speichert Vergleichsbild:
    - Blau  (BGR=255,0,0): Überschneidung (Maske ∧ Binär-Heatmap)
    - Rot   (BGR=0,0,255): Maske nur
    - Gelb  (BGR=0,255,255): Binär-Heatmap nur
    - Schwarz: Rest
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) wird benötigt, ist aber nicht verfügbar.")
    
    mask = mask_input.astype(bool)
    bin_hm = (heatmap.astype(np.float32) >= float(threshold))
    
    inter = np.logical_and(mask, bin_hm)
    mask_only = np.logical_and(mask, np.logical_not(bin_hm))
    hm_only = np.logical_and(bin_hm, np.logical_not(mask))
    
    H, W = mask.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)  # BGR
    
    # Blau für Überschneidung
    img[inter, 0] = 255  # B
    # Rot für Maske-only  
    img[mask_only, 2] = 255  # R
    # Gelb für Heatmap-only (R+G)
    img[hm_only, 1] = 255  # G
    img[hm_only, 2] = 255  # R
    
    import os
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cv2.imwrite(dest_path, img)
