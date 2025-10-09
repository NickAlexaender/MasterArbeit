"""
IoU-Kernlogik für MaskDINO Decoder Network Dissection.

Eingabe (pro Query):
- layer_idx: int
- image_id: str – Eindeutige Bild-ID (z.B. "image_1")
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
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


# -----------------------------
# Datenstrukturen
# -----------------------------

# Hinweis: IoUResultDecoder wird nicht verwendet, da die Aufrufe aus calculate_IoU_for_decoder
# direkt die Kernfunktionen (_compute_query_response_map, _scale_to_input_size, apply_per_query_binarization, _compute_iou)
# nutzen. Daher entfernen wir diese Struktur.


@dataclass 
class DecoderIoUInput:
    layer_idx: int
    image_id: str  # Geändert: image_id statt image_idx
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


# Die generische Threshold-Funktion wird nicht mehr benötigt (per-Query Threshold erfolgt extern)


def apply_per_query_binarization(
    heatmap: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Binarisiert Heatmap mit vorberechnetem per-Query Threshold.
    
    Args:
        heatmap: Shape (H, W) - skalierte Response-Map
        threshold: vorberechneter Query-Threshold
    
    Returns:
        binary_map: bool array, Shape (H, W)
    """
    binary_map = heatmap >= threshold
    return binary_map


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

"""
Die Hilfsfunktion compute_iou_decoder und der Typ IoUResultDecoder werden nicht benötigt,
da der Aufrufer in calculate_IoU_for_decoder.py die Schritte explizit ausführt.
"""


# -----------------------------
# Hilfsfunktionen für Export/Visualisierung 
# -----------------------------

# Die Export-/Overlay-Funktion wird nicht benötigt
