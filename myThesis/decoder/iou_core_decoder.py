# Kern von MaskDINO Decoder

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass 
class DecoderIoUInput:
    layer_idx: int
    image_id: str 
    query_idx: int
    query_features: np.ndarray
    pixel_embedding: np.ndarray
    input_size: Tuple[int, int]
    mask_input: np.ndarray

# Das berechnen der Map welche Werte wichtig sind, passiert hier über ein Dot-Produkt von dem jeweiligen Query mit dem Pixel-Embedding

def _compute_query_response_map(
    query_features: np.ndarray, 
    pixel_embedding: np.ndarray
) -> np.ndarray:
    # Dot-Product: (256,) @ (256, H*W) -> (H*W) -> (H, W)
    _, H, W = pixel_embedding.shape
    pixel_flat = pixel_embedding.reshape(pixel_embedding.shape[0], -1)  # (256, H*W)
    response_flat = np.dot(query_features, pixel_flat)  # (H*W,)
    response_map = response_flat.reshape(H, W)  # (H, W)
    
    return response_map

# Dann müssen wir die Map noch zurück auf Input-Größe skalieren

def _scale_to_input_size(
    response_map: np.ndarray, 
    target_size: Tuple[int, int]
) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) wird benötigt, ist aber nicht verfügbar.")
    
    H_target, W_target = target_size
    scaled = cv2.resize(
        response_map.astype(np.float32), 
        (W_target, H_target), 
        interpolation=cv2.INTER_LINEAR
    )
    return scaled


# Wir haben nen Threshold pro Query und den wenden wir hier für die Binarisierung an


def apply_per_query_binarization(
    heatmap: np.ndarray,
    threshold: float,
) -> np.ndarray:
    binary_map = heatmap >= threshold
    return binary_map

# Jetzt können wir die überschneidung zweier Masken berechnen und bekommen so die IoU

def _compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:

    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    
    intersection_count = np.count_nonzero(intersection)
    union_count = np.count_nonzero(union)
    
    if union_count == 0:
        return 0.0
    
    return float(intersection_count) / float(union_count)

