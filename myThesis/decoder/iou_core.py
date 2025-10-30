from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _compute_query_response_map(query_features: np.ndarray, pixel_embedding: np.ndarray) -> np.ndarray:
    """Compute response map as dot-product between query and pixel embeddings.

    Args:
        query_features: shape [C]
        pixel_embedding: shape [C, H, W]

    Returns:
        response_map: shape [H, W] (float32)
    """

    q = np.asarray(query_features, dtype=np.float32)
    pe = np.asarray(pixel_embedding, dtype=np.float32)

    if pe.ndim != 3:
        raise ValueError(f"pixel_embedding must be [C,H,W], got shape {pe.shape}")
    if q.ndim != 1:
        raise ValueError(f"query_features must be [C], got shape {q.shape}")
    if pe.shape[0] != q.shape[0]:
        raise ValueError(f"Channel mismatch: query C={q.shape[0]} vs embedding C={pe.shape[0]}")

    # Dot over channel dimension
    # pe: [C,H,W], q: [C] -> (H,W)
    H, W = pe.shape[1], pe.shape[2]
    response = np.tensordot(q, pe, axes=(0, 0)).astype(np.float32)
    if response.shape != (H, W):
        response = response.reshape(H, W)
    return response


def scale_to_input_size(response_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D response map to (H, W) using bilinear interpolation.

    Ziel: feinere, pixelgenaue Heatmaps für die anschließende Binarisierung,
    wie in gängigen Network-Dissection-Setups üblich. Die per-Query Schwelle
    bleibt unverändert (aus Verteilungen der ursprünglichen Feature-Map),
    angewendet wird sie auf die auf Eingabegröße bilinear hochgerechnete
    Response-Map.

    Hinweis: Falls streng blockige Masken gewünscht sind (z. B. für
    Aktivierungs-Massenerhaltung auf dem Feature-Grid), wäre "nearest"
    geeignet. Für feinkörnige Visuals und pixelgenaue Vorhersageflächen ist
    bilinear vorzuziehen.

    Lazy-imports OpenCV to avoid hard dependency at import time.
    """

    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) is required for resizing. Install with 'pip install opencv-python'."
        ) from e

    Hin, Win = int(target_size[0]), int(target_size[1])
    rm = np.asarray(response_map, dtype=np.float32)
    resized = cv2.resize(rm, (Win, Hin), interpolation=cv2.INTER_LINEAR)
    return resized


def apply_per_query_binarization(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize heatmap using a per-query threshold (>=). Returns uint8 mask."""

    hm = np.asarray(heatmap, dtype=np.float32)
    return (hm >= float(threshold)).astype(np.uint8)


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two boolean/0-1 masks.

    Returns 0.0 when the union is zero.
    """

    m1 = mask1.astype(bool)
    m2 = mask2.astype(bool)
    inter = np.logical_and(m1, m2).sum(dtype=np.int64)
    union = np.logical_or(m1, m2).sum(dtype=np.int64)
    if union == 0:
        return 0.0
    return float(inter) / float(union)
