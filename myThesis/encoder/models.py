"""Gemeinsame Datenmodelle für die Encoder-Pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class IoUInput:
    """Eingabe-Paket für IoU/Heatmap-Berechnungen.

    - layer_idx: Layer-Index
    - image_id: Bild-ID (z. B. "image 1")
    - feature_idx: Feature-Index (1-basiert)
    - tokens: 1D-Array (N,) mit Aktivierungswerten
    - shapes: shapes.json-Inhalt (u. a. spatial_shapes, level_start_index, input_size)
    - mask_input: boolsche Maske in Input-Größe (H_in, W_in)
    """
    layer_idx: int
    image_id: str
    feature_idx: int
    tokens: np.ndarray
    shapes: Dict
    mask_input: np.ndarray


@dataclass
class IoUCombinedResult:
    """Ergebnis einer kombinierten Heatmap über alle Levels (auf Inputgröße)."""
    layer_idx: int
    image_id: str  # String-ID (z.B. "image 1")
    feature_idx: int
    map_shape: Tuple[int, int]  # entspricht input_size (Hin, Win)
    threshold: float
    iou: float
    positives: int
    heatmap: Optional[np.ndarray] = None


__all__ = [
    "IoUInput",
    "IoUCombinedResult",
]
