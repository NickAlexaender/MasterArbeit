from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

# Datenstruktur für IoU/Heatmap-Berechnungen pro Feature

@dataclass
class IoUInput:
    layer_idx: int
    image_id: str
    feature_idx: int
    tokens: np.ndarray
    shapes: Dict
    mask_input: np.ndarray

# Ergebnis einer kombinierten Heatmap über alle Levels

@dataclass
class IoUCombinedResult:
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
