from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

# Für die IoU-Berechnung brauchen wir die Datenstruktur pro Layer/Image/Query

@dataclass(frozen=True)
class DecoderIoUInput:

    layer_idx: int
    image_id: str
    query_idx: int
    query_features: np.ndarray
    pixel_embedding: np.ndarray
    input_size: Tuple[int, int]
    mask_input: np.ndarray  # dtype bool

# Pro Query-Ergebnis speichern wir die Resultate

@dataclass(frozen=True)
class IoUCombinedResult:

    layer_idx: int
    image_id: str
    query_idx: int
    threshold: float
    heatmap: np.ndarray
    binary_map: np.ndarray
    mask_input: np.ndarray
    iou: float
