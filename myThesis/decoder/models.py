from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class DecoderIoUInput:
    """Payload for IoU computation per (layer, image, query).

    Fields mirror the original monolithic implementation's expectations.
    - query_features: shape [C]
    - pixel_embedding: shape [C, H, W]
    - mask_input: boolean mask with shape equal to input_size
    """

    layer_idx: int
    image_id: str
    query_idx: int
    query_features: np.ndarray
    pixel_embedding: np.ndarray
    input_size: Tuple[int, int]
    mask_input: np.ndarray  # dtype bool


@dataclass(frozen=True)
class IoUCombinedResult:
    """Optional combined result holder for visualization or debugging.

    Not required by the core pipeline but handy for testing or future
    extensions without adding logic to models.py.
    """

    layer_idx: int
    image_id: str
    query_idx: int
    threshold: float
    heatmap: np.ndarray
    binary_map: np.ndarray
    mask_input: np.ndarray
    iou: float
