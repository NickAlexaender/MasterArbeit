from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import MAX_INPUT_SIZE
from .io_utils import mask_dir as _default_mask_dir

logger = logging.getLogger(__name__)


def _import_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "OpenCV (cv2) is required for mask operations. Install with 'pip install opencv-python'."
        ) from e

# Umwandeln von Farbmasken in Binäre

def prepare_mask_binary(mask_bgr: np.ndarray) -> np.ndarray:

    if mask_bgr.ndim == 2:
        return (mask_bgr > 0)

    b = mask_bgr[..., 0].astype(np.int16)
    g = mask_bgr[..., 1].astype(np.int16)
    r = mask_bgr[..., 2].astype(np.int16)

    red_dominant = (r > 150) & (g < 100) & (b < 100)
    if np.count_nonzero(red_dominant) == 0:
        non_black = (r > 0) | (g > 0) | (b > 0)
        return non_black
    return red_dominant

# Aus den Metadaten werden die Input-Größen berechnet

def get_input_size_from_embedding(metadata: Dict[str, Any]) -> Tuple[int, int]:

    if "input_h" in metadata and "input_w" in metadata:
        return int(metadata["input_h"]), int(metadata["input_w"])

    embed_h = int(metadata.get("embed_h", metadata.get("height", 25)))
    embed_w = int(metadata.get("embed_w", metadata.get("width", 25)))

    if "stride" in metadata:
        stride = int(metadata["stride"])  # type: ignore
        return embed_h * stride, embed_w * stride

    return embed_h * 32, embed_w * 32

# Nun können wir die Input-Größen validieren

def validate_input_size(
    input_size: Tuple[int, int],
    metadata: Dict[str, Any],
    max_size: int = MAX_INPUT_SIZE,
) -> Tuple[int, int]:
    input_h, input_w = int(input_size[0]), int(input_size[1])
    embed_h = int(metadata.get("embed_h", metadata.get("height", 25)))
    embed_w = int(metadata.get("embed_w", metadata.get("width", 25)))

    if input_h < embed_h or input_w < embed_w:
        logger.warning(
            "Input size (%dx%d) smaller than embedding (%dx%d). Bumping to minimum.",
            input_h,
            input_w,
            embed_h,
            embed_w,
        )
        input_h = max(input_h, embed_h)
        input_w = max(input_w, embed_w)

    if input_h > max_size or input_w > max_size:
        logger.warning(
            "Input size (%dx%d) is large. Capping to %dx%d.", input_h, input_w, max_size, max_size
        )
        input_h = min(input_h, max_size)
        input_w = min(input_w, max_size)

    return input_h, input_w

# Jetzt laden wir die MAsken für ein Bild und skalieren sie auf die Input-Größe

def load_mask_for_image(
    image_id: str,
    input_size: Tuple[int, int],
    mask_dir: Optional[str] = None,
) -> np.ndarray:

    cv2 = _import_cv2()
    mdir = mask_dir or _default_mask_dir()

    search_name = image_id.replace("_", " ")
    mask_file: Optional[str] = None
    for ext in [".jpg", ".png", ".jpeg"]:
        candidate = os.path.join(mdir, f"{search_name}{ext}")
        if os.path.isfile(candidate):
            mask_file = candidate
            break
        candidate = os.path.join(mdir, f"{image_id}{ext}")
        if os.path.isfile(candidate):
            mask_file = candidate
            break

    if not mask_file:
        raise FileNotFoundError(
            f"Mask for {image_id} not found in {mdir} (tried '{search_name}.*' and '{image_id}.*')."
        )

    m = cv2.imread(mask_file, cv2.IMREAD_COLOR)
    if m is None:
        raise FileNotFoundError(f"Mask could not be loaded: {mask_file}")

    mask_bin = prepare_mask_binary(m).astype(np.uint8)
    Hin, Win = int(input_size[0]), int(input_size[1])
    mask_input = cv2.resize(mask_bin, (Win, Hin), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask_input

# Wir speichern die Overlay-Visualisierung ab.
# Rot -> Maske
# Blau -> Übereinstimmung
# Grün -> Vorhersage
# Schwarz -> Rest

def save_overlay(path: str, mask: np.ndarray, bin_map: np.ndarray) -> None:

    cv2 = _import_cv2()

    mask_b = mask.astype(bool)
    bin_b = bin_map.astype(bool)
    inter = np.logical_and(mask_b, bin_b)
    mask_only = np.logical_and(mask_b, np.logical_not(bin_b))
    hm_only = np.logical_and(bin_b, np.logical_not(mask_b))

    H, W = mask.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)

    # OpenCV uses BGR channel order
    img[inter, 0] = 255      # Blue: TP (mask ∧ pred)
    img[mask_only, 2] = 255  # Red: FN (mask ∧ ¬pred)
    img[hm_only, 1] = 255    # Green: FP (pred ∧ ¬mask)

    ok = cv2.imwrite(path, img)
    if not ok:
        logger.warning("Failed to write overlay to: %s", path)
