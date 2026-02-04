
from __future__ import annotations
import logging
import os
from typing import Optional, Tuple
import numpy as np
try:
    import cv2
    try:
        # Verhindere Thread-Überabonnierung; wir kontrollieren Threads via BLAS
        cv2.setNumThreads(0)
    except Exception:
        pass
except Exception:  # pragma: no cover
    cv2 = None

from .io_utils import _ensure_dir

logger = logging.getLogger(__name__)


def _require_cv2() -> None:
    if cv2 is None:
        raise ImportError(
            "OpenCV (cv2) ist nicht installiert. Bitte 'pip install opencv-python' "
            "(oder 'opencv-python-headless' für Headless-Umgebungen) ausführen."
        )

# Wandelt farbige Maske in binäre um

def _prepare_mask_binary(mask_bgr: np.ndarray) -> np.ndarray:
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

# Lädt die Maske für ein gegebenes Bild und liefert sie als bool-Array

def _load_mask_for_input(
    input_size: Tuple[int, int],
    image_id: str,
    mask_dir: Optional[str] = None,
) -> np.ndarray:
    _require_cv2()
    if mask_dir is None:
        from .io_utils import get_mask_dir

        mask_dir = get_mask_dir()

    mask_file = os.path.join(mask_dir, f"{image_id}.jpg")
    if not os.path.isfile(mask_file):
        mask_file = os.path.join(mask_dir, f"{image_id}.png")

    m = cv2.imread(mask_file, cv2.IMREAD_COLOR)
    if m is None:
        raise FileNotFoundError(f"Maske nicht gefunden: {mask_file}")

    mask_bin = _prepare_mask_binary(m).astype(np.uint8)
    Hin, Win = int(input_size[0]), int(input_size[1])
    mask_input = cv2.resize(mask_bin, (Win, Hin), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask_input

# Speichert Vergleichsbild mit Overlay der Maske und Heatmap
# Rot -> Maske, 
# Blau -> Überschneidung
# Grün -> Prediction
# Schwarz -> Rest

def _save_overlay_comparison(
    dest_path: str,
    mask_input: np.ndarray,
    heatmap: np.ndarray,
    threshold: float,
) -> None:
    _require_cv2()
    mask = mask_input.astype(bool)
    bin_hm = (heatmap.astype(np.float32) >= float(threshold))

    inter = np.logical_and(mask, bin_hm)
    mask_only = np.logical_and(mask, np.logical_not(bin_hm))
    hm_only = np.logical_and(bin_hm, np.logical_not(mask))

    H, W = mask.shape
    img = np.zeros((H, W, 3), dtype=np.uint8)  # BGR
    img[inter, 0] = 255  # Blau (TP)
    img[mask_only, 2] = 255  # Rot (FN)
    img[hm_only, 1] = 255  # Grün (FP)

    _ensure_dir(os.path.dirname(dest_path))
    cv2.imwrite(dest_path, img)


__all__ = [
    "_prepare_mask_binary",
    "_load_mask_for_input",
    "_save_overlay_comparison",
]
