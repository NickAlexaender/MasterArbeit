"""Iteratoren für CSV -> IoUInput-Generierung."""

from __future__ import annotations

import csv
import logging
import os
import re
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np

from .io_utils import (
    _encoder_out_dir,
    _find_layer_csvs,
    _load_all_shapes,
    _select_shapes_for,
)
from .mask_utils import _load_mask_for_input
from .models import IoUInput

logger = logging.getLogger(__name__)


# Neues Format: "image 1, Feature1" statt "Bild1, Feature1"
_NAME_RE = re.compile(r"^(.+?),\s*Feature(\d+)$")


def _iter_csv_rows(csv_path: str) -> Iterable[Tuple[str, int, np.ndarray]]:
    """Iteriert Zeilen einer feature.csv und liefert (image_id, feature_idx, tokens).

    tokens ist ein 1D np.ndarray[N] float32.
    image_id ist der String-Identifikator (z.B. "image 1").
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # Header
        for row in reader:
            if not row:
                continue
            name = row[0].strip()
            m = _NAME_RE.match(name)
            if not m:
                logger.debug("Zeile ohne gültigen Namen übersprungen: %s", name)
                continue
            image_id = m.group(1).strip()
            feat_idx = int(m.group(2))
            try:
                values = [float(x) for x in row[1:]]
            except ValueError:
                logger.warning("Nicht-parsbare Werte in %s -> Zeile übersprungen", csv_path)
                continue
            tokens = np.asarray(values, dtype=np.float32)
            yield image_id, feat_idx, tokens


def iter_iou_inputs(
    encoder_out_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
) -> Generator[IoUInput, None, None]:
    """Haupt-Iterator: liefert pro CSV-Zeile ein IoUInput-Paket.

    - Erkennt Layer-Index aus Ordnernamen.
    - Mappt image_id auf passende shapes.json (direkter Match bevorzugt).
    - Bereitet Maske für input-Size vor (gecacht pro input_size UND image_id).
    """
    layer_csvs = _find_layer_csvs(encoder_out_dir)
    all_shapes = _load_all_shapes(encoder_out_dir)

    # Cache: Maske je (input_size, image_id)
    mask_cache_input: Dict[Tuple[int, int, str], np.ndarray] = {}

    for lidx, csv_path in layer_csvs:
        for image_id, feat_idx, tokens in _iter_csv_rows(csv_path):
            shapes = _select_shapes_for(image_id, tokens.size, all_shapes)
            if shapes is None:
                # Ohne Shapes ist Reassemblierung schwierig – skippe
                logger.warning("Keine shapes.json für image_id='%s' -> Zeile übersprungen", image_id)
                continue
            Hin, Win = int(shapes["input_size"][0]), int(shapes["input_size"][1])
            image_id_from_shapes = shapes.get("image_id", image_id)

            key_in = (Hin, Win, image_id_from_shapes)
            if key_in in mask_cache_input:
                mask_input = mask_cache_input[key_in]
            else:
                mask_input = _load_mask_for_input((Hin, Win), image_id_from_shapes, mask_dir=mask_dir)
                mask_cache_input[key_in] = mask_input

            yield IoUInput(
                layer_idx=lidx,
                image_id=image_id_from_shapes,
                feature_idx=feat_idx,
                tokens=tokens,
                shapes=shapes,
                mask_input=mask_input,
            )


__all__ = [
    "iter_iou_inputs",
]
