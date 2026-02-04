from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CSV_MIOU_FILENAME
from .io_utils import ensure_dir, resolve_export_root
from .iterators import iter_decoder_iou_inputs
from .iou_core import _compute_query_response_map, scale_to_input_size, apply_per_query_binarization
from .mask_utils import save_overlay

logger = logging.getLogger(__name__)

# Wir exportieren die Angaben in ne CSV pro LAyer

def export_mean_iou_csv(
    mean_iou_results: Dict[int, List[Dict[str, object]]],
    percentile: float = 99.5,
    export_root: Optional[str] = None,
    decoder_out_dir: Optional[str] = None,
) -> None:

    root = resolve_export_root(export_root, decoder_out_dir)
    logger.info("Exporting mIoU CSVs to: %s", root)

    for layer_idx, results in mean_iou_results.items():
        layer_dir = os.path.join(root, f"layer{layer_idx}")
        ensure_dir(layer_dir)
        csv_path = os.path.join(layer_dir, CSV_MIOU_FILENAME)

        fieldnames = ["query_idx", "mean_iou", "num_images"]
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        logger.info("Layer %d: %d queries -> %s", layer_idx, len(results), csv_path)

# Wir wollen auch sehen, wie die masken aussehen, deswegen visualisieren wir die besten.
# Blau -> überschneidung
# Rot -> vorhersage
# Grün -> das was wir predicten
# Schwarz -> alles andere

def create_best_query_visualizations(
    mean_iou_results: Dict[int, List[Dict[str, object]]],
    query_thresholds: Dict[Tuple[int, int], float],
    decoder_out_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
    export_root: Optional[str] = None,
) -> None:

    root = resolve_export_root(export_root, decoder_out_dir)
    logger.info("Creating visualizations in: %s", root)

    for layer_idx, results in mean_iou_results.items():
        if not results:
            continue

        best_query = results[0]
        best_query_idx = int(best_query["query_idx"])
        best_miou = float(best_query["mean_iou"])  # noqa: F841 (for logging readability)

        key = (layer_idx, best_query_idx)
        threshold = query_thresholds.get(key)
        if threshold is None:
            logger.warning("No threshold for best query (layer=%d, query=%d). Skipping.", layer_idx, best_query_idx)
            continue

        saved = 0
        for item in iter_decoder_iou_inputs(decoder_out_dir=decoder_out_dir, mask_dir=mask_dir):
            if item.layer_idx != layer_idx or item.query_idx != best_query_idx:
                continue

            response_map = _compute_query_response_map(item.query_features, item.pixel_embedding)
            heatmap_scaled = scale_to_input_size(response_map, item.input_size)
            binary_map = apply_per_query_binarization(heatmap_scaled, threshold)

            layer_dir = os.path.join(root, f"layer{layer_idx}")
            ensure_dir(layer_dir)
            vis_dir = os.path.join(layer_dir, "visualizations")
            ensure_dir(vis_dir)

            vis_filename = f"best_query_{best_query_idx}_{item.image_id}.png"
            vis_path = os.path.join(vis_dir, vis_filename)
            save_overlay(vis_path, item.mask_input, binary_map)

            saved += 1
            if saved % 25 == 0:
                logger.info("Layer %d, query %d: saved %d visualizations…", layer_idx, best_query_idx, saved)

        if saved == 0:
            logger.warning("Layer %d: no data for visualization.", layer_idx)
        else:
            logger.info("Layer %d: saved %d visualizations.", layer_idx, saved)
