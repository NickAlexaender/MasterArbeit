from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from .iou_core import (
    _compute_query_response_map,
    scale_to_input_size,
    apply_per_query_binarization,
    compute_iou,
)
from .iterators import iter_decoder_iou_inputs

logger = logging.getLogger(__name__)


def compute_per_query_thresholds(
    percentile: float = 99.5,
    decoder_out_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
) -> Dict[Tuple[int, int], float]:
    """Compute per-query thresholds across all images using an activation percentile.

    Returns mapping (layer_idx, query_idx) -> threshold.
    """

    # Accept both [0, 100] and fractional [0, 1] inputs for percentile.
    # Users sometimes pass 0.90 to mean "top 10%"; numpy expects 90.0.
    pct = float(percentile)
    if 0.0 < pct <= 1.0:
        pct *= 100.0
        logger.debug(
            "Percentile provided as fraction; normalized to %.3f (x100)", pct
        )
    # Clamp to valid numpy percentile range
    pct = max(0.0, min(100.0, pct))

    logger.info("Computing per-query thresholds (percentile=%.3f)", pct)

    activation_values: Dict[Tuple[int, int], List[float]] = {}

    for item in iter_decoder_iou_inputs(decoder_out_dir=decoder_out_dir, mask_dir=mask_dir):
        key = (item.layer_idx, item.query_idx)

        response_map = _compute_query_response_map(item.query_features, item.pixel_embedding)
        values = response_map.ravel().tolist()
        if not values:
            continue
        if key not in activation_values:
            activation_values[key] = []
        activation_values[key].extend(values)

    thresholds: Dict[Tuple[int, int], float] = {}
    for key, values in activation_values.items():
        if len(values) == 0:
            logger.warning("Empty activation list for %s; using threshold 0.0", key)
            thresholds[key] = 0.0
            continue
        thresholds[key] = float(np.percentile(values, pct))

    logger.info("Computed thresholds for %d queries", len(thresholds))
    return thresholds


def compute_mean_iou_per_query(
    query_thresholds: Dict[Tuple[int, int], float],
    decoder_out_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
) -> Dict[int, List[Dict[str, object]]]:
    """Compute average IoU over all images for each query.

    Returns: dict[layer_idx] -> list of {"query_idx", "mean_iou", "num_images"},
    sorted by mean_iou desc per layer.
    """

    logger.info("Computing mean IoU per query…")

    query_ious: Dict[Tuple[int, int], List[float]] = {}

    for item in iter_decoder_iou_inputs(decoder_out_dir=decoder_out_dir, mask_dir=mask_dir):
        key = (item.layer_idx, item.query_idx)
        threshold = query_thresholds.get(key)
        if threshold is None:
            continue

        response_map = _compute_query_response_map(item.query_features, item.pixel_embedding)
        heatmap_scaled = scale_to_input_size(response_map, item.input_size)
        binary_map = apply_per_query_binarization(heatmap_scaled, threshold)
        iou = compute_iou(binary_map, item.mask_input)

        if key not in query_ious:
            query_ious[key] = []
        query_ious[key].append(iou)

    per_layer_results: Dict[int, List[Dict[str, object]]] = {}
    for (layer_idx, query_idx), ious in query_ious.items():
        mean_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
        result = {"query_idx": query_idx, "mean_iou": mean_iou, "num_images": len(ious)}
        per_layer_results.setdefault(layer_idx, []).append(result)

    for layer_idx in per_layer_results:
        per_layer_results[layer_idx].sort(key=lambda x: x["mean_iou"], reverse=True)

    logger.info("Computed mIoU for %d queries across %d layers", len(query_ious), len(per_layer_results))
    return per_layer_results
