# Hier haben wir 4 Schritte in der Pipeline: 
# - Berechnung der Thresholds, 
# - Berechnung der mIoUs, 
# - Export der CSVs,
# - Erstellung der Visualisierungen

from __future__ import annotations
import logging
import os
from typing import Dict, List, Optional, Tuple
from .config import DEFAULT_PERCENTILE
from .io_utils import decoder_out_dir as _decoder_out_dir_default
from .io_utils import mask_dir as _mask_dir_default
from .io_utils import resolve_export_root
from .aggregator import compute_per_query_thresholds, compute_mean_iou_per_query
from .export_utils import export_mean_iou_csv, create_best_query_visualizations

logger = logging.getLogger(__name__)

def main_network_dissection_per_query(
    percentile: float = DEFAULT_PERCENTILE,
    mask_dir: Optional[str] = None,
    decoder_out_dir: Optional[str] = None,
    export_root: Optional[str] = None,
) -> None:

    decoder_out_dir = _decoder_out_dir_default(decoder_out_dir)
    mask_dir = _mask_dir_default(mask_dir)
    export_root = resolve_export_root(export_root, decoder_out_dir)

    logger.info("Starting Network Dissection (percentile=%.3f)", percentile)
    logger.info("Paths:\n  • decoder_out_dir: %s\n  • mask_dir: %s\n  • export_root: %s", decoder_out_dir, mask_dir, export_root)

    # 1) thresholds
    logger.info("[1/4] Computing thresholds…")
    thresholds = compute_per_query_thresholds(percentile=percentile, decoder_out_dir=decoder_out_dir, mask_dir=mask_dir)
    if not thresholds:
        logger.error("No thresholds computed. Aborting.")
        return

    # 2) mIoU
    logger.info("[2/4] Computing mean IoU per query…")
    mean_iou_results = compute_mean_iou_per_query(thresholds, decoder_out_dir=decoder_out_dir, mask_dir=mask_dir)

    # 3) CSV export
    logger.info("[3/4] Exporting CSVs…")
    export_mean_iou_csv(mean_iou_results, percentile=percentile, export_root=export_root, decoder_out_dir=decoder_out_dir)

    # 4) Visualizations
    logger.info("[4/4] Creating visualizations…")
    create_best_query_visualizations(mean_iou_results, thresholds, decoder_out_dir=decoder_out_dir, mask_dir=mask_dir, export_root=export_root)

    # Summary
    logger.info("Network Dissection finished.")
    total_queries = len(thresholds)
    total_layers = len(mean_iou_results)
    logger.info("Stats: queries=%d, layers=%d, percentile=%.3f", total_queries, total_layers, percentile)

    # Top-3 per layer
    for layer_idx in sorted(mean_iou_results.keys()):
        results = mean_iou_results[layer_idx]
        if results:
            top = sorted(results, key=lambda x: x["mean_iou"], reverse=True)[:3]
            logger.info("Top queries for layer %d:", layer_idx)
            for i, q in enumerate(top, 1):
                logger.info("  %d. Query %s: mIoU=%.4f (%d images)", i, q["query_idx"], q["mean_iou"], q["num_images"])

    logger.info("Output directory: %s", export_root)
