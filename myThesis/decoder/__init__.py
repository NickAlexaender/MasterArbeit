"""Decoder toolkit for Network Dissection (MaskDINO).

Public entrypoints live in `decoder.pipeline` and the CLI module
`decoder.calculate_IoU_for_decoder`.

Modules:
- config: constants and defaults
- models: dataclasses and typed payloads
- io_utils: filesystem helpers and parsers
- mask_utils: mask loading and preprocessing
- iou_core: pure numerics (response, resize, binarize, IoU)
- iterators: streaming generators for decoder inputs
- aggregator: thresholds and mIoU computations
- export_utils: CSV and visualization helpers
- pipeline: orchestration of the full workflow
"""

from .config import (
    DEFAULT_PERCENTILE,
    MAX_INPUT_SIZE,
    EXPORT_SUBDIR,
    MASK_DEFAULT_SUBDIR,
    DECODER_DEFAULT_SUBDIR,
    CSV_MIOU_FILENAME,
)

__all__ = [
    "DEFAULT_PERCENTILE",
    "MAX_INPUT_SIZE",
    "EXPORT_SUBDIR",
    "MASK_DEFAULT_SUBDIR",
    "DECODER_DEFAULT_SUBDIR",
    "CSV_MIOU_FILENAME",
]
# decoder package