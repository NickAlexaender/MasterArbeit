from __future__ import annotations

"""Central configuration and constants for the decoder pipeline.

Note: Do not configure logging here. The CLI (calculate_IoU_for_decoder.py)
initializes logging. Modules only create module-level loggers.
"""

# Default percentile used for per-query thresholding
DEFAULT_PERCENTILE: float = 99.5

# Safety cap for input size upscaling (HxW)
MAX_INPUT_SIZE: int = 2048

# Subdirectories and filenames
EXPORT_SUBDIR: str = "iou_results"
MASK_DEFAULT_SUBDIR: str = "image/rot"
DECODER_DEFAULT_SUBDIR: str = "output/decoder"
CSV_MIOU_FILENAME: str = "mIoU_per_Query.csv"
