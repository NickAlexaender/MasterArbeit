from __future__ import annotations

"""CLI entrypoint for the decoder pipeline.

This module only configures logging and parses CLI arguments, then delegates to
decoder.pipeline.main_network_dissection_per_query.
"""

import argparse
import logging
from typing import Optional

from .config import DEFAULT_PERCENTILE
from .pipeline import main_network_dissection_per_query as _pipeline_main
from .io_utils import decoder_out_dir as _decoder_out_dir
from .io_utils import mask_dir as _mask_dir
from .io_utils import resolve_export_root as _resolve_export_root


def _ensure_cv2_available() -> None:
    try:
        import cv2  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "OpenCV (cv2) is required. Install with 'pip install opencv-python'."
        ) from e


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Network Dissection (MaskDINO decoder)")
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help=(
            "Percentile for per-query thresholds. Accepts values in [0,100] or a fraction in (0,1],\n"
            "e.g., 90 or 0.90 both mean 'top 10%'."
        ),
    )
    parser.add_argument("--decoder-out-dir", type=str, default=None, help="Path to decoder output directory (contains layer*/ and pixel_embeddings/)")
    parser.add_argument("--mask-dir", type=str, default=None, help="Path to mask directory")
    parser.add_argument("--export-root", type=str, default=None, help="Export directory for results (defaults under decoder-out-dir)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    # Early check for cv2 to provide actionable error messages
    _ensure_cv2_available()

    _pipeline_main(
        percentile=args.percentile,
        mask_dir=args.mask_dir,
        decoder_out_dir=args.decoder_out_dir,
        export_root=args.export_root,
    )


if __name__ == "__main__":
    main()


# Backward-compatible API for in-repo callers importing this module
def main_network_dissection_per_query(
    percentile: float = DEFAULT_PERCENTILE,
    mask_dir: Optional[str] = None,
    decoder_out_dir: Optional[str] = None,
    export_root: Optional[str] = None,
) -> None:
    """Compatibility wrapper delegating to pipeline.main_network_dissection_per_query.

    Kept to support existing imports like:
    from myThesis.decoder import calculate_IoU_for_decoder
    calculate_IoU_for_decoder.main_network_dissection_per_query(...)
    """

    _pipeline_main(
        percentile=percentile,
        mask_dir=mask_dir,
        decoder_out_dir=decoder_out_dir,
        export_root=export_root,
    )
