from __future__ import annotations
import argparse
import logging
from typing import Optional

from .config import DEFAULT_PERCENTILE
from .pipeline import main_network_dissection_per_query as _pipeline_main
from .io_utils import decoder_out_dir as _decoder_out_dir
from .io_utils import mask_dir as _mask_dir
from .io_utils import resolve_export_root as _resolve_export_root

# Wir öffnen die CSV

def _ensure_cv2_available() -> None:
    try:
        import cv2  # type: ignore  # noqa: F401
    except Exception as e:
        raise RuntimeError(
        ) from e

# Und übertragen die Daten

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
    _ensure_cv2_available()

    _pipeline_main(
        percentile=args.percentile,
        mask_dir=args.mask_dir,
        decoder_out_dir=args.decoder_out_dir,
        export_root=args.export_root,
    )


if __name__ == "__main__":
    main()

# Wir bauen die Funktionen so auf, dass wir sie von außerhalb aufrufen können
# 
def main_network_dissection_per_query(
    percentile: float = DEFAULT_PERCENTILE,
    mask_dir: Optional[str] = None,
    decoder_out_dir: Optional[str] = None,
    export_root: Optional[str] = None,
) -> None:

    _pipeline_main(
        percentile=percentile,
        mask_dir=mask_dir,
        decoder_out_dir=decoder_out_dir,
        export_root=export_root,
    )
