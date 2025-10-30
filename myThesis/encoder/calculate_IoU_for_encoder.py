"""Ausführungsdatei für die Network-Dissection-Pipeline.

Hält nur Logging-Setup und einen klaren Funktionsaufruf, der die Pipeline startet.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import LOG_LEVEL, configure_logging
from .pipeline import main_export_network_dissection as run_network_dissection


def main(
    percentile: Optional[float] = None,
    top_percent_active: Optional[float] = None,
    mask_dir: Optional[str] = None,
    encoder_out_dir: Optional[str] = None,
    export_root: Optional[str] = None,
    export_mode: str = "per-layer-best",
) -> None:
    """Startet die Network-Dissection-Pipeline mit optionalen Parametern."""
    configure_logging(LOG_LEVEL)
    logging.getLogger(__name__).info("Starte Network Dissection Pipeline…")
    run_network_dissection(
        percentile=percentile,
        top_percent_active=top_percent_active,
        mask_dir=mask_dir,
        encoder_out_dir=encoder_out_dir,
        export_root=export_root,
        export_mode=export_mode,
    )


if __name__ == "__main__":
    main()




