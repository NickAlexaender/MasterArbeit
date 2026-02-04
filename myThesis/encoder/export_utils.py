from __future__ import annotations
import csv
import logging
import os
from typing import Dict, List, Optional
from .io_utils import _ensure_dir

logger = logging.getLogger(__name__)


def resolve_export_root(encoder_out_dir: str | None) -> str:
    if encoder_out_dir is None:
        from .io_utils import get_encoder_out_dir

        encoder_out_dir = get_encoder_out_dir()
    export_root = os.path.join(encoder_out_dir, "network_dissection")
    _ensure_dir(export_root)
    return export_root

# Wir erstellen eine CSV Datei mit den Ergebnissen 

def _write_network_dissection_csv(path: str, rows: List[Dict[str, object]]) -> None:
    """Schreibt Network Dissection mIoU-Ergebnisse in CSV."""
    if not rows:
        logger.info("Keine Zeilen zum Schreiben: %s", path)
        return
    fieldnames = [
        "layer_idx",
        "feature_idx",
        "miou",
        "nd_threshold",
        "n_images",
        "individual_ious",
        "active_ratio_mean",
        "overlay_dir",
    ]
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Wir brauchen etwas, was DAten an eine CSV anhängt

class CsvAppender:

    FIELDNAMES = [
        "layer_idx",
        "feature_idx",
        "miou",
        "nd_threshold",
        "n_images",
        "individual_ious",
        "active_ratio_mean",
        "overlay_dir",
    ]

    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        _ensure_dir(os.path.dirname(csv_path))
        self._ensure_header_written()

    def _ensure_header_written(self) -> None:
        if not os.path.isfile(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def append_row(self, row: Dict[str, object]) -> None:
        # Sicherstellen, dass nur bekannte Felder geschrieben werden (fehlende -> leere Strings)
        out: Dict[str, object] = {k: row.get(k, "") for k in self.FIELDNAMES}
        with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writerow(out)


__all__ = [
    "resolve_export_root",
    "_write_network_dissection_csv",
    "CsvAppender",
]
