"""
Zentrale Konfiguration und Settings für die Encoder-Pipeline.

- Beinhaltet Konstanten (z. B. NETWORK_DISSECTION_PERCENTILE)
- Einfache Helper wie get_project_root()
- Standard-Logging-Level und Default-Combine-Strategie
"""

from __future__ import annotations

import logging
import os
from typing import Literal

# Network Dissection: Per-Feature Threshold (0-100 für Perzentil)
NETWORK_DISSECTION_PERCENTILE: float = 90.0

# Kombinationsmethode für Multi-Level-Heatmaps
DEFAULT_COMBINE: Literal["max", "sum", "mean"] = "max"

# Standard-Logging-Level
LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


def get_project_root() -> str:
    """Liefert das Root-Verzeichnis des Projekts `myThesis`.

    Annahme: Dieses File liegt in `myThesis/encoder/` -> eine Ebene hoch ist `myThesis`.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def configure_logging(level: str | int = LOG_LEVEL) -> None:
    """Konfiguriert das Root-Logging falls noch nicht geschehen.

    Kann mehrfach aufgerufen werden; nur der erste Aufruf hat Effekt.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


__all__ = [
    "NETWORK_DISSECTION_PERCENTILE",
    "DEFAULT_COMBINE",
    "LOG_LEVEL",
    "get_project_root",
    "configure_logging",
]
