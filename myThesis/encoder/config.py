from __future__ import annotations

import logging
import os
from typing import Literal

# Standardwert in Network Dissection-Paper
NETWORK_DISSECTION_PERCENTILE: float = 99.5

DEFAULT_COMBINE: Literal["max", "sum", "mean"] = "max"

# Welche Fehler werden geloggt
LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


def get_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def configure_logging(level: str | int = LOG_LEVEL) -> None:
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
