from __future__ import annotations

import csv
import json
import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .config import (
    DECODER_DEFAULT_SUBDIR,
    MASK_DEFAULT_SUBDIR,
    EXPORT_SUBDIR,
)

logger = logging.getLogger(__name__)


# neue Regex für den Namen in CSV
NAME_RE = re.compile(r"^(.+),\s*Query\s*(\d+)$")

# Erst wird die root des Projekts bestimmt 

def project_root() -> str:

    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Anschließend wird bestimmt, wo der Output des Decoders liegt

def decoder_out_dir(base: Optional[str] = None) -> str:
    if base:
        return base
    return os.path.join(project_root(), DECODER_DEFAULT_SUBDIR)

# DAs gleiche für die Masken

def mask_dir(base: Optional[str] = None) -> str:

    if base:
        return base
    return os.path.join(project_root(), MASK_DEFAULT_SUBDIR)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def resolve_export_root(export_root: Optional[str], base_decoder_out: Optional[str]) -> str:
    if export_root:
        return export_root
    base = decoder_out_dir(base_decoder_out)
    return os.path.join(base, EXPORT_SUBDIR)

# Wir formen für jede Zeile ein Paket mit image_id, query_idx und query_features
# SPeichersparend
def iter_csv_rows(csv_path: str) -> Iterator[Tuple[str, int, np.ndarray]]:

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # header
            for row in reader:
                if not row:
                    continue
                name = row[0].strip().strip('"')
                m = NAME_RE.match(name)
                if not m:
                    logger.warning("CSV row skipped due to name mismatch: %s", name)
                    continue
                image_id = m.group(1).strip()
                try:
                    query_idx = int(m.group(2))
                except ValueError:
                    logger.warning("CSV row skipped due to non-integer query index: %s", row[0])
                    continue

                try:
                    values = [float(x) for x in row[1:]]
                except ValueError:
                    logger.warning("CSV row skipped due to invalid floats: %s", row)
                    continue

                query_features = np.asarray(values, dtype=np.float32)
                yield image_id, query_idx, query_features
    except FileNotFoundError:
        logger.error("CSV file not found: %s", csv_path)
    except Exception:
        logger.exception("Error while reading CSV: %s", csv_path)

# Nun lesen wir die JSON aus

def read_json(path: str) -> Dict[str, Any]:

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("JSON file not found: %s", path)
    except json.JSONDecodeError as e:
        logger.warning("Invalid JSON (%s): %s", e, path)
    except Exception:
        logger.exception("Error reading JSON: %s", path)
    return {}
