
from __future__ import annotations
import csv
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from .config import get_project_root

logger = logging.getLogger(__name__)


def get_encoder_out_dir() -> str:
    return os.path.join(get_project_root(), "output", "encoder")


def get_mask_dir() -> str:
    return os.path.join(get_project_root(), "image", "rot")


def _encoder_out_dir() -> str:
    return get_encoder_out_dir()


def _mask_dir() -> str:
    return get_mask_dir()

# Suche alle Layer und ihre feature.csv auf um die Daten zu verwenden

def _find_layer_csvs(base_dir: Optional[str] = None) -> List[Tuple[int, str]]:
    base = base_dir or get_encoder_out_dir()
    if not os.path.isdir(base):
        logger.warning("Encoder-Output-Verzeichnis nicht gefunden: %s", base)
        return []
    layer_csvs: List[Tuple[int, str]] = []
    for name in os.listdir(base):
        if not name.startswith("layer"):
            continue
        m = re.match(r"layer(\d+)$", name)
        if not m:
            continue
        lidx = int(m.group(1))
        csv_path = os.path.join(base, name, "feature.csv")
        if os.path.isfile(csv_path):
            layer_csvs.append((lidx, csv_path))
    layer_csvs.sort(key=lambda x: x[0])
    return layer_csvs

# Wir laden alle shapes.json Dateien.

def _load_all_shapes(base_dir: Optional[str] = None) -> Dict[str, Dict]:
    base = base_dir or get_encoder_out_dir()
    out: Dict[str, Dict] = {}
    if not os.path.isdir(base):
        logger.warning("Encoder-Output-Verzeichnis nicht gefunden: %s", base)
        return out
    for name in os.listdir(base):
        # shapes liegen in Ordnern, die NICHT mit "layer" beginnen
        if name.startswith("layer"):
            continue
        shapes_path = os.path.join(base, name, "shapes.json")
        if os.path.isfile(shapes_path):
            try:
                with open(shapes_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                out[name] = data
            except Exception as e:  # pragma: no cover - robust gegen kaputte Dateien
                logger.warning("Konnte shapes.json nicht lesen (%s): %s", shapes_path, e)
    return out

# Wichtig ist, dass wir die richtige shapes.json pro ID des Bildes zuordnen.

def _select_shapes_for(image_id: str, n_tokens: int, all_shapes: Dict[str, Dict]) -> Optional[Dict]:
    if not all_shapes:
        return None

    # 1) Direkter Match über Ordnernamen
    if image_id in all_shapes:
        return all_shapes[image_id]

    # 2) Match über image_id in der shapes.json selbst (falls Ordnername abweicht)
    for _, shapes_data in all_shapes.items():
        if shapes_data.get("image_id") == image_id:
            return shapes_data

    # 3) Match über N_tokens als Fallback
    matches = [v for v in all_shapes.values() if int(v.get("N_tokens", -1)) == int(n_tokens)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Mehrere Matches mit gleicher Token-Anzahl – deterministisch sortieren
        matches_with_id = [(v.get("image_id", ""), v) for v in matches]
        matches_with_id.sort(key=lambda x: x[0])
        return matches_with_id[0][1]

    # 4) Keine Übereinstimmung gefunden
    logger.warning("Keine passende shapes.json für image_id='%s' gefunden (N_tokens=%s)", image_id, n_tokens)
    return None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


__all__ = [
    "get_encoder_out_dir",
    "get_mask_dir",
    "_encoder_out_dir",
    "_mask_dir",
    "_find_layer_csvs",
    "_load_all_shapes",
    "_select_shapes_for",
    "_ensure_dir",
]
