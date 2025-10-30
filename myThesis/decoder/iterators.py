from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from .io_utils import decoder_out_dir as _default_decoder_out_dir
from .io_utils import iter_csv_rows, read_json
from .mask_utils import (
    get_input_size_from_embedding,
    validate_input_size,
    load_mask_for_image,
)
from .models import DecoderIoUInput

logger = logging.getLogger(__name__)


def find_layer_csvs(base_dir: Optional[str] = None) -> List[Tuple[int, str]]:
    """Find all layer*/Query.csv files and extract layer indices.

    Returns list of (layer_idx, csv_path) sorted by layer.
    """

    base = _default_decoder_out_dir(base_dir)
    if not os.path.isdir(base):
        logger.warning("Decoder output dir not found: %s", base)
        return []

    results: List[Tuple[int, str]] = []
    for name in os.listdir(base):
        if not name.startswith("layer"):
            continue
        m = re.match(r"layer(\d+)$", name)
        if not m:
            continue
        lidx = int(m.group(1))
        csv_path = os.path.join(base, name, "Query.csv")
        if os.path.isfile(csv_path):
            results.append((lidx, csv_path))

    results.sort(key=lambda x: x[0])
    return results


def load_all_pixel_embeddings(base_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Load all available metadata and pixel embeddings.

    Returns: {embedding_id: {"metadata": dict, "embedding": np.ndarray}}
    """

    root = _default_decoder_out_dir(base_dir)
    pixel_embed_dir = os.path.join(root, "pixel_embeddings")
    out: Dict[str, Dict[str, Any]] = {}

    if not os.path.isdir(pixel_embed_dir):
        logger.warning("Pixel embeddings dir not found: %s", pixel_embed_dir)
        return out

    metadata_files = [
        f for f in os.listdir(pixel_embed_dir) if f.startswith("metadata_") and f.endswith(".json")
    ]

    for metadata_file in metadata_files:
        metadata_path = os.path.join(pixel_embed_dir, metadata_file)
        metadata = read_json(metadata_path)
        if not metadata:
            continue

        npy_file = metadata.get("npy_file")
        if not npy_file:
            logger.warning("Missing 'npy_file' in metadata: %s", metadata_path)
            continue
        npy_path = os.path.join(pixel_embed_dir, npy_file)
        if not os.path.isfile(npy_path):
            logger.warning("Missing npy file: %s", npy_path)
            continue

        try:
            embedding = np.load(npy_path)
        except Exception:
            logger.exception("Failed to load npy: %s", npy_path)
            continue

        embedding_id = metadata.get("embedding_id")
        if not embedding_id:
            image_id = metadata.get("image_id")
            if image_id:
                embedding_id = f"embed_{image_id}"
            else:
                embedding_id = os.path.splitext(metadata_file)[0].replace("metadata_", "embed_")

        out[embedding_id] = {"metadata": metadata, "embedding": embedding}

    return out


def select_pixel_embedding_for(image_id: str, all_embeddings: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick pixel embedding for a given image id.

    Strategy:
    1) Exact match on key = f"embed_{image_id}"
    2) Scan metadata.image_id
    """

    if not all_embeddings:
        return None

    expected = f"embed_{image_id}"
    if expected in all_embeddings:
        return all_embeddings[expected]

    for _, data in all_embeddings.items():
        if data.get("metadata", {}).get("image_id") == image_id:
            return data

    logger.warning("No pixel embedding found for image_id='%s'", image_id)
    return None


def iter_decoder_iou_inputs(
    decoder_out_dir: Optional[str] = None,
    mask_dir: Optional[str] = None,
) -> Iterator[DecoderIoUInput]:
    """Stream DecoderIoUInput per row in layer*/Query.csv files.

    Missing pixel embeddings or masks are logged and skipped.
    """

    layer_csvs = find_layer_csvs(decoder_out_dir)
    all_embeddings = load_all_pixel_embeddings(decoder_out_dir)

    # cache mask per (image_id, input_size)
    mask_cache: Dict[Tuple[str, Tuple[int, int]], np.ndarray] = {}

    for lidx, csv_path in layer_csvs:
        for image_id, query_idx, query_features in iter_csv_rows(csv_path):
            embedding_data = select_pixel_embedding_for(image_id, all_embeddings)
            if embedding_data is None:
                logger.warning("Skipping image '%s' due to missing embedding", image_id)
                continue

            metadata = embedding_data["metadata"]
            input_size = get_input_size_from_embedding(metadata)
            input_size = validate_input_size(input_size, metadata)

            key = (image_id, input_size)
            if key in mask_cache:
                mask_input = mask_cache[key]
            else:
                try:
                    mask_input = load_mask_for_image(image_id, input_size, mask_dir=mask_dir)
                    mask_cache[key] = mask_input
                except FileNotFoundError as e:
                    logger.warning("%s Skipping '%s'.", e, image_id)
                    continue

            yield DecoderIoUInput(
                layer_idx=lidx,
                image_id=image_id,
                query_idx=query_idx,
                query_features=query_features,
                pixel_embedding=embedding_data["embedding"],
                input_size=input_size,
                mask_input=mask_input,
            )
