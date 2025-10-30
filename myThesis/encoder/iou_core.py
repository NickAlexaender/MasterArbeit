"""
Rein numerische Kernlogik für Heatmaps, Binarisierung und IoU.

Dieses Modul enthält keinerlei Datei-/Pfad- oder OpenCV-spezifische Logik außer dem Resize.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
	import cv2
except Exception:  # pragma: no cover
	cv2 = None

from .models import IoUCombinedResult


logger = logging.getLogger(__name__)


# -----------------------------
# Tokens -> Karten je Level
# -----------------------------

def tokens_to_level_maps(tokens: np.ndarray, shapes: Dict) -> List[np.ndarray]:
	"""Zerlegt das 1D-Token-Array (N,) in Level-Karten gemäß shapes.

	Rückgabe: Liste von 2D-Arrays [H_i, W_i] in float32.
	"""
	spatial_shapes = shapes.get("spatial_shapes")  # [[H,W], ...]
	level_start_index = shapes.get("level_start_index")  # [s0, s1, ...]
	if not isinstance(spatial_shapes, (list, tuple)) or not isinstance(level_start_index, (list, tuple)):
		raise ValueError("shapes muss 'spatial_shapes' und 'level_start_index' enthalten")

	tokens = np.asarray(tokens, dtype=np.float32)
	maps: List[np.ndarray] = []
	for i, (hw, s) in enumerate(zip(spatial_shapes, level_start_index)):
		H, W = int(hw[0]), int(hw[1])
		n_i = H * W
		s = int(s)
		e = s + n_i
		if e > tokens.size:
			raise ValueError(f"Tokens zu kurz für Level {i}: slice {s}:{e}, N={tokens.size}")
		m = tokens[s:e].reshape(H, W)
		maps.append(m)
	return maps


def resize_to_input(m: np.ndarray, input_size: Sequence[int]) -> np.ndarray:
	"""Skaliert Karte m (H,W) auf (H_in,W_in) als float32."""
	if cv2 is None:
		raise RuntimeError("OpenCV (cv2) wird für Resize benötigt. Bitte 'opencv-python' (oder -headless) installieren.")
	Hin, Win = int(input_size[0]), int(input_size[1])
	return cv2.resize(m.astype(np.float32), (Win, Hin), interpolation=cv2.INTER_LINEAR)


def combine_level_maps_to_input(
	tokens: np.ndarray,
	shapes: Dict,
	combine: str = "max",
) -> np.ndarray:
	"""Kombiniert alle Level-Karten zu EINER Heatmap in Inputgröße.

	combine:
	- 'max'  : pixelweises Maximum
	- 'sum'  : Summe
	- 'mean' : Mittelwert
	"""
	maps = tokens_to_level_maps(tokens, shapes)
	Hin, Win = int(shapes["input_size"][0]), int(shapes["input_size"][1])
	upsampled = [resize_to_input(m, (Hin, Win)) for m in maps]
	if not upsampled:
		return np.zeros((Hin, Win), dtype=np.float32)
	stack = np.stack(upsampled, axis=0).astype(np.float32)
	if combine == "sum":
		combined = stack.sum(axis=0)
	elif combine == "mean":
		combined = stack.mean(axis=0)
	else:  # default: max
		combined = stack.max(axis=0)
	return combined.astype(np.float32, copy=False)


def unpack_iou_input(
	item: Union[
		Tuple[int, str, int, np.ndarray, Dict, np.ndarray],  # build_iou_core_input
		Any,  # direkte Struktur aus calculate_IoU (IoUInput)
	]
) -> Tuple[int, str, int, np.ndarray, Dict, np.ndarray]:
	"""Akzeptiert entweder das Tupel aus `build_iou_core_input` oder ein IoUInput-ähnliches Objekt.

	Verzichtet auf Modul-Importe und nutzt Attributprüfung (Duck-Typing),
	damit der Aufruf sowohl innerhalb eines Pakets als auch als Skript funktioniert.
	"""
	if isinstance(item, tuple) and len(item) == 6:
		return item  # already in canonical form

	# Duck-Typing: prüfe erforderliche Attribute
	required_attrs = ("layer_idx", "image_id", "feature_idx", "tokens", "shapes", "mask_input")
	if all(hasattr(item, a) for a in required_attrs):
		return (
			int(getattr(item, "layer_idx")),
			str(getattr(item, "image_id")),
			int(getattr(item, "feature_idx")),
			np.asarray(getattr(item, "tokens")),
			dict(getattr(item, "shapes")),
			np.asarray(getattr(item, "mask_input")),
		)
	raise TypeError("Unsupported input type for iou_core: expected 6-tuple or object with layer_idx,image_id,feature_idx,tokens,shapes,mask_input")


def binarize_map(
	m: np.ndarray,
	method: str = "percentile",
	value: float = 80.0,  # niedrigeres Perzentil => mehr Pixel werden True
	absolute: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
	"""Binarisiert eine Aktivierungskarte.

	Methoden:
	- absolute: direktes Threshold
	- percentile: Schwelle = np.percentile(m, value)
	- mean_std: Schwelle = mean + value * std (value = k)
	Rückgabe: (bin_mask(bool), schwelle)
	"""
	m = m.astype(np.float32, copy=False)
	if absolute is not None:
		thr = float(absolute)
	elif method == "mean_std":
		mu = float(m.mean())
		sd = float(m.std())
		thr = mu + float(value) * sd
	else:  # percentile
		thr = float(np.percentile(m, float(value)))
	bin_mask = m >= thr
	return bin_mask, thr


def iou_from_masks(a: np.ndarray, b: np.ndarray) -> float:
	"""Berechnet IoU zweier bool-Maps gleicher Größe."""
	if a.shape != b.shape:
		raise ValueError("Masken müssen gleiche Shape haben")
	a = a.astype(bool)
	b = b.astype(bool)
	inter = np.logical_and(a, b).sum(dtype=np.int64)
	union = np.logical_or(a, b).sum(dtype=np.int64)
	if union == 0:
		return 0.0
	return float(inter) / float(union)


# -----------------------------
# Hauptfunktionen
# -----------------------------

def compute_iou_combined(
	item: Union[Tuple[int, str, int, np.ndarray, Dict, np.ndarray], Any],
	threshold_method: str = "percentile",
	threshold_value: float = 80.0,
	threshold_absolute: Optional[float] = None,
	combine: str = "max",
	return_heatmap: bool = True,
) -> IoUCombinedResult:
	"""Berechnet eine kombinierte Heatmap (über alle Levels) in Inputgröße und deren IoU.

	- Kombiniert erst alle Level (max/sum/mean) auf Inputgröße.
	- Binarisiert die kombinierte Heatmap gemäß Threshold-Strategie.
	- Berechnet die IoU gegen mask_input.
	"""
	layer_idx, image_id, feature_idx, tokens, shapes, mask_input = unpack_iou_input(item)
	Hin, Win = int(shapes["input_size"][0]), int(shapes["input_size"][1])
	heatmap = combine_level_maps_to_input(tokens, shapes, combine=combine)
	bin_m, thr = binarize_map(heatmap, method=threshold_method, value=threshold_value, absolute=threshold_absolute)
	iou = iou_from_masks(bin_m, mask_input)
	return IoUCombinedResult(
		layer_idx=layer_idx,
		image_id=image_id,
		feature_idx=feature_idx,
		map_shape=(Hin, Win),
		threshold=thr,
		iou=iou,
		positives=int(bin_m.sum()),
		heatmap=(heatmap if return_heatmap else None),
	)


# -----------------------------
# Neue Funktionen für globale Binarisierung
# -----------------------------

def generate_heatmap_only(
	item: Union[Tuple[int, str, int, np.ndarray, Dict, np.ndarray], Any],
	combine: str = "max",
) -> np.ndarray:
	"""Erzeugt ausschließlich die kontinuierliche Heatmap (OHNE Binarisierung/IoU).

	Verwendungszweck: Sammeln aller Heatmaps eines Features für globale Binarisierung.

	Args:
		item: IoUInput-Tupel oder -Objekt
		combine: Kombinationsmethode für Multi-Level ('max'|'sum'|'mean')

	Returns:
		np.ndarray: Kontinuierliche Heatmap in Inputgröße (H_in, W_in), dtype=float32
	"""
	layer_idx, image_id, feature_idx, tokens, shapes, mask_input = unpack_iou_input(item)
	heatmap = combine_level_maps_to_input(tokens, shapes, combine=combine)
	return heatmap


def compute_iou_from_heatmap(
	heatmap: np.ndarray,
	mask_input: np.ndarray,
	threshold: float,
) -> float:
	"""Berechnet IoU zwischen binarisierter Heatmap und Ground-Truth-Maske.

	Args:
		heatmap: Kontinuierliche Aktivierungskarte (H, W) float32
		mask_input: Binäre Ground-Truth-Maske (H, W) bool
		threshold: Absoluter Schwellenwert für Binarisierung

	Returns:
		float: IoU-Wert [0, 1]
	"""
	if heatmap.shape != mask_input.shape:
		raise ValueError(f"Shape-Mismatch: Heatmap {heatmap.shape} vs. Maske {mask_input.shape}")
	bin_heatmap = heatmap >= threshold
	return iou_from_masks(bin_heatmap, mask_input)


