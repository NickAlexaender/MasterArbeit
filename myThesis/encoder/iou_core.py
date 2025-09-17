"""
Grundstruktur für IoU-Berechnung auf Basis der vorbereiteten Inputs aus
`myThesis/encoder/calculate_IoU.py`.

Eingabe (pro Eintrag):
- layer_idx: int
- image_idx: int
- feature_idx: int
- tokens: np.ndarray, Shape (N,) – Tokens für EIN Feature über alle Level
- shapes: Dict – enthält spatial_shapes [[H,W],...], level_start_index [...], input_size [H_in,W_in]
- mask_input: np.ndarray[bool], Shape (H_in, W_in)

Kernschritte:
- Tokens je Level mittels shapes (level_start_index, spatial_shapes) in Karten (H_i, W_i) rekonstruieren
- Auf Inputgröße skalieren
- Binarisieren (Schwellenwertstrategie konfigurierbar)
- IoU mit mask_input berechnen


Überlegung:
Gerade erstellen wir eine kombinierte_IoU. Pro-Level-IoU könnte allerdings auch interessant sein.

Pro-Level-IoU: sagt dir, auf welcher Auflösung/Skala eine Unit auf ein Konzept anspringt.

Kombiniert-IoU: sagt dir, ob die Unit insgesamt (über alle Skalen hinweg) zuverlässig ein Konzept abbildet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
	import cv2
except Exception:  # pragma: no cover
	cv2 = None


# -----------------------------
# Datenstrukturen
# -----------------------------

@dataclass
class IoUResult:
	layer_idx: int
	image_idx: int
	feature_idx: int
	level_idx: int
	map_shape: Tuple[int, int]
	threshold: float
	iou: float
	positives: int
	# Optional: die auf Inputgröße hochskalierte, kontinuierliche Heatmap (float32)
	# Dies entspricht genau der Karte, die anschließend binarisiert wird.
	heatmap: Optional[np.ndarray] = None


@dataclass
class IoUCombinedResult:
	"""Ergebnis für eine kombinierte Heatmap über alle Levels (auf Inputgröße)."""
	layer_idx: int
	image_idx: int
	feature_idx: int
	map_shape: Tuple[int, int]  # entspricht input_size (Hin, Win)
	threshold: float
	iou: float
	positives: int
	heatmap: Optional[np.ndarray] = None


# -----------------------------
# Helper: Eingabe entpacken
# -----------------------------

def unpack_iou_input(
	item: Union[
		Tuple[int, int, int, np.ndarray, Dict, np.ndarray],  # build_iou_core_input
		Any,  # direkte Struktur aus calculate_IoU (IoUInput)
	]
) -> Tuple[int, int, int, np.ndarray, Dict, np.ndarray]:
	"""Akzeptiert entweder das Tupel aus `build_iou_core_input` oder ein IoUInput-ähnliches Objekt.

	Verzichtet auf Modul-Importe und nutzt Attributprüfung (Duck-Typing),
	damit der Aufruf sowohl innerhalb eines Pakets als auch als Skript funktioniert.
	"""
	if isinstance(item, tuple) and len(item) == 6:
		return item  # already in canonical form

	# Duck-Typing: prüfe erforderliche Attribute
	required_attrs = ("layer_idx", "image_idx", "feature_idx", "tokens", "shapes", "mask_input")
	if all(hasattr(item, a) for a in required_attrs):
		return (
			int(getattr(item, "layer_idx")),
			int(getattr(item, "image_idx")),
			int(getattr(item, "feature_idx")),
			np.asarray(getattr(item, "tokens")),
			dict(getattr(item, "shapes")),
			np.asarray(getattr(item, "mask_input")),
		)
	raise TypeError("Unsupported input type for iou_core: expected 6-tuple or object with layer_idx,image_idx,feature_idx,tokens,shapes,mask_input")


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
		raise RuntimeError("OpenCV (cv2) benötigt für Resize.")
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
# Hauptfunktion pro Eintrag
# -----------------------------

def compute_iou_per_levels(
	item: Union[Tuple[int, int, int, np.ndarray, Dict, np.ndarray], Any],
	threshold_method: str = "percentile",
	threshold_value: float = 80.0,
	threshold_absolute: Optional[float] = None,
	return_heatmap: bool = False,
) -> List[IoUResult]:
	"""Berechnet IoU zwischen Featureaktivierung (pro Level) und maske (input_size).

	Rückgabe: Liste IoUResult – ein Eintrag pro Level.
	"""
	layer_idx, image_idx, feature_idx, tokens, shapes, mask_input = unpack_iou_input(item)
	maps = tokens_to_level_maps(tokens, shapes)
	Hin, Win = int(shapes["input_size"][0]), int(shapes["input_size"][1])

	results: List[IoUResult] = []
	for lidx, m in enumerate(maps):
		m_up = resize_to_input(m, (Hin, Win))
		bin_m, thr = binarize_map(m_up, method=threshold_method, value=threshold_value, absolute=threshold_absolute)
		iou = iou_from_masks(bin_m, mask_input)
		results.append(
			IoUResult(
				layer_idx=layer_idx,
				image_idx=image_idx,
				feature_idx=feature_idx,
				level_idx=lidx,
				map_shape=m.shape,
				threshold=thr,
				iou=iou,
				positives=int(bin_m.sum()),
				heatmap=(m_up if return_heatmap else None),
			)
		)
	return results


def compute_iou_combined(
	item: Union[Tuple[int, int, int, np.ndarray, Dict, np.ndarray], Any],
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
	layer_idx, image_idx, feature_idx, tokens, shapes, mask_input = unpack_iou_input(item)
	Hin, Win = int(shapes["input_size"][0]), int(shapes["input_size"][1])
	heatmap = combine_level_maps_to_input(tokens, shapes, combine=combine)
	bin_m, thr = binarize_map(heatmap, method=threshold_method, value=threshold_value, absolute=threshold_absolute)
	iou = iou_from_masks(bin_m, mask_input)
	return IoUCombinedResult(
		layer_idx=layer_idx,
		image_idx=image_idx,
		feature_idx=feature_idx,
		map_shape=(Hin, Win),
		threshold=thr,
		iou=iou,
		positives=int(bin_m.sum()),
		heatmap=(heatmap if return_heatmap else None),
	)


# -----------------------------
# Optional: kleiner Runner
# -----------------------------

def _main_preview(limit: int = 5) -> None:
	try:
		from .calculate_IoU import iter_iou_inputs
	except Exception as e:  # pragma: no cover
		print(f"Konnte iter_iou_inputs nicht importieren: {e}")
		return

	count = 0
	for item in iter_iou_inputs():
		res = compute_iou_per_levels(item, threshold_method="percentile", threshold_value=80)
		for r in res:
			print(
				f"Layer={r.layer_idx} Bild={r.image_idx} Feature={r.feature_idx} Level={r.level_idx} "
				f"map={r.map_shape} thr={r.threshold:.4f} IoU={r.iou:.4f} pos={r.positives}"
			)
		count += 1
		if count >= limit:
			break
	if count == 0:
		print("Keine Einträge gefunden. Bitte vorher calculate_IoU ausführen.")


if __name__ == "__main__":  # pragma: no cover
	_main_preview()


