"""
Vorstufe für iou_core.py:
- iteriert über output/encoder/layer*/feature.csv
- extrahiert Layer-, Bild- und Feature-Index sowie Tokens (pro Zeile)
- lädt passende Bild-Infos aus output/encoder/<image_id>/shapes.json
- bereitet die Maske aus myThesis/image/colours/rot.png in input-Größe vor

Ergebnis: Generator, der iou_core mit allen benötigten Inputs versorgt.
"""

from __future__ import annotations

import os
import re
import csv
import json
from dataclasses import dataclass
from typing import Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np

try:
	import cv2  # Für Bild-/Maskenverarbeitung
except Exception:  # pragma: no cover
	cv2 = None


# -----------------------------
# Datenstruktur für iou_core.py
# -----------------------------

@dataclass
class IoUInput:
	layer_idx: int
	image_idx: int  # 1-basiert gemäß CSV-Erzeugung
	feature_idx: int  # 1-basiert gemäß CSV-Erzeugung
	tokens: np.ndarray  # Shape: (N,) float32
	shapes: Dict  # shapes.json-Inhalt
	mask_input: np.ndarray  # bool, Shape: (H_in, W_in)


# -----------------------------
# Hilfsfunktionen Pfade/IO
# -----------------------------

def _project_root() -> str:
	# Dieses File liegt in myThesis/encoder -> eine Ebene hoch ist myThesis
	return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _encoder_out_dir() -> str:
	return os.path.join(_project_root(), "output", "encoder")


def _mask_path() -> str:
	return os.path.join(_project_root(), "image", "colours", "rot.png")


def _find_layer_csvs() -> List[Tuple[int, str]]:
	base = _encoder_out_dir()
	if not os.path.isdir(base):
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


def _load_all_shapes() -> Dict[str, Dict]:
	"""Lädt alle verfügbaren shapes.json unter output/encoder/<image_id>/shapes.json.

	Rückgabe: { image_id: payload_dict }
	"""
	base = _encoder_out_dir()
	out: Dict[str, Dict] = {}
	if not os.path.isdir(base):
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
			except Exception:
				pass
	return out


def _select_shapes_for(image_idx: int, n_tokens: int, all_shapes: Dict[str, Dict]) -> Optional[Dict]:
	"""Wählt passende shapes.json für den gegebenen Bild-Index.

	Strategie:
	1) Wenn es exakt eine shapes.json gibt -> nimm diese.
	2) Versuche Match über N_tokens.
	3) Fallback: alphabetisch sortieren und 1-basiert zuordnen.
	"""
	if not all_shapes:
		return None
	if len(all_shapes) == 1:
		return next(iter(all_shapes.values()))

	# 2) Match über N_tokens
	matches = [v for v in all_shapes.values() if int(v.get("N_tokens", -1)) == int(n_tokens)]
	if len(matches) == 1:
		return matches[0]
	if len(matches) > 1:
		# Nicht eindeutig – wähle deterministisch erstes nach image_id
		items = sorted(((k, v) for k, v in all_shapes.items() if v in matches), key=lambda kv: kv[0])
		return items[0][1]

	# 3) Fallback: 1-basierte Reihenfolge über alphabetische Sortierung
	items_sorted = sorted(all_shapes.items(), key=lambda kv: kv[0])
	idx0 = max(0, min(len(items_sorted) - 1, image_idx - 1))
	return items_sorted[idx0][1]


# -----------------------------
# Maskenaufbereitung (nur input-Größe benötigt)
# -----------------------------

def _prepare_mask_binary(mask_bgr: np.ndarray) -> np.ndarray:
	"""Wandelt farbige Maske in binäre (bool) um. Erwartet BGR oder RGB.

	Heuristik für 'rot': R hoch, G/B niedrig. Fallback: alles != schwarz.
	Rückgabe: bool-Array [H, W].
	"""
	if mask_bgr.ndim == 2:
		# Bereits Graustufen -> threshold > 0
		return (mask_bgr > 0)

	# Falls Bild in RGB statt BGR vorliegt, spielt es für die Heuristik kaum Rolle
	b = mask_bgr[..., 0].astype(np.int16)
	g = mask_bgr[..., 1].astype(np.int16)
	r = mask_bgr[..., 2].astype(np.int16)

	red_dominant = (r > 150) & (g < 100) & (b < 100)
	if np.count_nonzero(red_dominant) == 0:
		# Fallback: alle nicht-schwarzen Pixel
		non_black = (r > 0) | (g > 0) | (b > 0)
		return non_black
	return red_dominant

def _load_mask_for_input(input_size: Tuple[int, int]) -> np.ndarray:
	"""Lädt die Maske und liefert sie als bool-Array in input-Größe (H_in, W_in)."""
	mask_file = _mask_path()
	if cv2 is None:
		raise RuntimeError("OpenCV (cv2) wird benötigt, ist aber nicht verfügbar.")
	m = cv2.imread(mask_file, cv2.IMREAD_COLOR)
	if m is None:
		raise FileNotFoundError(f"Maske nicht gefunden: {mask_file}")

	# In bool umwandeln und direkt auf input-Größe skalieren (nearest für binär)
	mask_bin = _prepare_mask_binary(m).astype(np.uint8)
	Hin, Win = int(input_size[0]), int(input_size[1])
	mask_input = cv2.resize(mask_bin, (Win, Hin), interpolation=cv2.INTER_NEAREST).astype(bool)
	return mask_input


# -----------------------------
# CSV-Iteration und Paketierung
# -----------------------------

_NAME_RE = re.compile(r"^Bild(\d+),\s*Feature(\d+)$")


def _iter_csv_rows(csv_path: str) -> Iterable[Tuple[int, int, np.ndarray]]:
	"""Iteriert Zeilen einer feature.csv und liefert (image_idx, feature_idx, tokens).

	tokens ist ein 1D np.ndarray[N] float32.
	"""
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.reader(f)
		header = next(reader, None)
		for row in reader:
			if not row:
				continue
			name = row[0].strip()
			m = _NAME_RE.match(name)
			if not m:
				continue
			img_idx = int(m.group(1))
			feat_idx = int(m.group(2))
			try:
				values = [float(x) for x in row[1:]]
			except ValueError:
				# Überspringe fehlerhafte Zeilen
				continue
			tokens = np.asarray(values, dtype=np.float32)
			yield img_idx, feat_idx, tokens


def iter_iou_inputs() -> Generator[IoUInput, None, None]:
	"""Haupt-Iterator: liefert pro CSV-Zeile ein IoUInput-Paket.
	- Erkennt Layer-Index aus Ordnernamen.
	- Mappt Bild-Index auf passende shapes.json.
	- Bereitet Maske für input-Size vor (gecacht pro input_size).
	"""
	layer_csvs = _find_layer_csvs()
	all_shapes = _load_all_shapes()
	# Cache: Maske je input_size
	mask_cache_input: Dict[Tuple[int, int], np.ndarray] = {}

	for lidx, csv_path in layer_csvs:
		for img_idx, feat_idx, tokens in _iter_csv_rows(csv_path):
			shapes = _select_shapes_for(img_idx, tokens.size, all_shapes)
			if shapes is None:
				# Ohne Shapes ist Reassemblierung in iou_core schwierig – skippe
				continue
			Hin, Win = int(shapes["input_size"][0]), int(shapes["input_size"][1])

			# Maske beschaffen (gecacht nach input-Größe)
			key_in = (Hin, Win)
			if key_in in mask_cache_input:
				mask_input = mask_cache_input[key_in]
			else:
				mask_input = _load_mask_for_input((Hin, Win))
				mask_cache_input[key_in] = mask_input

			yield IoUInput(
				layer_idx=lidx,
				image_idx=img_idx,
				feature_idx=feat_idx,
				tokens=tokens,
				shapes=shapes,
				mask_input=mask_input,
			)


# -----------------------------
# Grundstruktur + Ausgabe
# -----------------------------

def build_iou_core_input(item: IoUInput) -> Tuple[int, int, int, np.ndarray, Dict, np.ndarray]:
	"""Formt IoUInput in das erwartete Tupel für iou_core.py um.

	Rückgabe: (layer_idx, image_idx, feature_idx, tokens, shapes, mask_input)
	"""
	return (
		item.layer_idx,
		item.image_idx,
		item.feature_idx,
		item.tokens,
		item.shapes,
		item.mask_input,
	)


def _ensure_dir(p: str) -> None:
	os.makedirs(p, exist_ok=True)


def _export_root() -> str:
	# Sammelordner für kombinierte IoUs & Heatmaps
	return os.path.join(_encoder_out_dir(), "iou_combined")


def _write_csv(path: str, rows: List[Dict[str, object]]) -> None:
	if not rows:
		return
	# Bestimme Spalten aus Keys des ersten Eintrags
	fieldnames = [
		"layer_idx",
		"image_idx",
		"feature_idx",
		"iou",
		"threshold",
		"positives",
		"heatmap_path",
	"overlay_path",
	]
	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def _save_heatmap_png(dest_path: str, heatmap: np.ndarray) -> None:
	"""Speichert eine float32-Heatmap als PNG (0-255 skaliert)."""
	if cv2 is None:
		raise RuntimeError("OpenCV (cv2) wird benötigt, ist aber nicht verfügbar.")
	h = heatmap.astype(np.float32, copy=False)
	vmin = float(h.min())
	vmax = float(h.max())
	if vmax <= vmin + 1e-8:
		img = np.zeros_like(h, dtype=np.uint8)
	else:
		hn = (h - vmin) / (vmax - vmin)
		img = np.clip(hn * 255.0, 0, 255).astype(np.uint8)
	_ensure_dir(os.path.dirname(dest_path))
	cv2.imwrite(dest_path, img)


def _save_overlay_comparison(dest_path: str, mask_input: np.ndarray, heatmap: np.ndarray, threshold: float) -> None:
	"""Erstellt und speichert Vergleichsbild:
	- Blau  (BGR=255,0,0): Überschneidung (Maske ∧ Binär-Heatmap)
	- Rot   (BGR=0,0,255): Maske nur
	- Gelb  (BGR=0,255,255): Binär-Heatmap nur
	- Schwarz: Rest
	"""
	if cv2 is None:
		raise RuntimeError("OpenCV (cv2) wird benötigt, ist aber nicht verfügbar.")
	mask = mask_input.astype(bool)
	bin_hm = (heatmap.astype(np.float32) >= float(threshold))

	inter = np.logical_and(mask, bin_hm)
	mask_only = np.logical_and(mask, np.logical_not(bin_hm))
	hm_only = np.logical_and(bin_hm, np.logical_not(mask))

	H, W = mask.shape
	img = np.zeros((H, W, 3), dtype=np.uint8)  # BGR
	# Blau für Überschneidung
	img[inter, 0] = 255  # B
	# Rot für Maske-only
	img[mask_only, 2] = 255  # R
	# Gelb für Heatmap-only (R+G)
	img[hm_only, 1] = 255  # G
	img[hm_only, 2] = 255  # R

	_ensure_dir(os.path.dirname(dest_path))
	cv2.imwrite(dest_path, img)


def main_export_combined() -> None:
	"""Erzeugt pro Layer eine sortierte CSV aller kombinierten IoUs und speichert Heatmaps als PNG.

	Ausgabe-Struktur:
	- myThesis/output/encoder/iou_combined/layer<L>/iou_sorted.csv
	- myThesis/output/encoder/iou_combined/layer<L>/heatmaps/Bild<I>_Feature<F>.png
	"""
	# Lazy-Import von iou_core, kompatibel für Modul & Skript
	try:
		from .iou_core import compute_iou_combined  # type: ignore
	except Exception:
		import os as _os, sys as _sys
		_sys.path.append(_os.path.dirname(__file__))
		from iou_core import compute_iou_combined  # type: ignore

	export_root = _export_root()
	_ensure_dir(export_root)

	# Sammle Ergebnisse pro Layer
	per_layer: Dict[int, List[Dict[str, object]]] = {}
	# Tracking der besten(n) Einträge pro Layer (inkl. Arrays für Overlay)
	per_layer_best: Dict[int, Dict[str, object]] = {}

	count = 0
	for item in iter_iou_inputs():
		res = compute_iou_combined(
			item,
			threshold_method="percentile",
			threshold_value=80.0,
			threshold_absolute=None,
			combine="max",
			return_heatmap=True,
		)

		layer_dir = os.path.join(export_root, f"layer{res.layer_idx}")
		heat_dir = os.path.join(layer_dir, "heatmaps")
		_ensure_dir(heat_dir)
		heat_name = f"Bild{res.image_idx}_Feature{res.feature_idx}.png"
		heat_path = os.path.join(heat_dir, heat_name)

		if res.heatmap is not None:
			_save_heatmap_png(heat_path, res.heatmap)
		else:
			heat_path = ""  # sollte nicht passieren, return_heatmap=True

		row = {
			"layer_idx": res.layer_idx,
			"image_idx": res.image_idx,
			"feature_idx": res.feature_idx,
			"iou": float(res.iou),
			"threshold": float(res.threshold),
			"positives": int(res.positives),
			"heatmap_path": os.path.relpath(heat_path, start=export_root) if heat_path else "",
			"overlay_path": "",
		}
		per_layer.setdefault(res.layer_idx, []).append(row)

		# Bestleistung pro Layer aktualisieren (mit Toleranz für Ties)
		best = per_layer_best.get(res.layer_idx)
		if best is None:
			per_layer_best[res.layer_idx] = {
				"best_iou": float(res.iou),
				"items": [
					{
						"image_idx": res.image_idx,
						"feature_idx": res.feature_idx,
						"threshold": float(res.threshold),
						"heatmap": res.heatmap,
						"mask_input": item.mask_input,
					}
				],
			}
		else:
			cur_best = float(best["best_iou"])  # type: ignore
			if float(res.iou) > cur_best + 1e-12:
				best["best_iou"] = float(res.iou)
				best["items"] = [
					{
						"image_idx": res.image_idx,
						"feature_idx": res.feature_idx,
						"threshold": float(res.threshold),
						"heatmap": res.heatmap,
						"mask_input": item.mask_input,
					}
				]
			elif abs(float(res.iou) - cur_best) <= 1e-12:
				best.setdefault("items", []).append(
					{
						"image_idx": res.image_idx,
						"feature_idx": res.feature_idx,
						"threshold": float(res.threshold),
						"heatmap": res.heatmap,
						"mask_input": item.mask_input,
					}
				)

		count += 1

	# Erzeuge Overlays der besten Features und schreibe pro Layer CSV, sortiert nach IoU absteigend
	for lidx, rows in per_layer.items():
		# Overlays für beste(n) Eintrag/Einträge
		best = per_layer_best.get(lidx)
		overlay_map: Dict[Tuple[int, int], str] = {}
		if best is not None:
			items = best.get("items", [])  # type: ignore
			layer_dir = os.path.join(export_root, f"layer{lidx}")
			cmp_dir = os.path.join(layer_dir, "comparisons")
			_ensure_dir(cmp_dir)
			for it in items:  # type: ignore
				img_idx = int(it["image_idx"])  # type: ignore
				feat_idx = int(it["feature_idx"])  # type: ignore
				thr = float(it["threshold"])  # type: ignore
				hm = it["heatmap"]  # type: ignore
				msk = it["mask_input"]  # type: ignore
				if hm is None:
					continue
				cmp_name = f"best_Bild{img_idx}_Feature{feat_idx}.png"
				cmp_path = os.path.join(cmp_dir, cmp_name)
				_save_overlay_comparison(cmp_path, msk, hm, thr)
				overlay_map[(img_idx, feat_idx)] = os.path.relpath(cmp_path, start=export_root)

		rows_sorted = sorted(rows, key=lambda r: r.get("iou", 0.0), reverse=True)
		# Füge overlay_path für Best-Items ein
		for r in rows_sorted:
			key = (int(r["image_idx"]), int(r["feature_idx"]))
			if key in overlay_map:
				r["overlay_path"] = overlay_map[key]

		layer_dir = os.path.join(export_root, f"layer{lidx}")
		_ensure_dir(layer_dir)
		csv_path = os.path.join(layer_dir, "iou_sorted.csv")
		_write_csv(csv_path, rows_sorted)

	if count == 0:
		print("Keine Daten gefunden. Bitte zuvor die Extraktion ausführen.")
	else:
		print(f"Export abgeschlossen. Root: {export_root}")



def main_print_all() -> None:
	"""Berechnet und druckt alle IoUs inkl. Layer-, Bild- und Feature-Zuordnung.

	Ausgabe pro Level in der Form:
	Layer=<L> Bild=<B> Feature=<F> Level=<LVL> map=<HxW> thr=<T> IoU=<IOU> pos=<N>
	"""
	# Import hier durchführen, damit Skript sowohl als Modul als auch direkt lauffähig ist
	try:
		from .iou_core import compute_iou_per_levels  # type: ignore
	except Exception:
		# Fallback, wenn als Skript ohne Paketstruktur ausgeführt
		import os as _os, sys as _sys
		_sys.path.append(_os.path.dirname(__file__))
		from iou_core import compute_iou_per_levels  # type: ignore
	count = 0
	for item in iter_iou_inputs():
		results = compute_iou_per_levels(
			item,
			threshold_method="percentile",
			threshold_value=80.0,
			threshold_absolute=None,
		)
		for r in results:
			print(
				f"Layer={r.layer_idx} Bild={r.image_idx} Feature={r.feature_idx} "
				f"Level={r.level_idx} map={r.map_shape} thr={r.threshold:.4f} "
				f"IoU={r.iou:.6f} pos={r.positives}"
			)
			count += 1
	if count == 0:
		print("Keine Daten gefunden. Bitte zuvor die Extraktion ausführen.")


if __name__ == "__main__":
	# Exportiere kombinierte Heatmaps & IoUs pro Layer
	main_export_combined()

