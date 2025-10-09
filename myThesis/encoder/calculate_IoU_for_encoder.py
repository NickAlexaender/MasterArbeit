"""
Network Dissection für Encoder-Features.

Implementiert die Network Dissection Methode zur Identifikation konzept-spezifischer Features:
- iteriert über output/encoder/layer*/feature.csv
- extrahiert Layer-, Bild- und Feature-Index sowie Tokens (pro Zeile)
- lädt passende Bild-Infos aus output/encoder/<image_id>/shapes.json
- bereitet Masken aus myThesis/image/rot/ in Input-Größe vor
- berechnet per-Feature Thresholds über alle Bilder (Network Dissection)
- exportiert mIoU-Ergebnisse und Visualisierungen

Hauptfunktion: main_export_network_dissection()

Ausgabe:
- myThesis/output/encoder/network_dissection/layerX/miou_network_dissection.csv
- myThesis/output/encoder/network_dissection/layerX/network_dissection/FeatureY/*.png
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
# Globale Konfiguration
# -----------------------------

# Network Dissection: Per-Feature Threshold (0-100 für Perzentil)
NETWORK_DISSECTION_PERCENTILE = 90.0


# -----------------------------
# Datenstruktur für iou_core.py
# -----------------------------

@dataclass
class IoUInput:
	layer_idx: int
	image_id: str  # String-ID (z.B. "image 1") statt numerischer Index
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


def _mask_dir() -> str:
	return os.path.join(_project_root(), "image", "rot")


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


def _select_shapes_for(image_id: str, n_tokens: int, all_shapes: Dict[str, Dict]) -> Optional[Dict]:
	"""Wählt passende shapes.json für die gegebene image_id.

	Strategie:
	1) Direkter Match über image_id (bevorzugt).
	2) Wenn nicht gefunden: Match über N_tokens.
	3) Fallback: gebe None zurück (kein unsicheres Raten mehr).
	"""
	if not all_shapes:
		return None

	# 1) Direkter Match über image_id (z.B. "image 1" -> Ordner "image 1")
	if image_id in all_shapes:
		return all_shapes[image_id]

	# 2) Match über image_id in der shapes.json selbst (falls Ordnername abweicht)
	for folder_name, shapes_data in all_shapes.items():
		if shapes_data.get("image_id") == image_id:
			return shapes_data

	# 3) Match über N_tokens als Fallback (wenn mehrere vorhanden, nimm ersten Match)
	matches = [v for v in all_shapes.values() if int(v.get("N_tokens", -1)) == int(n_tokens)]
	if len(matches) == 1:
		return matches[0]
	if len(matches) > 1:
		# Mehrere Matches mit gleicher Token-Anzahl – wähle deterministisch nach image_id
		matches_with_id = [(v.get("image_id", ""), v) for v in matches]
		matches_with_id.sort(key=lambda x: x[0])
		return matches_with_id[0][1]

	# 4) Keine Übereinstimmung gefunden
	print(f"⚠️ Keine passende shapes.json für image_id='{image_id}' gefunden (N_tokens={n_tokens})")
	return None


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

def _load_mask_for_input(input_size: Tuple[int, int], image_id: str) -> np.ndarray:
	"""Lädt die Maske für das gegebene Bild und liefert sie als bool-Array in input-Größe (H_in, W_in)."""
	mask_dir = _mask_dir()
	# Konstruiere den Maskenpfad basierend auf der image_id
	# Beispiel: image_id = "image 1" -> mask_file = "image 1.jpg"
	mask_file = os.path.join(mask_dir, f"{image_id}.jpg")
	if not os.path.isfile(mask_file):
		# Falls .jpg nicht existiert, versuche .png
		mask_file = os.path.join(mask_dir, f"{image_id}.png")
	
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

# Neues Format: "image 1, Feature1" statt "Bild1, Feature1"
_NAME_RE = re.compile(r"^(.+?),\s*Feature(\d+)$")


def _iter_csv_rows(csv_path: str) -> Iterable[Tuple[str, int, np.ndarray]]:
	"""Iteriert Zeilen einer feature.csv und liefert (image_id, feature_idx, tokens).

	tokens ist ein 1D np.ndarray[N] float32.
	image_id ist der String-Identifikator (z.B. "image 1").
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
			image_id = m.group(1).strip()
			feat_idx = int(m.group(2))
			try:
				values = [float(x) for x in row[1:]]
			except ValueError:
				# Überspringe fehlerhafte Zeilen
				continue
			tokens = np.asarray(values, dtype=np.float32)
			yield image_id, feat_idx, tokens


def iter_iou_inputs() -> Generator[IoUInput, None, None]:
	"""Haupt-Iterator: liefert pro CSV-Zeile ein IoUInput-Paket.
	- Erkennt Layer-Index aus Ordnernamen.
	- Mappt image_id auf passende shapes.json (direkter Match).
	- Bereitet Maske für input-Size vor (gecacht pro input_size UND image_id).
	"""
	layer_csvs = _find_layer_csvs()
	all_shapes = _load_all_shapes()
	# Cache: Maske je (input_size, image_id) - da jetzt verschiedene Masken für verschiedene Bilder
	mask_cache_input: Dict[Tuple[int, int, str], np.ndarray] = {}

	for lidx, csv_path in layer_csvs:
		for image_id, feat_idx, tokens in _iter_csv_rows(csv_path):
			shapes = _select_shapes_for(image_id, tokens.size, all_shapes)
			if shapes is None:
				# Ohne Shapes ist Reassemblierung in iou_core schwierig – skippe
				continue
			Hin, Win = int(shapes["input_size"][0]), int(shapes["input_size"][1])
			# image_id aus shapes verwenden (sollte identisch sein, aber sicherer)
			image_id_from_shapes = shapes.get("image_id", image_id)

			# Maske beschaffen (gecacht nach input-Größe UND image_id)
			key_in = (Hin, Win, image_id_from_shapes)
			if key_in in mask_cache_input:
				mask_input = mask_cache_input[key_in]
			else:
				mask_input = _load_mask_for_input((Hin, Win), image_id_from_shapes)
				mask_cache_input[key_in] = mask_input

			yield IoUInput(
				layer_idx=lidx,
				image_id=image_id_from_shapes,
				feature_idx=feat_idx,
				tokens=tokens,
				shapes=shapes,
				mask_input=mask_input,
			)


# -----------------------------
# FeatureHeatmapAggregator für globale Binarisierung
# -----------------------------

class FeatureHeatmapAggregator:
	"""Sammelt Heatmaps eines Features über alle Bilder für globale Binarisierung."""

	def __init__(self, layer_idx: int, feature_idx: int):
		self.layer_idx = layer_idx
		self.feature_idx = feature_idx
		self.heatmaps: List[np.ndarray] = []
		self.masks: List[np.ndarray] = []
		self.image_ids: List[str] = []

	def add_heatmap(self, heatmap: np.ndarray, mask: np.ndarray, image_id: str) -> None:
		"""Fügt eine Heatmap + zugehörige Maske hinzu."""
		if heatmap.shape != mask.shape:
			raise ValueError(f"Shape-Mismatch: Heatmap {heatmap.shape} vs. Maske {mask.shape}")
		self.heatmaps.append(heatmap)
		self.masks.append(mask)
		self.image_ids.append(image_id)

	def compute_network_dissection_threshold(self, percentile: float = None) -> float:
		"""Berechnet Network Dissection Threshold über alle Pixel aller Heatmaps dieses Features.

		Berechnet das Perzentil über die vereinte Menge aller Aktivierungswerte.
		Dies entspricht der Network Dissection Methode.

		Args:
			percentile: Perzentil (0-100) - falls None, wird NETWORK_DISSECTION_PERCENTILE verwendet

		Returns:
			float: Per-Feature Threshold
		"""
		if not self.heatmaps:
			return 0.0
		
		if percentile is None:
			percentile = NETWORK_DISSECTION_PERCENTILE
		
		# Flatten alle Heatmaps zu einem 1D-Array aller Aktivierungswerte
		all_values = np.concatenate([hm.flatten() for hm in self.heatmaps])
		return float(np.percentile(all_values, float(percentile)))

	def compute_miou(self, threshold: float) -> Tuple[float, List[float]]:
		"""Berechnet mIoU über alle Bilder mit gegebenem Threshold.

		Returns:
			(mIoU, List[individual_ious])
		"""
		if not self.heatmaps:
			return 0.0, []
		# Lazy-Import von iou_core
		try:
			from .iou_core import compute_iou_from_heatmap  # type: ignore
		except Exception:
			import os as _os, sys as _sys
			_sys.path.append(_os.path.dirname(__file__))
			from iou_core import compute_iou_from_heatmap  # type: ignore

		ious = [
			compute_iou_from_heatmap(hm, msk, threshold)
			for hm, msk in zip(self.heatmaps, self.masks)
		]
		miou = float(np.mean(ious))
		return miou, ious


# -----------------------------
# Grundstruktur + Ausgabe
# -----------------------------

def _ensure_dir(p: str) -> None:
	os.makedirs(p, exist_ok=True)


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


# -----------------------------
# Network Dissection Pipeline
# -----------------------------

def main_export_network_dissection() -> None:
	"""Network Dissection Pipeline mit per-Feature Thresholding.

	Workflow:
	1. Sammle alle Heatmaps pro Feature
	2. Pro Feature: Berechne Network Dissection Threshold (Perzentil über ALLE Pixel ALLER Bilder)
	3. Binarisiere alle Heatmaps mit diesem Feature-Threshold
	4. Berechne mIoU über alle Bilder
	5. Exportiere:
	   - layerX/miou_network_dissection.csv (sortiert nach mIoU)
	   - layerX/network_dissection/FeatureY/*.png (Visualisierungen ALLER Bilder für beste Features)
	"""
	# Lazy-Import
	try:
		from .iou_core import generate_heatmap_only  # type: ignore
	except Exception:
		import os as _os, sys as _sys
		_sys.path.append(_os.path.dirname(__file__))
		from iou_core import generate_heatmap_only  # type: ignore

	export_root = os.path.join(_encoder_out_dir(), "network_dissection")
	_ensure_dir(export_root)

	# Schritt 1: Gruppiere nach (layer_idx, feature_idx) und sammle Heatmaps
	aggregators: Dict[Tuple[int, int], FeatureHeatmapAggregator] = {}

	print("Sammle Heatmaps pro Feature (Network Dissection)...")
	count = 0
	for item in iter_iou_inputs():
		key = (item.layer_idx, item.feature_idx)
		if key not in aggregators:
			aggregators[key] = FeatureHeatmapAggregator(item.layer_idx, item.feature_idx)
		
		# Generiere kontinuierliche Heatmap (KEINE Binarisierung)
		heatmap = generate_heatmap_only(item, combine="max")
		aggregators[key].add_heatmap(heatmap, item.mask_input, item.image_id)
		count += 1

	if count == 0:
		print("Keine Daten gefunden. Bitte zuvor die Extraktion ausführen.")
		return

	print(f"Gesammelt: {count} Heatmaps aus {len(aggregators)} Features")

	# Schritt 2: Berechne mIoU mit Network Dissection Thresholding
	per_layer: Dict[int, List[Dict[str, object]]] = {}
	per_layer_best: Dict[int, Dict[str, object]] = {}

	print(f"Berechne mIoU mit Network Dissection Threshold (Perzentil={NETWORK_DISSECTION_PERCENTILE})...")
	for (lidx, fidx), agg in aggregators.items():
		# Network Dissection Threshold: Perzentil über alle Pixel aller Bilder dieses Features
		nd_threshold = agg.compute_network_dissection_threshold(percentile=NETWORK_DISSECTION_PERCENTILE)
		miou, ious = agg.compute_miou(nd_threshold)

		row = {
			"layer_idx": lidx,
			"feature_idx": fidx,
			"miou": float(miou),
			"nd_threshold": float(nd_threshold),
			"n_images": len(ious),
			"individual_ious": ",".join(f"{x:.6f}" for x in ious),
			"overlay_dir": "",
		}
		per_layer.setdefault(lidx, []).append(row)

		# Tracking der besten Features pro Layer
		best = per_layer_best.get(lidx)
		if best is None:
			per_layer_best[lidx] = {
				"best_miou": float(miou),
				"features": [
					{
						"feature_idx": fidx,
						"threshold": nd_threshold,
						"aggregator": agg,
					}
				],
			}
		else:
			cur_best = float(best["best_miou"])  # type: ignore
			if float(miou) > cur_best + 1e-12:
				best["best_miou"] = float(miou)
				best["features"] = [
					{
						"feature_idx": fidx,
						"threshold": nd_threshold,
						"aggregator": agg,
					}
				]
			elif abs(float(miou) - cur_best) <= 1e-12:
				best.setdefault("features", []).append(
					{
						"feature_idx": fidx,
						"threshold": nd_threshold,
						"aggregator": agg,
					}
				)

	# Schritt 3: Exportiere binäre CSVs und Overlays für beste Features
	print("Erstelle binäre CSVs und Overlays...")
	for lidx, rows in per_layer.items():
		layer_dir = os.path.join(export_root, f"layer{lidx}")
		_ensure_dir(layer_dir)

		# Overlays für beste(s) Feature(s) - ALLE Bilder visualisieren
		best = per_layer_best.get(lidx)
		overlay_map: Dict[int, str] = {}  # feature_idx -> overlay_dir
		if best is not None:
			features = best.get("features", [])  # type: ignore
			for feat_info in features:  # type: ignore
				fidx = int(feat_info["feature_idx"])  # type: ignore
				thr = float(feat_info["threshold"])  # type: ignore
				agg_best = feat_info["aggregator"]  # type: ignore

				# Unterordner für dieses Feature
				feat_dir = os.path.join(layer_dir, "network_dissection", f"Feature{fidx}")
				_ensure_dir(feat_dir)

				# Visualisiere ALLE Bilder dieses Features
				for image_id, heatmap, mask in zip(agg_best.image_ids, agg_best.heatmaps, agg_best.masks):
					overlay_path = os.path.join(feat_dir, f"{image_id}.png")
					_save_overlay_comparison(overlay_path, mask, heatmap, thr)
				
				overlay_map[fidx] = os.path.relpath(feat_dir, start=export_root)

		# Sortiere Rows nach mIoU absteigend
		rows_sorted = sorted(rows, key=lambda r: r.get("miou", 0.0), reverse=True)
		# Füge overlay_dir für Best-Features ein
		for r in rows_sorted:
			fidx = int(r["feature_idx"])
			if fidx in overlay_map:
				r["overlay_dir"] = overlay_map[fidx]

		# Schreibe mIoU-CSV
		csv_path = os.path.join(layer_dir, "miou_network_dissection.csv")
		_write_network_dissection_csv(csv_path, rows_sorted)

	print(f"Network Dissection Export abgeschlossen. Root: {export_root}")


def _write_network_dissection_csv(path: str, rows: List[Dict[str, object]]) -> None:
	"""Schreibt Network Dissection mIoU-Ergebnisse in CSV."""
	if not rows:
		return
	fieldnames = [
		"layer_idx",
		"feature_idx",
		"miou",
		"nd_threshold",
		"n_images",
		"individual_ious",
		"overlay_dir",
	]
	with open(path, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


if __name__ == "__main__":
	# Network Dissection Pipeline mit per-Feature Thresholding
	main_export_network_dissection()




