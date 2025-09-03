"""
Network Dissection für den MaskDINO Decoder

Ziel:
- Für ein gegebenes Bild und eine Zielmaske (z. B. Rot-Bereich) werden die Decoder-Ausgaben
  (pro Layer und pro Query) analysiert.
- Wir messen die IoU jeder Query-Maske gegen die Zielmaske und finden die beste(n) Query/Layer.
- Ergebnisse werden als Visualisierung (PNG) und als JSON-Report gespeichert.

Hinweise/Design:
- Wir registrieren einen Forward-Hook am MaskDINODecoder (sem_seg_head.predictor), um dessen
  Output-Dict (inkl. aux_outputs = Zwischenlayer) abzugreifen.
- Ohne invasive Modeländerungen nutzen wir so alle Decoder-Masken: initial_pred, L1..L{N}.
- Masken werden mit Sigmoid in Wahrscheinlichkeiten konvertiert und bei 0.5 binarisiert.

Abhängigkeiten:
- Aufbau & Pfade analog zu network_dissection_encoder.py und fine-tune.py
"""

# PIL/Pillow Kompatibilität Fix (muss vor allen anderen Imports stehen)
try:
	import PIL.Image
	if not hasattr(PIL.Image, 'LINEAR'):
		PIL.Image.LINEAR = PIL.Image.Resampling.BILINEAR
	if not hasattr(PIL.Image, 'CUBIC'):
		PIL.Image.CUBIC = PIL.Image.Resampling.BICUBIC
	if not hasattr(PIL.Image, 'LANCZOS'):
		PIL.Image.LANCZOS = PIL.Image.Resampling.LANCZOS
	if not hasattr(PIL.Image, 'NEAREST'):
		PIL.Image.NEAREST = PIL.Image.Resampling.NEAREST
	print("✅ PIL compatibility fixed")
except Exception as e:
	print(f"⚠️ PIL fix failed, but continuing: {e}")

# NumPy Kompatibilität Fix für neuere NumPy-Versionen
try:
	import numpy as np
	if not hasattr(np, 'bool'):
		np.bool = bool
	if not hasattr(np, 'int'):
		np.int = int
	if not hasattr(np, 'float'):
		np.float = float
	if not hasattr(np, 'complex'):
		np.complex = complex
	print("✅ NumPy compatibility fixed")
except Exception as e:
	print(f"⚠️ NumPy fix failed, but continuing: {e}")

import os
import sys
import cv2
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional

from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

# MaskDINO Setup
MASKDINO_PATH = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO"
if MASKDINO_PATH not in sys.path:
	sys.path.insert(0, MASKDINO_PATH)
try:
	from maskdino import add_maskdino_config
	from maskdino.modeling.transformer_decoder.maskdino_decoder import MaskDINODecoder
except Exception as e:
	print(f"❌ Failed to import MaskDINO modules: {e}")
	raise

# Trainings-YAML (aus Fine-Tune übernommen)
TRAIN_YAML_PATH = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"


def register_datasets():
	"""Registriere das Car-Parts-COCO-Dataset (für Metadaten/Konsistenz)."""
	dataset_root = "/Users/nicklehmacher/Alles/MasterArbeit/ultralytics/datasets"
	name = "car_parts_train"
	if name not in DatasetCatalog.list():
		register_coco_instances(
			name,
			{},
			os.path.join(dataset_root, "annotations", "instances_train2017.json"),
			os.path.join(dataset_root, "images"),
		)
	car_parts_classes = [
		'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
		'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
		'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
		'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
		'tailgate', 'trunk', 'wheel'
	]
	MetadataCatalog.get(name).set(thing_classes=car_parts_classes)
	return car_parts_classes


class DecoderDissector:
	"""Netzwerk-Dissektion für den MaskDINO-Decoder (Queries x Layer)."""

	def __init__(self, weights_path: str, device: str = "cpu"):
		self.weights_path = weights_path
		self.device = device
		self.model = None
		self.cfg = None
		self.decoder_out: Optional[Dict] = None
		self.output_dir = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/decoder_dissection"
		os.makedirs(self.output_dir, exist_ok=True)

		self._build_and_load()
		self._hook_decoder()

	def _build_and_load(self):
		cfg = get_cfg()
		add_maskdino_config(cfg)
		# Erlaube unbekannte Keys aus dem Trainings-YAML (z. B. wenn Detectron2-Version abweicht)
		try:
			cfg.set_new_allowed(True)
		except Exception:
			pass
		cfg.merge_from_file(TRAIN_YAML_PATH)
		cfg.MODEL.WEIGHTS = self.weights_path
		cfg.MODEL.DEVICE = self.device
		# Nach dem Merge wieder schließen
		try:
			cfg.set_new_allowed(False)
		except Exception:
			pass
		self.cfg = cfg.clone()
		self.cfg.freeze()

		self.model = build_model(self.cfg)
		self.model.eval()
		checkpointer = DetectionCheckpointer(self.model)
		print(f"🔧 Lade Gewichte: {self.weights_path}")
		checkpointer.load(self.weights_path)
		print("✅ Modell geladen")

	def _hook_decoder(self):
		"""Hook nur am MaskDINODecoder registrieren und Ausgaben streng validieren."""
		def hook_fn(module, inputs, output):
			# Erwartet wird ein Dict mit 'pred_masks' (Tensor[B,Q,H,W]) und 'aux_outputs' (Liste/Tuple von Dicts)
			out_dict = None
			if isinstance(output, dict):
				out_dict = output
			elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], dict):
				out_dict = output[0]
			else:
				print(f"⚠️ Decoder-Hook: Unerwarteter Output-Typ: {type(output)}")
				self.decoder_out = None
				return

			pred_masks = out_dict.get('pred_masks', None)
			aux_outputs = out_dict.get('aux_outputs', None)
			valid_aux = isinstance(aux_outputs, (list, tuple)) and all(isinstance(d, dict) and 'pred_masks' in d for d in aux_outputs)
			if isinstance(pred_masks, torch.Tensor) and pred_masks.ndim == 4 and valid_aux:
				self.decoder_out = out_dict
			else:
				keys = list(out_dict.keys())
				print(f"⚠️ Decoder-Hook: Fehlende/inkorrekte Keys. Verfügbare Keys: {keys}")
				self.decoder_out = None

		for name, module in self.model.named_modules():
			if isinstance(module, MaskDINODecoder):
				module.register_forward_hook(hook_fn)
				print(f"🎯 Decoder-Hook registriert an: {name}")
				return
		print("⚠️ Kein MaskDINODecoder-Modul für Hook gefunden – Analyse könnte fehlschlagen.")

	# ---------------------- I/O & Vorverarbeitung ----------------------
	def _preprocess_image(self, image_path: str):
		img_bgr = cv2.imread(image_path)
		if img_bgr is None:
			raise ValueError(f"Bild konnte nicht geladen werden: {image_path}")
		img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

		h, w = img.shape[:2]
		scale = min(self.cfg.INPUT.MIN_SIZE_TEST / min(h, w), self.cfg.INPUT.MAX_SIZE_TEST / max(h, w))
		new_h, new_w = int(h * scale), int(w * scale)
		img_resized = cv2.resize(img, (new_w, new_h))

		# Keine manuelle Normalisierung: unnormalisierte RGB-Werte (0–255, float32)
		tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()
		return tensor, img_resized

	@staticmethod
	def _load_binary_mask(mask_path: str) -> np.ndarray:
		m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
		if m is None:
			raise ValueError(f"Maske konnte nicht geladen werden: {mask_path}")
		return (m > 128).astype(np.uint8)

	# -------------------------- Kernanalyse ----------------------------
	@staticmethod
	def _compute_iou(a: np.ndarray, b: np.ndarray) -> float:
		inter = np.logical_and(a, b).sum()
		union = np.logical_or(a, b).sum()
		return float(inter) / float(union) if union > 0 else 0.0

	def _gather_all_layer_masks(self, out_dict: Dict) -> List[Tuple[str, torch.Tensor]]:
		"""Sammle alle [B, Q, H, W]-Maskentensoren aus aux_outputs + finalem Output.
		Rückgabe: Liste aus (LayerLabel, Tensor)
		LayerLabel: "initial", "dec1", ..., "decN"
		"""
		masks: List[Tuple[str, torch.Tensor]] = []

		aux = out_dict.get('aux_outputs', []) or []
		for idx, d in enumerate(aux):
			label = "initial" if idx == 0 else f"dec{idx}"
			if 'pred_masks' in d and isinstance(d['pred_masks'], torch.Tensor):
				masks.append((label, d['pred_masks']))

		# letztes (finales) Layer separat
		if 'pred_masks' in out_dict and isinstance(out_dict['pred_masks'], torch.Tensor):
			# finaler Layer-Index = Anzahl aux-Elemente
			final_label = f"dec{len(aux)}" if len(aux) > 0 else "final"
			masks.append((final_label, out_dict['pred_masks']))

		return masks

	def analyze(self, image_path: str, target_mask_path: str, bin_threshold: float = 0.5) -> bool:
		print("🚀 Starte Decoder-Dissektion …")
		print(f"   📸 Bild: {image_path}")
		print(f"   🎯 Zielmaske: {target_mask_path}")

		image_tensor, image_vis = self._preprocess_image(image_path)
		target_mask = self._load_binary_mask(target_mask_path)

		# vorwärts – Hook sammelt Decoder-Output
		with torch.no_grad():
			self.decoder_out = None
			inputs = [{
				"image": image_tensor,
				"height": image_tensor.shape[1],
				"width": image_tensor.shape[2],
			}]
			_ = self.model(inputs)

		if self.decoder_out is None:
			print("❌ Decoder-Ausgaben konnten nicht abgegriffen werden.")
			return False

		layer_masks = self._gather_all_layer_masks(self.decoder_out)
		if not layer_masks:
			print("❌ Keine Decoder-Masken gefunden.")
			return False

		print(f"🔎 Gefundene Ebenen: {[lbl for lbl, _ in layer_masks]}")

		# IoU-Auswertung pro Layer & Query
		Ht, Wt = target_mask.shape
		best = {
			'layer': None,
			'query': None,
			'iou': 0.0,
			'mask_prob': None,
			'mask_bin': None,
		}
		heatmap_rows: List[str] = []

		for layer_idx, (label, masks_tensor) in enumerate(layer_masks):
			# masks_tensor: [B, Q, Hm, Wm]; wir nutzen Batch 0
			masks_prob = torch.sigmoid(masks_tensor[0]).cpu().numpy()  # [Q, Hm, Wm]
			Q, Hm, Wm = masks_prob.shape

			ious_this_layer: List[Tuple[int, float]] = []
			for q in range(Q):
				pm = masks_prob[q]
				if (Hm, Wm) != (Ht, Wt):
					pm_resized = cv2.resize(pm, (Wt, Ht))
				else:
					pm_resized = pm
				mb = (pm_resized >= bin_threshold).astype(np.uint8)
				iou = self._compute_iou(mb, target_mask)
				ious_this_layer.append((q, iou))
				if iou > best['iou']:
					best.update({
						'layer': label,
						'query': q,
						'iou': iou,
						'mask_prob': pm_resized,
						'mask_bin': mb,
					})

			# Top-10 für Heatmap/Report
			ious_this_layer.sort(key=lambda x: x[1], reverse=True)
			top_line = f"{label}: " + ", ".join([f"q{qi}:{iou:.3f}" for qi, iou in ious_this_layer[:10]])
			heatmap_rows.append(top_line)
			print("   ", top_line)

		# Alle 300 Queries des FINALEN Layers als 16-stufige Heatmaps speichern
		final_label, final_tensor = layer_masks[-1]
		final_prob = torch.sigmoid(final_tensor[0]).cpu().numpy()  # [Q, Hm, Wm]
		self._save_query_heatmaps(final_label, final_prob, target_mask.shape, num_levels=16)

		if best['layer'] is None:
			print("❌ Keine passende Query gefunden.")
			return False

		print(f"✅ Bestes Ergebnis: Layer={best['layer']} Query={best['query']} IoU={best['iou']:.4f}")
		self._visualize_and_report(image_vis, target_mask, best, heatmap_rows)
		return best['iou'] > 0.3

	# --------------------- Visualisierung & Report ---------------------
	def _visualize_and_report(self, image_vis: np.ndarray, target_mask: np.ndarray,
							   best: Dict, heatmap_rows: List[str]):
		# 2x3 Layout wie im Encoder-Skript
		fig, axes = plt.subplots(2, 3, figsize=(15, 10))
		fig.suptitle(
			f"Decoder-Dissektion – Layer {best['layer']} · Query {best['query']} · IoU {best['iou']:.4f}",
			fontsize=16,
		)

		axes[0, 0].imshow(image_vis)
		axes[0, 0].set_title('Original')
		axes[0, 0].axis('off')

		axes[0, 1].imshow(target_mask, cmap='Reds')
		axes[0, 1].set_title('Zielmaske (GT)')
		axes[0, 1].axis('off')

		axes[0, 2].imshow(best['mask_prob'], cmap='viridis')
		axes[0, 2].set_title('Beste Query – Wahrscheinlichkeit')
		axes[0, 2].axis('off')

		axes[1, 0].imshow(best['mask_bin'], cmap='Blues')
		axes[1, 0].set_title('Beste Query – Binär (p≥0.5)')
		axes[1, 0].axis('off')

		overlay = np.zeros((target_mask.shape[0], target_mask.shape[1], 3), dtype=np.float32)
		overlay[:, :, 0] = target_mask  # Rot = GT
		overlay[:, :, 2] = best['mask_bin']  # Blau = Query
		axes[1, 1].imshow(overlay)
		axes[1, 1].set_title('Overlay (Rot=GT, Blau=Query)')
		axes[1, 1].axis('off')

		# Kompakter Text-Block der Top-10 pro Layer (Heatmap-ähnlich)
		axes[1, 2].axis('off')
		heat_text = "\n".join(heatmap_rows)
		axes[1, 2].text(0.0, 0.5, heat_text, va='center', ha='left', fontsize=9, family='monospace')
		axes[1, 2].set_title('Top-10 IoUs pro Ebene')

		plt.tight_layout()
		vis_path = os.path.join(self.output_dir, 'decoder_dissection.png')
		plt.savefig(vis_path, dpi=300, bbox_inches='tight')
		plt.close(fig)
		print(f"💾 Visualisierung gespeichert: {vis_path}")

		# Report
		report = {
			'best_layer': best['layer'],
			'best_query': int(best['query']),
			'best_iou': float(best['iou']),
			'notes': 'IoU mit binarisiertem p>=0.5, Masken vorher via Sigmoid.',
			'layers_top10_preview': heatmap_rows,
		}
		rpt_path = os.path.join(self.output_dir, 'decoder_dissection_report.json')
		with open(rpt_path, 'w') as f:
			json.dump(report, f, indent=2)
		print(f"💾 Report gespeichert: {rpt_path}")

	# ---------------------------- Export -------------------------------
	def _save_query_heatmaps(self, layer_label: str, masks_prob: np.ndarray,
						   target_shape: Tuple[int, int], num_levels: int = 16):
		"""Speichere für alle Queries 16-stufig quantisierte Heatmaps als PNG (Graustufen).
		masks_prob: [Q, Hm, Wm] nach Sigmoid in [0,1]
		target_shape: (Ht, Wt) – Zielgröße (i. d. R. Maske/Visualisierung)
		num_levels: Anzahl diskreter Stufen (Standard 16)
		"""
		out_dir = os.path.join(self.output_dir, "query_masks", layer_label)
		os.makedirs(out_dir, exist_ok=True)

		Ht, Wt = target_shape
		Q = masks_prob.shape[0]
		saved = 0
		for q in range(Q):
			pm = masks_prob[q]
			if pm.shape != (Ht, Wt):
				pm = cv2.resize(pm, (Wt, Ht))
			# Quantisierung in num_levels Stufen (0 .. num_levels-1)
			bins = np.floor(pm * num_levels)
			bins = np.clip(bins, 0, num_levels - 1).astype(np.uint8)
			# Auf 0..255 skalieren für bessere Visualisierung
			if num_levels > 1:
				levels_255 = (bins.astype(np.float32) * (255.0 / (num_levels - 1))).round().astype(np.uint8)
			else:
				levels_255 = (bins * 0).astype(np.uint8)
			path = os.path.join(out_dir, f"q{q:03d}.png")
			cv2.imwrite(path, levels_255)
			saved += 1
		print(f"💾 {saved} quantisierte Query-Heatmaps (\u2264{num_levels} Stufen) gespeichert unter: {out_dir}")


def main():
	print("🔵 MaskDINO Decoder-Dissektion")
	print("=" * 60)

	model_path = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_final.pth"
	image_path = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/new_21_png_jpg.rf.d0c9323560db430e693b33b36cb84c3b.jpg"
	target_mask_path = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/colours/rot.png"

	# Vorbedingungen prüfen
	for p, n in [(model_path, 'Modell'), (image_path, 'Bild'), (target_mask_path, 'Maske')]:
		if not os.path.exists(p):
			print(f"❌ {n} nicht gefunden: {p}")
			return

	setup_logger(name="maskdino")
	register_datasets()

	try:
		dissector = DecoderDissector(model_path, device="cpu")
		success = dissector.analyze(image_path, target_mask_path)
		if success:
			print("🎉 Analyse erfolgreich abgeschlossen (IoU > 0.3).")
		else:
			print("⚠️ Analyse abgeschlossen, jedoch niedrige Übereinstimmung.")
	except Exception as e:
		print(f"❌ Fehler während der Analyse: {e}")
		import traceback
		traceback.print_exc()


if __name__ == "__main__":
	main()

