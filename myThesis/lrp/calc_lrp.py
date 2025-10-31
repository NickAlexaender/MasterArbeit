"""
LRP/Attributions-Analyse für MaskDINO-Transformer Encoder

Funktion:
- Lädt das MaskDINO-Modell (wie in myThesis/fine-tune.py konfiguriert)
- Führt eine Attribution (LRP) für ein wählbares Encoder-/Decoder-Layer
	und ein bestimmtes Feature (Kanalindex) durch
- Aggregiert Beiträge der vorherigen Features (Kanäle) über alle Bilder im Ordner
- Exportiert Ergebnisse als CSV-Datei

"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import logging
import gc
from typing import Dict, List, Tuple, Iterable, Any, Optional

# PIL/Pillow Kompatibilität Fix (muss vor allen anderen Imports stehen)
try:
	import PIL.Image
	# Fix für neuere Pillow-Versionen
	if not hasattr(PIL.Image, 'LINEAR'):
		PIL.Image.LINEAR = PIL.Image.Resampling.BILINEAR
	if not hasattr(PIL.Image, 'CUBIC'):
		PIL.Image.CUBIC = PIL.Image.Resampling.BICUBIC
	if not hasattr(PIL.Image, 'LANCZOS'):
		PIL.Image.LANCZOS = PIL.Image.Resampling.LANCZOS
	if not hasattr(PIL.Image, 'NEAREST'):
		PIL.Image.NEAREST = PIL.Image.Resampling.NEAREST
except Exception as e:  # pragma: no cover - defensive
	print(f"⚠️ PIL fix failed, but continuing: {e}")

# NumPy Kompatibilität Fix
try:
	import numpy as np
	if not hasattr(np, 'bool'):
		np.bool = bool  # type: ignore[attr-defined]
	if not hasattr(np, 'int'):
		np.int = int  # type: ignore[attr-defined]
	if not hasattr(np, 'float'):
		np.float = float  # type: ignore[attr-defined]
	if not hasattr(np, 'complex'):
		np.complex = complex  # type: ignore[attr-defined]
except Exception as e:  # pragma: no cover - defensive
	print(f"⚠️ NumPy fix failed, but continuing: {e}")

import torch
import torch.nn as nn
from torch import Tensor

import pandas as pd

# Detectron2 & MaskDINO
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor  # nur für Referenz; nicht genutzt
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog


# MaskDINO zum Python-Pfad hinzufügen (wie in fine-tune.py)
MASKDINO_PATH = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO"
if MASKDINO_PATH not in sys.path:
	sys.path.insert(0, MASKDINO_PATH)

try:
	from maskdino import add_maskdino_config  # type: ignore
except Exception as e:
	raise RuntimeError(
		f"MaskDINO konnte nicht importiert werden: {e}. Prüfe Pfad {MASKDINO_PATH}"
	)

# LRP Engine (robust import for both -m module and direct script execution)
try:
	from myThesis.lrp.engine import LRPConfig, LRPTracer, propagate_lrp  # type: ignore
except Exception:
	# Fallback: adjust sys.path when run as a script
	_this_dir = os.path.dirname(os.path.abspath(__file__))
	_parent = os.path.dirname(_this_dir)
	if _parent not in sys.path:
		sys.path.insert(0, _parent)
	from myThesis.lrp.engine import LRPConfig, LRPTracer, propagate_lrp  # type: ignore


DEFAULT_WEIGHTS = (
	"/Users/nicklehmacher/Alles/MasterArbeit/myThesis/weights/"
	"maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"
)


def build_cfg_for_inference(device: str = "cpu"):
	"""Erzeuge eine Inferenz-Konfiguration, die mit den Gewichten kompatibel ist.

	Wir lehnen uns eng an myThesis/fine-tune.py an, aber nur mit den nötigen Schlüsseln
	für das Laden des Modells zur Inferenz.
	"""
	cfg = get_cfg()
	add_maskdino_config(cfg)

	# Meta-Architektur
	cfg.MODEL.META_ARCHITECTURE = "MaskDINO"

	# Backbone / ResNet
	cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
	cfg.MODEL.BACKBONE.FREEZE_AT = 0
	cfg.MODEL.RESNETS.DEPTH = 50
	cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
	cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
	cfg.MODEL.RESNETS.NORM = "FrozenBN"
	cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
	cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
	cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]

	# Normalisierung
	cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
	cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

	# SemSeg-Head / MaskDINO Head
	cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskDINOHead"
	cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
	cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 23
	cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
	cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
	cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
	cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
	cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MaskDINOEncoder"
	cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
	cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
	cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 3
	cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
	cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
	cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
	cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6

	# MaskDINO Decoder/Transformer Einstellungen (wie in fine-tune.py)
	cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "MaskDINODecoder"
	cfg.MODEL.MaskDINO.DEEP_SUPERVISION = True
	cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT = 0.1
	cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
	cfg.MODEL.MaskDINO.MASK_WEIGHT = 5.0
	cfg.MODEL.MaskDINO.DICE_WEIGHT = 5.0
	cfg.MODEL.MaskDINO.BOX_WEIGHT = 5.0
	cfg.MODEL.MaskDINO.GIOU_WEIGHT = 2.0
	cfg.MODEL.MaskDINO.HIDDEN_DIM = 256
	cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 300
	cfg.MODEL.MaskDINO.NHEADS = 8
	cfg.MODEL.MaskDINO.DROPOUT = 0.0
	cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = 2048
	cfg.MODEL.MaskDINO.ENC_LAYERS = 0
	cfg.MODEL.MaskDINO.PRE_NORM = False
	cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ = False
	cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY = 32
	cfg.MODEL.MaskDINO.DEC_LAYERS = 3
	cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS = 12544
	cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO = 3.0
	cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO = 0.75
	cfg.MODEL.MaskDINO.INITIAL_PRED = True
	cfg.MODEL.MaskDINO.TWO_STAGE = True
	cfg.MODEL.MaskDINO.DN = "seg"
	cfg.MODEL.MaskDINO.DN_NUM = 100
	cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE = "bitmask"

	# Test/Infer Einstellungen
	cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
	cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
	cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
	cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.8
	cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.25
	cfg.MODEL.MaskDINO.TEST.SCORE_THRESH_TEST = 0.5

	# Inputformat für Predictor
	cfg.INPUT.FORMAT = "RGB"
	cfg.INPUT.MIN_SIZE_TEST = 800
	cfg.INPUT.MAX_SIZE_TEST = 1333

	# Geräte/Weg
	cfg.MODEL.WEIGHTS = DEFAULT_WEIGHTS
	cfg.MODEL.DEVICE = device

	# Minimal-Datasets (werden registriert, bevor das Model gebaut wird)
	cfg.DATASETS.TRAIN = ("car_parts_minimal",)
	cfg.DATASETS.TEST = ("car_parts_minimal",)
	return cfg


def list_encoder_like_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
	"""Finde plausible Encoder-Layer-Module innerhalb MaskDINO.

	Strategie:
	- Bevorzuge Module, deren Pfadnamen Schlüsselwörter enthalten: 'encoder', 'transformer', 'pixel_decoder'
	- Filtere auf Blöcke, die selbst Unter-Module enthalten (Layer-Container), z.B. *EncoderLayer*
	- Ergebnisliste ist in DFS-Reihenfolge der Namen, stabil für Indexwahl
	"""
	keywords = ("encoder", "transformer", "pixel_decoder")
	candidates: List[Tuple[str, nn.Module]] = []
	for name, module in model.named_modules():
		lname = name.lower()
		if any(k in lname for k in keywords):
			# Wir wollen die eigentlichen Layer-Objekte, kein gesamter Encoder-Stack
			# Heuristik: Module mit Layer-Untermodulen oder mit Attention/FFN
			submods = list(module.children())
			if not submods:
				continue
			# Wenn das Modul selbst ein Layer ist (z.B. DeformableTransformerEncoderLayer oder TransformerEncoderLayer),
			# erkennen wir das an typischen Untermodule-Namen.
			subnames = {type(m).__name__.lower() for m in submods}
			if any(s in subnames for s in ["multiheadattention", "msdeformattn", "selfattention", "selfattn"]):
				candidates.append((name, module))
			else:
				# Wenn ein Container mehrere ähnliche Layer enthält, nehmen wir deren direkte Kinder mit passenden Mustern
				for cidx, child in enumerate(submods):
					cname = f"{name}.{cidx}"
					csub = list(child.children())
					csubnames = {type(m).__name__.lower() for m in csub}
					if any(s in csubnames for s in ["multiheadattention", "msdeformattn", "selfattention", "selfattn"]):
						candidates.append((cname, child))
	# Entferne Duplikate (gleiche Module-Objekte mit verschiedenen Namen)
	uniq: List[Tuple[str, nn.Module]] = []
	seen = set()
	for n, m in candidates:
		if id(m) not in seen:
			uniq.append((n, m))
			seen.add(id(m))
	return uniq


def list_decoder_like_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
	"""Finde plausible Decoder-Layer-Module innerhalb MaskDINO/Detectron2.

	Heuristik wie Encoder, aber Keywords auf Decoder fokussiert.
	"""
	keywords = ("decoder", "transformer_decoder")
	candidates: List[Tuple[str, nn.Module]] = []
	for name, module in model.named_modules():
		lname = name.lower()
		if any(k in lname for k in keywords):
			submods = list(module.children())
			if not submods:
				continue
			subnames = {type(m).__name__.lower() for m in submods}
			if any(s in subnames for s in ["multiheadattention", "msdeformattn", "selfattention", "selfattn"]):
				candidates.append((name, module))
			else:
				for cidx, child in enumerate(submods):
					cname = f"{name}.{cidx}"
					csub = list(child.children())
					csubnames = {type(m).__name__.lower() for m in csub}
					if any(s in csubnames for s in ["multiheadattention", "msdeformattn", "selfattention", "selfattn"]):
						candidates.append((cname, child))
	uniq: List[Tuple[str, nn.Module]] = []
	seen = set()
	for n, m in candidates:
		if id(m) not in seen:
			uniq.append((n, m))
			seen.add(id(m))
	return uniq


def _flatten(obj: Any) -> List[Any]:
	"""Flacht verschachtelte Strukturen (tuple/list) ab, behält Reihenfolge."""
	if isinstance(obj, (list, tuple)):
		res: List[Any] = []
		for it in obj:
			res.extend(_flatten(it))
		return res
	return [obj]


def _iter_tensors(obj: Any) -> Iterable[Tensor]:
	for x in _flatten(obj):
		if isinstance(x, torch.Tensor):
			yield x


def _first_tensor(obj: Any) -> Tensor:
	for t in _iter_tensors(obj):
		return t
	raise ValueError("Keine Tensor-Ausgabe im Layer gefunden")


class LayerCapture:
	"""Legacy placeholder kept for compatibility (no longer used with LRP).

	The LRP engine uses its own tracer; this class is retained to minimize diffs
	if external code still imports it. It does nothing.
	"""
	def __init__(self, module: nn.Module):
		self.module = module
		self.inputs_flat: List[Tensor] = []
		self.grad_inputs_flat: List[Optional[Tensor]] = []
		self.output_tensor: Optional[Tensor] = None

	def remove(self):
		pass


class _LayerTap:
	"""Registers a temporary forward hook on a chosen layer to cache x/y in the LRP tracer
	when the tracer didn't register it (unsupported module types)."""
	def __init__(self, tracer: LRPTracer, module: nn.Module, retain_grad: bool = False):
		self.tracer = tracer
		self.module = module
		self.retain_grad = retain_grad
		self._h = module.register_forward_hook(self._on_forward)

	def _on_forward(self, mod: nn.Module, inp, out):
		try:
			x = _first_tensor(inp)
			y = _first_tensor(out)
			# Falls Gradpfad gewünscht ist: Grad auf x behalten
			if self.retain_grad and isinstance(x, torch.Tensor) and x.requires_grad:
				x.retain_grad()
			self.tracer.store.set(mod, x=x, y=y)
		except Exception:
			pass

	def remove(self):
		try:
			self._h.remove()
		except Exception:
			pass


def _to_BTC(t: Tensor) -> Tensor:
	"""Bringe Tensor robust in Form (B, T, C).

	Regeln:
	- 4D: (B, C, H, W) -> (B, H*W, C)
	- 3D: Identifiziere Batch-Achse als Achse mit dem kleinsten Dimensionwert,
	  Tokens als größte Achse und die verbleibende als Kanal-Achse. Permutiere zu (B,T,C).
	  Dies deckt typische Fälle ab: (L,B,C), (B,L,C), (B,C,L).
	"""
	if t.dim() == 4:  # (B, C, H, W) -> (B, H*W, C)
		B, C, H, W = t.shape
		return t.permute(0, 2, 3, 1).reshape(B, H * W, C)
	if t.dim() == 3:
		dims = list(t.shape)
		# Bestimme Batch-/Token-/Kanal-Achsen heuristisch über Größenordnung
		b_axis = min(range(3), key=lambda i: dims[i])  # kleinste Dimension = Batch (typ. 1..8)
		t_axis = max(range(3), key=lambda i: dims[i])  # größte Dimension = Tokens (typ. 100+)
		c_axis = ({0, 1, 2} - {b_axis, t_axis}).pop()
		if (b_axis, t_axis, c_axis) == (0, 1, 2):
			return t  # bereits (B,T,C)
		return t.permute(b_axis, t_axis, c_axis)
	raise ValueError(f"Unerwartete Tensorform: {tuple(t.shape)}")


def aggregate_channel_relevance(R_in: Tensor) -> Tensor:
	"""Aggregiere Eingangsrelevanz zu einem Vektor (C_in,)."""
	if R_in.dim() == 4:  # (B, C, H, W)
		return R_in.sum(dim=(0, 2, 3)).detach().cpu()
	if R_in.dim() == 3:  # (B, L, C)
		return R_in.sum(dim=(0, 1)).detach().cpu()
	if R_in.dim() == 2:  # (B, C)
		return R_in.sum(dim=0).detach().cpu()
	raise ValueError(f"Unerwartete R_in-Form: {tuple(R_in.shape)}")


def build_target_relevance(
	layer_output: Tensor,
	feature_index: int,
	token_reduce: str,
	target_norm: str = "sum1",
	index_axis: str = "channel",
) -> Tensor:
	"""Erzeuge eine Start-Relevanz R_out ohne Gradienten.

	index_axis:
	- "channel": feature_index adressiert den Kanal (C-Achse)
	- "token":   feature_index adressiert den Token/Query (T-Achse)

	token_reduce wirkt nur bei index_axis="channel" und steuert die Verteilung über Tokens.
	"""
	y = _to_BTC(layer_output)  # (B, T, C)
	B, T, C = y.shape

	base = torch.zeros_like(y)

	if index_axis == "channel":
		if feature_index < 0 or feature_index >= C:
			raise IndexError(
				f"feature_index {feature_index} außerhalb [0, {C-1}] (axis=channel)"
			)
		feat = y[..., feature_index]
		if token_reduce == "mean":
			w = torch.ones_like(feat)
		elif token_reduce == "max":
			w = torch.relu(feat)
		else:
			raise ValueError("token_reduce muss 'mean' oder 'max' sein")
		# Normierung über alle Tokens/Batch
		s = w.sum().clamp_min(1e-12)
		if target_norm == "sum1":
			w = w / s
		elif target_norm == "sumT":
			w = w / s * float(T)
		elif target_norm == "none":
			pass
		else:
			raise ValueError("target_norm muss 'sum1', 'sumT' oder 'none' sein")
		base[..., feature_index] = w
	elif index_axis == "token":
		if feature_index < 0 or feature_index >= T:
			raise IndexError(
				f"feature_index {feature_index} außerhalb [0, {T-1}] (axis=token)"
			)
		# Verteile Gewicht gleichmäßig über Kanäle für den gewählten Token
		w_tok = torch.ones_like(y[:, feature_index, :])  # (B, C)
		s = w_tok.sum().clamp_min(1e-12)
		if target_norm == "sum1":
			w_tok = w_tok / s
		elif target_norm == "sumT":
			# für Token-Modus interpretieren wir 'T' als C-Anzahl
			w_tok = w_tok / s * float(C)
		elif target_norm == "none":
			pass
		else:
			raise ValueError("target_norm muss 'sum1', 'sumT' oder 'none' sein")
		base[:, feature_index, :] = w_tok
	else:
		raise ValueError("index_axis muss 'channel' oder 'token' sein")

	# Form zurück wie layer_output
	if layer_output.dim() == 4:
		if index_axis == "token":
			raise ValueError("index_axis='token' wird für 4D-Outputs nicht unterstützt")
		# (B,T,C)->(B,C,H,W)
		B2, C2, H, W = layer_output.shape
		assert B2 == B and C2 == C and H * W == T
		base = base.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
	return base


def collect_images(images_dir: str) -> List[str]:
	exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
	files: List[str] = []
	for ext in exts:
		files.extend(glob.glob(os.path.join(images_dir, ext)))
	return sorted(files)


def run_analysis(
	images_dir: str,
	layer_index: int,
	feature_index: int,
	output_csv: str,
	target_norm: str = "sum1",
	lrp_epsilon: float = 1e-6,
	which_module: str = "encoder",
	method: str = "lrp",
	index_kind: str = "auto",
):
	logger = logging.getLogger("lrp")
	logger.info("Starte LRP/Attribution-Analyse…")

	if not os.path.exists(DEFAULT_WEIGHTS):
		raise FileNotFoundError(f"Gewichtsdatei nicht gefunden: {DEFAULT_WEIGHTS}")

	# Konfiguration erstellen (immer CPU)
	device = "cpu"
	cfg = build_cfg_for_inference(device=device)
	setup_logger()  # detectron2 logger

	# Minimalen Dataset-Eintrag registrieren (nur Metadaten/Classes)
	try:
		car_parts_classes = [
			'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
			'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
			'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
			'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
			'tailgate', 'trunk', 'wheel'
		]
		MetadataCatalog.get("car_parts_minimal").set(thing_classes=car_parts_classes)
	except Exception:
		pass

	# Modell direkt bauen (kein DefaultPredictor; LRP arbeitet ohne Gradienten)
	model = build_model(cfg)
	DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
	model.eval()
	model.to(device)

	# LRP Tracer-Konfiguration (nur benötigt, wenn method == 'lrp'). Immer epsilon-Regel.
	lrp_cfg = LRPConfig(rule_linear="epsilon", rule_conv="epsilon", alpha=1.0, beta=0.0, epsilon=lrp_epsilon)
	tracer = LRPTracer(lrp_cfg)

	# Encoder- oder Decoder-Layer finden
	if which_module == "decoder":
		enc_layers = list_decoder_like_layers(model)
		layer_role = "Decoder"
	else:
		enc_layers = list_encoder_like_layers(model)
		layer_role = "Encoder"
	if not enc_layers:
		# Fallback: alle Module mit passenden Keywords listen
		key = "decoder" if which_module == "decoder" else "encoder"
		names = [n for n, _ in model.named_modules() if key in n.lower()]
		raise RuntimeError(
			f"Konnte keine {layer_role}-Layer finden. Gefundene '{key}'-Module: " + ", ".join(names)
		)

	# layer_index als 1-basiert interpretieren (intuitiver für Nutzer)
	if layer_index <= 0 or layer_index > len(enc_layers):
		msg = (
			f"layer_index {layer_index} ungültig. Es gibt {len(enc_layers)} {layer_role}-Kandidaten.\n"
			+ "Gefundene Layer:\n"
			+ "\n".join([f"  {i+1}: {n} ({type(m).__name__})" for i, (n, m) in enumerate(enc_layers)])
		)
		raise IndexError(msg)

	chosen_name, chosen_layer = enc_layers[layer_index - 1]
	logger.info(f"Gewähltes {layer_role}-Layer [{layer_index}]: {chosen_name} ({type(chosen_layer).__name__})")

	# Index-Achse bestimmen: Decoder -> Token, Encoder -> Kanal (wenn auto)
	if index_kind not in ("auto", "channel", "token"):
		raise ValueError("index_kind muss 'auto', 'channel' oder 'token' sein")
	index_axis = (
		("token" if which_module == "decoder" else "channel")
		if index_kind == "auto"
		else index_kind
	)
	logger.info(f"Index-Achse: {index_axis} (index_kind={index_kind})")

	# Bilder sammeln
	img_files = collect_images(images_dir)
	if not img_files:
		raise FileNotFoundError(f"Keine Bilder in {images_dir} gefunden")

	# Aggregation über Bilder
	agg_attr: Tensor | None = None
	processed = 0

	# Vorverarbeiter analog zum DefaultPredictor (feste Werte)
	resize_aug = T.ResizeShortestEdge(short_edge_length=320, max_size=512)

	logger.info(f"Verarbeite {len(img_files)} Bilder aus: {images_dir}")
	logger.debug("Dateiliste:\n" + "\n".join(img_files))

	# Wir registrieren Hooks erst im Bild-Loop
	tmp_tap = None

	for img_path in img_files:
		try:
			# Bild laden (BGR wie in DefaultPredictor, dann ggf. nach RGB drehen)
			pil_im = PIL.Image.open(img_path).convert("RGB")
			original_rgb = np.array(pil_im)  # RGB (H, W, 3)
			original_h, original_w = original_rgb.shape[:2]

			# Falls Modell RGB erwartet, lassen wir RGB; ansonsten drehen nach BGR
			if cfg.INPUT.FORMAT == "RGB":
				model_input = original_rgb
			else:
				model_input = original_rgb[:, :, ::-1]  # RGB -> BGR

			# Resize
			tfm = resize_aug.get_transform(original_rgb)
			model_input = tfm.apply_image(model_input)

			# Tensor (C,H,W)
			image_tensor = torch.as_tensor(model_input.astype("float32").transpose(2, 0, 1))
			image_tensor = image_tensor.to(device)

			# Batch zusammenstellen
			batched_inputs = [{
				"image": image_tensor,
				"height": original_h,
				"width": original_w,
			}]

			# Forward
			if method == "lrp":
				# LRP benötigt keine Gradients; speicherschonend
				with torch.inference_mode():
					# Pass 1: y-only Hooks auf Subtree, um Startmodul und y_start zu bestimmen
					tracer.add_from_module_y_only(chosen_layer)
					tmp_tap = _LayerTap(tracer, chosen_layer)
					_ = model(batched_inputs)
					cache = tracer.store.get(chosen_layer)
					if not cache or ("y" not in cache):
						logger.warning(f"Kein LRP-Cache für Layer-Ausgabe bei Bild: {os.path.basename(img_path)}")
						# Cleanup
						tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
						continue
					y_layer = cache["y"]

					def is_supported(m: nn.Module) -> bool:
						return isinstance(m, (nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU, nn.LayerNorm, nn.BatchNorm2d, nn.MultiheadAttention))

					candidates: List[Tuple[str, nn.Module, Tensor]] = []
					for rel_name, module in chosen_layer.named_modules():
						full_name = chosen_name if rel_name == "" else f"{chosen_name}.{rel_name}"
						if module is chosen_layer:
							continue
						if is_supported(module):
							c = tracer.store.get(module)
							if c and ("y" in c):
								y_c = c["y"]
								candidates.append((full_name, module, y_c))

					# Startmodul immer der gewählte Layer
					start_name, start_module, y_start = chosen_name, chosen_layer, y_layer

					# Pass 1 Cleanup (alle Hooks entfernen, Caches leeren)
					tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None

					# Pass 2: Nur full-Hooks auf dem Startmodul registrieren
					tracer = LRPTracer(lrp_cfg)
					for m in start_module.modules():
						tracer.add_module(m)
					tmp_tap = _LayerTap(tracer, start_module)
					# Zweiter Forward ebenfalls speicherschonend
					with torch.inference_mode():
						_ = model(batched_inputs)

					# Zielrelevanz erstellen und LRP durchführen
					R_out = build_target_relevance(y_start, feature_index, "mean", target_norm, index_axis=index_axis)
					rels = propagate_lrp(model, tracer, start_module, R_out, lrp_cfg)
					R_in = rels.get(start_module)
					if R_in is None:
						logger.warning("LRP lieferte keine Eingangsrelevanz; überspringe Bild.")
						# Cleanup
						tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
						continue
					# Kanäle aggregieren
					attr = aggregate_channel_relevance(R_in)
					# NaN/Inf guard
					if not torch.isfinite(attr).all():
						logger.warning("Nicht-endliche Relevanzwerte detektiert; Bild übersprungen.")
						tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
						continue
					if agg_attr is None:
						agg_attr = attr.clone()
					else:
						agg_attr += attr
					processed += 1
					# Speicher aufräumen
					del y_layer, y_start, R_out, rels, R_in, attr
					tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
					gc.collect()
					if torch.cuda.is_available():
						torch.cuda.empty_cache()
			else:
				# Grad*Input: wir benötigen Autograd für das Startmodul-Eingang
				model.zero_grad(set_to_none=True)
				# Hook auf gewähltem Layer (oder Submodul) mit Grad-Retain
				tmp_tap = _LayerTap(tracer, chosen_layer, retain_grad=True)
				# Klassischer Forward MIT Gradients
				tracer.add_from_module_y_only(chosen_layer)
				out = model(batched_inputs)
				cache = tracer.store.get(chosen_layer)
				if not cache or ("y" not in cache) or ("x" not in cache):
					logger.warning("Kein Cache x/y am gewählten Layer; Bild übersprungen.")
					tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
					continue
				y_layer = cache["y"]
				x_layer = cache["x"]
				# Zielmaskierung nur auf feature_index
				R_out = build_target_relevance(y_layer, feature_index, "mean", target_norm, index_axis=index_axis)
				# Rückwärts: d(y_feature) nach x
				# Wir bauen einen Skalar, indem wir die Zielmaske mit y_layer multiplizieren und summieren
				loss = (R_out * y_layer).sum()
				loss.backward()
				if x_layer.grad is None:
					logger.warning("Kein Grad am Layer-Eingang; Bild übersprungen.")
					tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
					continue
				grad = x_layer.grad.detach()
				attr = aggregate_channel_relevance(grad * x_layer.detach())
				if not torch.isfinite(attr).all():
					logger.warning("Nicht-endliche Grad*Input-Werte; Bild übersprungen.")
					tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
					continue
				if agg_attr is None:
					agg_attr = attr.clone()
				else:
					agg_attr += attr
				processed += 1
				# Cleanup
				tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
				model.zero_grad(set_to_none=True)
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
				# Pass 1: y-only Hooks auf Subtree, um Startmodul und y_start zu bestimmen
				tracer.add_from_module_y_only(chosen_layer)
				tmp_tap = _LayerTap(tracer, chosen_layer)
				_ = model(batched_inputs)
				cache = tracer.store.get(chosen_layer)
				if not cache or ("y" not in cache):
					logger.warning(f"Kein LRP-Cache für Layer-Ausgabe bei Bild: {os.path.basename(img_path)}")
					# Cleanup
					tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
					continue
				y_layer = cache["y"]

				def is_supported(m: nn.Module) -> bool:
					return isinstance(m, (nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU, nn.LayerNorm, nn.BatchNorm2d, nn.MultiheadAttention))

				candidates: List[Tuple[str, nn.Module, Tensor]] = []
				for rel_name, module in chosen_layer.named_modules():
					full_name = chosen_name if rel_name == "" else f"{chosen_name}.{rel_name}"
					if module is chosen_layer:
						continue
					if is_supported(module):
						c = tracer.store.get(module)
						if c and ("y" in c):
							y_c = c["y"]
							candidates.append((full_name, module, y_c))

				# Startmodul immer der gewählte Layer
				start_name, start_module, y_start = chosen_name, chosen_layer, y_layer

				# Pass 1 Cleanup (alle Hooks entfernen, Caches leeren)
				tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None

				# Pass 2: Nur full-Hooks auf dem Startmodul registrieren
				tracer = LRPTracer(lrp_cfg)
				for m in start_module.modules():
					tracer.add_module(m)
				tmp_tap = _LayerTap(tracer, start_module)
				# Zweiter Forward ebenfalls speicherschonend
				with torch.inference_mode():
					_ = model(batched_inputs)

				# Zielrelevanz erstellen und LRP durchführen
				R_out = build_target_relevance(y_start, feature_index, "mean", target_norm, index_axis=index_axis)
				rels = propagate_lrp(model, tracer, start_module, R_out, lrp_cfg)
				R_in = rels.get(start_module)
				if R_in is None:
					logger.warning("LRP lieferte keine Eingangsrelevanz; überspringe Bild.")
					# Cleanup
					tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
					continue
				# Kanäle aggregieren
				attr = aggregate_channel_relevance(R_in)
				# NaN/Inf guard
				if not torch.isfinite(attr).all():
					logger.warning("Nicht-endliche Relevanzwerte detektiert; Bild übersprungen.")
					tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
					continue
				if agg_attr is None:
					agg_attr = attr.clone()
				else:
					agg_attr += attr
				processed += 1
				# Speicher aufräumen
				# Drop large tensors explicitly
				del y_layer, y_start, R_out, rels, R_in, attr
				tracer.remove(); tracer.store.clear(); tmp_tap.remove() if tmp_tap else None
				gc.collect()
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
		except Exception as e:
			logger.exception(f"Fehler bei {img_path}: {e}")
		finally:
			# Drop cached tensors to avoid leakage across images
			tracer.store.clear()

	if tmp_tap:
		tmp_tap.remove()
	tracer.remove()

	if agg_attr is None or processed == 0:
		raise RuntimeError("Keine Attributionen konnten berechnet werden.")

	# Mittelwert über Bilder
	agg_attr = agg_attr / float(processed)

	# Export nach CSV
	df = pd.DataFrame(
		{
			"prev_feature_idx": list(range(len(agg_attr))),
			"relevance": agg_attr.numpy().tolist(),
			"layer_index": layer_index,
			"layer_name": chosen_name,
			"feature_index": feature_index,
			"epsilon": lrp_epsilon,
			"module_role": layer_role,
			"target_norm": target_norm,
			"method": method,
		}
	).sort_values("relevance", ascending=False)

	os.makedirs(os.path.dirname(output_csv), exist_ok=True)
	df.to_csv(output_csv, index=False)

	# Logging der Top-10 Beiträge
	topk = df.head(10)
	logger.info("Top-10 vorherige Features nach Relevanz:")
	for _, row in topk.iterrows():
		logger.info(f"  idx={int(row.prev_feature_idx):4d}  rel={row.relevance:.6f}")

	logger.info(
		f"Fertig. Auswertung über {processed} Bilder. Ergebnis gespeichert in: {output_csv}"
	)


def parse_args() -> argparse.Namespace:
	"""CLI-Parser für rückwärtskompatible Nutzung."""
	parser = argparse.ArgumentParser(description="LRP/Attribution für MaskDINO-Encoder/Decoder")
	parser.add_argument(
		"--images-dir",
		type=str,
		default="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/1images",
		help="Ordner mit Eingabebildern (jpg/png)",
	)
	parser.add_argument(
		"--layer-index",
		type=int,
		default=3,
		help="1-basierter Index des Encoder-/Decoder-Layers (z.B. 3)",
	)
	parser.add_argument(
		"--feature-index",
		type=int,
		default=214,
		help="Kanalindex (Feature) im gewählten Layer (z.B. 235)",
	)
	parser.add_argument(
		"--target-norm",
		type=str,
		default="sum1",
		choices=["sum1", "sumT", "none"],
		help="Norm der Zielrelevanz: sum1 (Summe=1), sumT (Summe=T), none (keine Norm)",
	)
	parser.add_argument(
		"--lrp-epsilon",
		type=float,
		default=1e-6,
		help="Epsilon-Stabilisator für ε/z+",
	)
	parser.add_argument(
		"--output-csv",
		type=str,
		default="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/lrp_result.csv",
		help="Pfad zur Ausgabedatei (CSV)",
	)
	parser.add_argument(
		"--which-module",
		type=str,
		default="encoder",
		choices=["encoder", "decoder"],
		help="Wähle Encoder oder Decoder für die LRP-Analyse",
	)
	parser.add_argument(
		"--method",
		type=str,
		default="lrp",
		choices=["gradinput", "lrp"],
		help="Attributionsmethode: gradinput (Grad*Input am Layer-Eingang) oder lrp (LRP-Regeln)",
	)
	return parser.parse_args()


def main(
	images_dir: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/1images",
	layer_index: int = 3,
	feature_index: int = 214,
	target_norm: str = "sum1",
	lrp_epsilon: float = 1e-6,
	output_csv: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/lrp_result.csv",
	which_module: str = "encoder",
	method: str = "lrp",
):
	"""Programmierbarer Einstiegspunkt mit denselben Parametern wie der CLI-Parser.

	Hinweise:
	- ``layer_index`` ist 1-basiert (wie zuvor in der CLI).
	- ``limit_images``: 0 oder negativ bedeutet alle Bilder (intern wird ``None`` übergeben).
	"""
	logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

	run_analysis(
		images_dir=images_dir,
		layer_index=layer_index,
		feature_index=feature_index,
		output_csv=output_csv,
		target_norm=target_norm,
		lrp_epsilon=lrp_epsilon,
		which_module=which_module,
		method=method,
	)


if __name__ == "__main__":
	# Rückwärtskompatibler CLI-Einstiegspunkt
	_args = parse_args()
	main(
		images_dir=_args.images_dir,
		layer_index=_args.layer_index,
		feature_index=_args.feature_index,
		target_norm=_args.target_norm,
		lrp_epsilon=_args.lrp_epsilon,
		output_csv=_args.output_csv,
		which_module=_args.which_module,
		method=_args.method,
	)
