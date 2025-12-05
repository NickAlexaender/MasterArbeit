"""
Vorbereitung für Network Dissection auf dem Transformer Decoder in MaskDINO.

Dieses Modul übernimmt ausschließlich:
- Aufbau der MaskDINO-Konfiguration und Laden der finetuned Gewichte
- Bereitstellung einer Bildliste (Beispieleingaben)
- Auffinden des Transformer-Decoders im Modell und Ausgabe einer Kurz-Zusammenfassung
- Aufruf von (noch zu definierenden) Hook-/Analyse-Funktionen für den Decoder

Wichtig: Die eigentliche Network-Dissection-Logik ist hier NICHT implementiert.
Die folgenden Funktionen werden bewusst referenziert, aber nicht definiert –
damit du sie separat implementieren kannst:

- attach_decoder_hooks(decoder) -> hook_handles
- detach_decoder_hooks(hook_handles) -> None
- run_network_dissection_on_decoder(model, decoder, image_list, classes, weights_path) -> Any

Siehe Kommentare weiter unten für eine knappe „Contract“-Beschreibung.
"""

# --- Kompatibilitätsfixe (Pillow/NumPy) ---
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

# --- Standard-Imports ---
import os
import sys
from typing import List, Tuple
import torch
import argparse

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

# MaskDINO-Repo binden
maskdino_path = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO"
if maskdino_path not in sys.path:
	sys.path.insert(0, maskdino_path)
from maskdino import add_maskdino_config

# myThesis-Repo binden
mythesis_path = "/Users/nicklehmacher/Alles/MasterArbeit"
if mythesis_path not in sys.path:
	sys.path.insert(0, mythesis_path)

# Übergabe-Funktion importieren (Baustein 2)
# Import der Übergabe-Funktion aus der Nachbar-Datei
from myThesis.decoder.weights_extraction_transformer_decoder import accept_weights_model_images


# --- Dataset-Registrierung ---
def register_datasets() -> List[str]:
	dataset_root = "/Users/nicklehmacher/Alles/MasterArbeit/ultralytics/datasets"

	if "car_parts_train" not in DatasetCatalog.list():
		register_coco_instances(
			"car_parts_train", {},
			os.path.join(dataset_root, "annotations", "instances_train2017.json"),
			os.path.join(dataset_root, "images")
		)

	car_parts_classes = [
		'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
		'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
		'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
		'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
		'tailgate', 'trunk', 'wheel'
	]
	MetadataCatalog.get("car_parts_train").set(thing_classes=car_parts_classes)
	return car_parts_classes


# --- Config/Modellaufbau ---
def build_cfg():
	cfg = get_cfg()
	add_maskdino_config(cfg)

	# Backbone/Head weitgehend wie in bestehender Analyse eingestellt
	cfg.MODEL.META_ARCHITECTURE = "MaskDINO"
	cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
	cfg.MODEL.BACKBONE.FREEZE_AT = 0
	cfg.MODEL.RESNETS.DEPTH = 50
	cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
	cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
	cfg.MODEL.RESNETS.NORM = "FrozenBN"
	cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
	cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
	cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]

	cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
	cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

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

	cfg.INPUT.FORMAT = "RGB"
	cfg.INPUT.MIN_SIZE_TEST = 800
	cfg.INPUT.MAX_SIZE_TEST = 1333

	cfg.DATASETS.TRAIN = ("car_parts_train",)

	cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
	cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
	cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
	cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.8
	cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.25
	cfg.MODEL.MaskDINO.TEST.SCORE_THRESH_TEST = 0.5

	cfg.MODEL.DEVICE = "cpu"
	return cfg


# --- Hilfsfunktionen für Decoder-Suche & -Zusammenfassung ---
def _count_parameters(module: torch.nn.Module) -> int:
	return sum(p.numel() for p in module.parameters() if p is not None)


def find_transformer_decoder(model: torch.nn.Module) -> torch.nn.Module:
	"""
	Versucht, den Transformer-Decoder in einem MaskDINO-Modell zu finden.

	Heuristik:
	- Suche nach Modulen, deren Klassenname "Decoder" enthält und (bevorzugt)
	  auch "Mask" oder "DINO" oder "Transformer" im Namen hat.
	- Falls mehrere Kandidaten existieren, wähle den mit den meisten Parametern.

	Raises:
		RuntimeError bei Nichterfolg mit Hinweisen zur manuellen Pfadangabe.
	"""
	candidates = []
	for name, module in model.named_modules():
		cls = module.__class__.__name__
		cls_l = cls.lower()
		name_l = name.lower()
		if "decoder" in cls_l or "decoder" in name_l:
			score = 0
			if "transformer" in cls_l or "transformer" in name_l:
				score += 2
			if "mask" in cls_l or "mask" in name_l:
				score += 1
			if "dino" in cls_l or "dino" in name_l:
				score += 1
			params = _count_parameters(module)
			candidates.append((score, params, name, module))

	if not candidates:
		# Optional: gezielte Pfade versuchen (best guess; je nach MaskDINO-Version):
		try:
			maybe = getattr(model, "sem_seg_head", None)
			if maybe is not None:
				# Häufig liegt der Decoder innerhalb des HEADs
				for attr in ["transformer_decoder", "decoder", "transformer", "predictor"]:
					if hasattr(maybe, attr):
						mod = getattr(maybe, attr)
						if isinstance(mod, torch.nn.Module) and "decoder" in mod.__class__.__name__.lower():
							return mod
		except Exception:
			pass

		raise RuntimeError(
			"Konnte keinen Transformer-Decoder automatisch finden. "
			"Bitte passe die Heuristik in find_transformer_decoder() an oder gib den Pfad manuell an."
		)

	# Beste/r Kandidat/en nach Score und Parametern wählen
	candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
	best = candidates[0]
	_, _, best_name, best_module = best
	print(f"🔎 Decoder-Kandidat gewählt: {best_name} :: {best_module.__class__.__name__}")
	return best_module


def print_decoder_summary(decoder: torch.nn.Module, max_children: int = 20) -> None:
	"""
	Gibt eine kurze Zusammenfassung des gefundenen Decoders aus.
	"""
	total_params = _count_parameters(decoder)
	print("—" * 60)
	print(f"Decoder: {decoder.__class__.__name__}")
	print(f"Parameter: {total_params:,}")
	print(f"Train/Eval-Modus: {'train' if decoder.training else 'eval'}")
	print("Direkte Kinder:")
	for i, (n, m) in enumerate(decoder.named_children()):
		if i >= max_children:
			print("  … (gekürzt)")
			break
		print(f"  - {n}: {m.__class__.__name__}")
	print("—" * 60)


def build_model_and_load_weights(weights_path: str) -> Tuple[torch.nn.Module, List[str]]:
	"""Baut das Modell und lädt die angegebenen Gewichte. Gibt (model, classes) zurück."""
	classes = register_datasets()
	cfg = build_cfg()
	assert len(MetadataCatalog.get("car_parts_train").thing_classes) == cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, \
		"NUM_CLASSES ≠ Anzahl Labels in Dataset"
	cfg.MODEL.WEIGHTS = ""
	cfg.freeze()

	model = build_model(cfg)
	model.eval()
	DetectionCheckpointer(model).load(weights_path)
	return model, classes


def gather_images(image_dir: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/rot") -> List[Tuple[str, str]]:
	"""
	Stellt eine Liste von (image_id, image_path) Tupeln zusammen.
	image_id wird aus dem Dateinamen extrahiert (z.B. "image 1.jpg" -> "image_1")
	"""
	candidates = []
	
	# Durchsuche das rot-Verzeichnis nach Bilddateien
	if os.path.exists(image_dir):
		for filename in os.listdir(image_dir):
			if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
				full_path = os.path.join(image_dir, filename)
				# Extrahiere image_id aus Dateinamen (ohne Erweiterung, Leerzeichen durch _ ersetzen)
				image_id = os.path.splitext(filename)[0].replace(" ", "_")
				candidates.append((image_id, full_path))
	
	return sorted(candidates, key=lambda x: x[1])


# --- Vertragsbeschreibung (für zu implementierende Funktionen) ---
# Erwartete Signaturen (NICHT hier definieren):
#
# def attach_decoder_hooks(decoder: torch.nn.Module):
#     """Registriert Forward-/Backward-Hooks am Decoder und gibt Handles zurück."""
#     ...
#
# def detach_decoder_hooks(hook_handles):
#     """Hebt alle registrierten Hooks wieder auf."""
#     ...
#
# def run_network_dissection_on_decoder(
#     model: torch.nn.Module,
#     decoder: torch.nn.Module,
#     image_list: List[str],
#     classes: List[str],
#     weights_path: str,
# ):
#     """Führt die ND-Analyse für den Decoder durch (E2E, inkl. Forward-Pässe)."""
#     ...


def _require(name: str) -> None:
	if name not in globals():
		raise NotImplementedError(
			f"Bitte implementiere `{name}(...)` in diesem Modul oder importiere sie explizit. "
			f"Siehe die Vertragsbeschreibung oben für die erwartete Signatur."
		)


def main(    
    images_dir: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/rot",
    weights_path: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_final.pth",
    output_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "encoder")),
):
	setup_logger(name="maskdino")
 
	if not os.path.exists(weights_path):
		raise FileNotFoundError(f"Gewichte nicht gefunden: {weights_path}")

	print("🔧 Baue Modell und lade Gewichte…")
	model, classes = build_model_and_load_weights(weights_path)

	print("🖼️  Sammle Bilder…")
	image_list = gather_images(images_dir)
	if not image_list:
		# Für den Start tolerieren wir eine leere Liste, aber melden es sichtbar
		print("⚠️ Keine Bilder gefunden – Analyse erfolgt ggf. mit leerer Liste.")

	# Decoder suchen und kurze Zusammenfassung ausgeben
	print("🔎 Suche Transformer-Decoder…")
	decoder = find_transformer_decoder(model)
	print_decoder_summary(decoder)

	# Sicherstellen, dass die Analyse-Hooks implementiert werden (dynamisch auflösen)
	#for fn in ("attach_decoder_hooks", "detach_decoder_hooks", "run_network_dissection_on_decoder"):
	#	try:
	#		_require(fn)
	#	except NotImplementedError as e:
	#		# Frühes, klares Feedback für die nächste Implementationsphase
	#		print(f"❗ {e}")
	#		print("Beende, da die erforderlichen ND-Funktionen noch fehlen.")
	#		return

	#fn_attach = globals().get("attach_decoder_hooks")
	# fn_detach = globals().get("detach_decoder_hooks")
 
	print(f"📁 Ausgabeziel: {output_dir}")
	accept_weights_model_images(weights_path, model, image_list, output_dir=output_dir)

	print("🪝 Registriere Decoder-Hooks…")
	#hook_handles = fn_attach(decoder)  # type: ignore[operator]
	#try:
	#	print("🧪 Starte Network Dissection auf dem Decoder…")
	#	fn_run(  # type: ignore[misc]
	#		model=model,
	#		decoder=decoder,
	#		image_list=image_list,
	#		classes=classes,
	#		weights_path=weights_path,
	#	)
	#	print("✅ ND-Lauf abgeschlossen.")
	#finally:
	#	print("🧹 Löse Decoder-Hooks…")
	#	fn_detach(hook_handles)  # type: ignore[misc]


if __name__ == "__main__":
	main()

