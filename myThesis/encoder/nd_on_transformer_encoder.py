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

# --- Sicheres TMP-Verzeichnis setzen (fix für tempfile/portalocker) ---
import os
import tempfile

def _ensure_tmpdir():
    """Stellt sicher, dass ein nutzbares TMP-Verzeichnis existiert, bevor
    Bibliotheken wie portalocker/iopath/detectron2 importiert werden.
    """
    try:
        d = tempfile.gettempdir()
        # Schreibtest
        test_path = os.path.join(d, "__tmp_write_test__")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        return  # alles gut
    except Exception:
        pass

    # Fallback: lokales .tmp im Projekt anlegen und als TMPDIR setzen
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    except Exception:
        project_root = os.getcwd()
    fallback = os.path.abspath(os.path.join(project_root, ".tmp"))
    try:
        os.makedirs(fallback, exist_ok=True)
        for env in ("TMPDIR", "TMP", "TEMP"):
            os.environ[env] = fallback
        tempfile.tempdir = fallback
        print(f"⚙️ TMPDIR Fallback aktiviert: {fallback}")
    except Exception as e:
        print(f"⚠️ Konnte Fallback-TMP nicht setzen: {e}")

_ensure_tmpdir()

# --- Standard-Imports ---
import sys
from typing import List
import torch

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

# myThesis-Repo zum Pfad hinzufügen
mythesis_path = "/Users/nicklehmacher/Alles/MasterArbeit"
if mythesis_path not in sys.path:
    sys.path.insert(0, mythesis_path)

# Übergabe-Funktion importieren
# Import der Übergabe-Funktion aus der Nachbar-Datei
from myThesis.encoder.weights_extraction_transformer_encoder import accept_weights_model_images
# Import der zentralen Modell-Konfiguration
from myThesis.model_config import get_model_config, get_classes, get_num_classes, get_dataset_name, get_dataset_root


# --- Dataset-Registrierung ---
def register_datasets(model: str = "car") -> List[str]:
    """Registriert das Dataset für das angegebene Modell (car oder butterfly)."""
    config = get_model_config(model)
    dataset_name = config["dataset_name"]
    dataset_root = config["dataset_root"]
    classes = config["classes"]

    if dataset_name not in DatasetCatalog.list():
        register_coco_instances(
            dataset_name, {},
            os.path.join(dataset_root, config["annotations_train"]),
            os.path.join(dataset_root, config["images_subdir"])
        )

    MetadataCatalog.get(dataset_name).set(thing_classes=classes)
    return classes


# --- Config/Modellaufbau ---
def build_cfg(model: str = "car"):
    """Baut die Konfiguration für das angegebene Modell (car oder butterfly)."""
    num_classes = get_num_classes(model)
    dataset_name = get_dataset_name(model)
    
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
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes
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
    # Angepasst für 256x256 Eingabebilder (keine Skalierung nötig)
    cfg.INPUT.MIN_SIZE_TEST = 256
    cfg.INPUT.MAX_SIZE_TEST = 256

    cfg.DATASETS.TRAIN = (dataset_name,)

    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.8
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.25
    cfg.MODEL.MaskDINO.TEST.SCORE_THRESH_TEST = 0.5

    cfg.MODEL.DEVICE = "cpu"
    return cfg


def build_model_and_load_weights(weights_path: str, model: str = "car") -> torch.nn.Module:
    """Baut das Modell und lädt die angegebenen Gewichte."""
    classes = register_datasets(model)
    cfg = build_cfg(model)
    dataset_name = get_dataset_name(model)
    assert len(MetadataCatalog.get(dataset_name).thing_classes) == cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, \
        "NUM_CLASSES ≠ Anzahl Labels in Dataset"
    cfg.MODEL.WEIGHTS = ""
    cfg.freeze()

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(weights_path)
    return model


def gather_images(image_dir: str) -> List[str]:
    """Stellt eine Liste von Bildpfaden zusammen aus dem angegebenen Verzeichnis."""
    if not os.path.exists(image_dir):
        print(f"⚠️ Verzeichnis nicht gefunden: {image_dir}")
        return []
    # Alle Bilddateien aus dem Verzeichnis sammeln
    image_files = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            full_path = os.path.join(image_dir, filename)
            image_files.append(full_path)
    print(f"📁 Gefundene Bilder in {image_dir}: {len(image_files)}")
    return sorted(image_files)  # Sortiert für konsistente Reihenfolge


def main(
    images_dir: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/rot",
    weights_path: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_final.pth",
    output_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output", "encoder")),
    model: str = "car",
):
    setup_logger(name="maskdino")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Gewichte nicht gefunden: {weights_path}")

    # Ausgabeordner vorbereiten
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔧 Baue Modell ({model}) und lade Gewichte…")
    model_nn = build_model_and_load_weights(weights_path, model)

    print("🖼️  Sammle Bilder…")
    image_list = gather_images(images_dir)
    if not image_list:
        # Für den Start tolerieren wir eine leere Liste, aber melden es sichtbar
        print("⚠️ Keine Bilder gefunden – Übergabe erfolgt mit leerer Liste.")

    print("➡️  Übergabe an weights_extraction_transformer_encoder.accept_weights_model_images …")
    accept_weights_model_images(weights_path, model_nn, image_list, base_out_layers=output_dir)
    print("✅ Übergabe abgeschlossen.")


if __name__ == "__main__":
    main()