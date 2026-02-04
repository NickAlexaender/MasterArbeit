# Vorbereitung des Transformer Encoder in MaskDINO auf Network Dissection.


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
except Exception:
    pass

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
except Exception:
    pass

# --- Sicheres TMP-Verzeichnis ---
import os
import tempfile

def _ensure_tmpdir():
    try:
        d = tempfile.gettempdir()
        test_path = os.path.join(d, "__tmp_write_test__")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        return
    except Exception:
        pass

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
    except Exception:
        pass

_ensure_tmpdir()

# --- Standard-Imports ---
import sys
from typing import List, Optional, Tuple
import torch

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

# MaskDINO-Repo
maskdino_path = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO"
if maskdino_path not in sys.path:
    sys.path.insert(0, maskdino_path)
from maskdino import add_maskdino_config

# myThesis-Repo
mythesis_path = "/Users/nicklehmacher/Alles/MasterArbeit"
if mythesis_path not in sys.path:
    sys.path.insert(0, mythesis_path)

from myThesis.model_config import get_model_config, get_num_classes, get_dataset_name
from myThesis.linear_probing.linear_probing_extraction import extract_encoder_features_with_labels


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-Registrierung
# ─────────────────────────────────────────────────────────────────────────────

def register_datasets(model: str = "butterfly") -> List[str]:
    """Registriert das Dataset für das angegebene Modell."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Config/Modellaufbau
# ─────────────────────────────────────────────────────────────────────────────

def build_cfg(model: str = "butterfly"):
    """Baut die Konfiguration für das angegebene Modell."""
    num_classes = get_num_classes(model)
    dataset_name = get_dataset_name(model)
    
    cfg = get_cfg()
    add_maskdino_config(cfg)

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


def build_model_and_load_weights(weights_path: str, model: str = "butterfly") -> torch.nn.Module:
    """Baut das Modell und lädt die Gewichte."""
    classes = register_datasets(model)
    cfg = build_cfg(model)
    dataset_name = get_dataset_name(model)
    assert len(MetadataCatalog.get(dataset_name).thing_classes) == cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, \
        "NUM_CLASSES ≠ Anzahl Labels in Dataset"
    cfg.MODEL.WEIGHTS = ""
    cfg.freeze()

    model_nn = build_model(cfg)
    model_nn.eval()
    DetectionCheckpointer(model_nn).load(weights_path)
    return model_nn


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen für Farbkonzept-Masken
# ─────────────────────────────────────────────────────────────────────────────

def get_color_concept_paths(model: str, image_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Ermittelt die Pfade zu den Farbkonzept-Masken für ein Bild.
    
    Args:
        model: "butterfly" oder "car"
        image_id: Bild-ID (ohne Extension)
        
    Returns:
        (grau_mask_path, orange_mask_path, blau_mask_path) - None wenn nicht vorhanden
    """
    base_path = f"/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/{model}"
    
    # Prüfe verschiedene Dateierweiterungen
    extensions = [".png", ".jpg", ".jpeg"]
    
    grau_path = None
    for ext in extensions:
        candidate = os.path.join(base_path, "grau", f"{image_id}{ext}")
        if os.path.exists(candidate):
            grau_path = candidate
            break
    
    orange_path = None
    for ext in extensions:
        candidate = os.path.join(base_path, "orange", f"{image_id}{ext}")
        if os.path.exists(candidate):
            orange_path = candidate
            break
    
    blau_path = None
    for ext in extensions:
        candidate = os.path.join(base_path, "blau", f"{image_id}{ext}")
        if os.path.exists(candidate):
            blau_path = candidate
            break
    
    return grau_path, orange_path, blau_path


def gather_images_with_masks(images_dir: str, model: str) -> List[Tuple[str, str, Optional[str], Optional[str], Optional[str]]]:
    """
    Sammelt Bilder mit zugehörigen Farbkonzept-Masken.
    
    Returns:
        Liste von (image_path, image_id, grau_mask_path, orange_mask_path, blau_mask_path)
    """
    if not os.path.exists(images_dir):
        print(f"⚠️ Verzeichnis nicht gefunden: {images_dir}")
        return []
    
    results = []
    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        
        image_path = os.path.join(images_dir, filename)
        image_id = os.path.splitext(filename)[0]
        
        grau_path, orange_path, blau_path = get_color_concept_paths(model, image_id)
        results.append((image_path, image_id, grau_path, orange_path, blau_path))
    
    print(f"📁 Gefundene Bilder: {len(results)}")
    masks_found = sum(1 for _, _, g, o, b in results if g or o or b)
    print(f"🎨 Bilder mit Farbkonzept-Masken: {masks_found}")
    
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Hauptfunktion
# ─────────────────────────────────────────────────────────────────────────────

def main(
    images_dir: str,
    weights_path: str,
    output_dir: str,
    model: str = "butterfly",
    layer: Optional[str] = None,
):
    """
    Hauptfunktion für Linear Probing Feature-Extraktion auf dem Encoder.
    
    Args:
        images_dir: Verzeichnis mit Eingabebildern
        weights_path: Pfad zu den Modell-Gewichten (.pth)
        output_dir: Ausgabeverzeichnis für CSVs
        model: Modelltyp ("butterfly" oder "car")
        layer: Optional - nur dieses Layer extrahieren (z.B. "layer1")
    """
    setup_logger(name="maskdino")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Gewichte nicht gefunden: {weights_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Alte CSVs entfernen
    for root, _, files in os.walk(output_dir):
        for fn in files:
            if fn.lower() == "patches.csv":
                try:
                    os.remove(os.path.join(root, fn))
                except Exception:
                    pass
    
    print(f"🔧 Baue Modell ({model}) und lade Gewichte…")
    model_nn = build_model_and_load_weights(weights_path, model)
    
    print("🖼️  Sammle Bilder mit Farbkonzept-Masken…")
    image_list = gather_images_with_masks(images_dir, model)
    
    if not image_list:
        print("⚠️ Keine Bilder gefunden.")
        return
    
    print(f"🚀 Starte Encoder Linear Probing Extraktion für {len(image_list)} Bilder…")
    
    success_count = 0
    for image_path, image_id, grau_mask, orange_mask, blau_mask in image_list:
        print(f"\n📷 Verarbeite: {image_id}")
        
        success = extract_encoder_features_with_labels(
            model=model_nn,
            image_path=image_path,
            image_id=image_id,
            grau_mask_path=grau_mask,
            orange_mask_path=orange_mask,
            output_dir=output_dir,
            layer=layer,
            blau_mask_path=blau_mask,
        )
        
        if success:
            success_count += 1
    
    print(f"\n✅ Encoder Linear Probing abgeschlossen: {success_count}/{len(image_list)} erfolgreich")
    print(f"📂 Output: {output_dir}")


if __name__ == "__main__":
    # Beispiel-Aufruf für Butterfly-Dataset
    main(
        images_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/butterfly/1images",
        weights_path="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/weights/butterfly_finetuned.pth",
        output_dir="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/linear_probing/encoder",
        model="butterfly",
    )
