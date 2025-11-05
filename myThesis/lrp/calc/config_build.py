"""
Model configuration builder and constants for MaskDINO inference.
Includes DEFAULT_WEIGHTS and add_maskdino_config wiring.
"""
from __future__ import annotations

import os
import sys

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog  # Imported for potential dataset metadata wiring

# Make MaskDINO importable (as in original script)
MASKDINO_PATH = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO"
if MASKDINO_PATH not in sys.path:
    sys.path.insert(0, MASKDINO_PATH)

try:
    from maskdino import add_maskdino_config  # type: ignore
except Exception as e:  # pragma: no cover - defensive
    raise RuntimeError(
        f"MaskDINO konnte nicht importiert werden: {e}. Prüfe Pfad {MASKDINO_PATH}"
    )

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
