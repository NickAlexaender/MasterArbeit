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
    print("✅ PIL compatibility fixed")
except Exception as e:
    print(f"⚠️ PIL fix failed, but continuing: {e}")

import torch
import os
import sys
import traceback
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# MaskDINO zum Python-Pfad hinzufügen
sys.path.insert(0, "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO")

# Importiere MaskDINO-spezifische Module
from maskdino import add_maskdino_config

# Bild laden
image_path = "/Users/nicklehmacher/Alles/MasterArbeit/image/Gemini_Generated_Image_bytbzbytbzbytbzb.png"
im = cv2.imread(image_path)
if im is None:
    raise FileNotFoundError(f"❌ Bild nicht gefunden: {image_path}")

# Bild verkleinern für CPU-Inferenz (um Speicherproblem zu vermeiden)
print(f"📏 Original Bildgröße: {im.shape}")
# Verkleinere auf max 800 Pixel Breite
height, width = im.shape[:2]
if width > 800:
    scale = 800 / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    im = cv2.resize(im, (new_width, new_height))
    print(f"📏 Verkleinertes Bild: {im.shape}")

im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Konfiguration laden
cfg = get_cfg()
add_maskdino_config(cfg)

# Base-Konfiguration laden (ohne die problematische STEM_TYPE)
try:
    config_path = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"
    
    # Versuche YAML-Datei zu laden und problematische Schlüssel zu entfernen
    import yaml
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Entferne _BASE_ um rekursive Abhängigkeiten zu vermeiden
    if '_BASE_' in yaml_config:
        print("⚠️ Removing _BASE_ dependency to avoid recursive loading")
        del yaml_config['_BASE_']
    
    # Entferne problematische Schlüssel
    if 'MODEL' in yaml_config and 'RESNETS' in yaml_config['MODEL']:
        if 'STEM_TYPE' in yaml_config['MODEL']['RESNETS']:
            print("⚠️ Removing deprecated STEM_TYPE from config")
            del yaml_config['MODEL']['RESNETS']['STEM_TYPE']
    
    # Füge Base-Konfiguration manuell hinzu
    if 'MODEL' not in yaml_config:
        yaml_config['MODEL'] = {}
    
    # Backbone-Einstellungen
    yaml_config['MODEL']['BACKBONE'] = {
        'FREEZE_AT': 0,
        'NAME': 'build_resnet_backbone'
    }
    yaml_config['MODEL']['WEIGHTS'] = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
    yaml_config['MODEL']['PIXEL_MEAN'] = [123.675, 116.280, 103.530]
    yaml_config['MODEL']['PIXEL_STD'] = [58.395, 57.120, 57.375]
    
    # ResNet-Einstellungen
    if 'RESNETS' not in yaml_config['MODEL']:
        yaml_config['MODEL']['RESNETS'] = {}
    yaml_config['MODEL']['RESNETS'].update({
        'DEPTH': 50,
        'STEM_OUT_CHANNELS': 64,
        'STRIDE_IN_1X1': False,
        'OUT_FEATURES': ["res2", "res3", "res4", "res5"],
        'RES5_MULTI_GRID': [1, 1, 1]
    })
    
    # Dataset-Einstellungen
    yaml_config['DATASETS'] = {
        'TRAIN': ("coco_2017_train",),
        'TEST': ("coco_2017_val",)
    }
    
    # Input-Einstellungen
    yaml_config['INPUT'] = {
        'FORMAT': 'RGB',
        'MIN_SIZE_TRAIN': (800,),
        'MAX_SIZE_TRAIN': 1333,
        'MIN_SIZE_TEST': 800,
        'MAX_SIZE_TEST': 1333
    }
    
    # Erstelle temporäre bereinigte Konfigurationsdatei
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        yaml.dump(yaml_config, temp_file)
        temp_config_path = temp_file.name
    
    # Lade die bereinigte Konfiguration
    cfg.merge_from_file(temp_config_path)
    print("✅ Configuration file loaded successfully (with fixes)")
    
    # Entferne temporäre Datei
    os.unlink(temp_config_path)
    
except Exception as e:
    print(f"⚠️ Configuration file loading failed: {e}")
    print("   → Will use manual configuration")
    
    # Manuelle Konfiguration (vollständig und korrekt)
    cfg.MODEL.META_ARCHITECTURE = "MaskDINO"
    
    # Backbone-Konfiguration
    cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.RESNETS.NORM = "FrozenBN"
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
    cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False
    cfg.MODEL.RESNETS.RES5_MULTI_GRID = [1, 1, 1]
    
    # Pixel Mean/Std für Normalization
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]
    
    # SEM_SEG_HEAD Konfiguration (wichtig für MaskDINO)
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskDINOHead"
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 80
    cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.NORM = "GN"
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "MaskDINOEncoder"
    cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024  # Entspricht der Gewichtsdatei hid1024
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3  # Standard 3 Feature Levels
    cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 3  # Entspricht der YAML-Konfiguration
    cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6
    
    # MaskDINO-spezifische Einstellungen (angepasst an die Gewichte)
    cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME = "MaskDINODecoder"
    cfg.MODEL.MaskDINO.DEEP_SUPERVISION = True
    cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MaskDINO.CLASS_WEIGHT = 4.0
    cfg.MODEL.MaskDINO.MASK_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.DICE_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.BOX_WEIGHT = 5.0
    cfg.MODEL.MaskDINO.GIOU_WEIGHT = 2.0
    cfg.MODEL.MaskDINO.HIDDEN_DIM = 256  # Standard Hidden Dimension
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 300  # 300 Object Queries
    cfg.MODEL.MaskDINO.NHEADS = 8
    cfg.MODEL.MaskDINO.DROPOUT = 0.0
    cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = 2048  # Gewichte erwarten 2048 (nicht 1024)
    cfg.MODEL.MaskDINO.ENC_LAYERS = 0
    cfg.MODEL.MaskDINO.PRE_NORM = False
    cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ = False
    cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY = 32
    cfg.MODEL.MaskDINO.DEC_LAYERS = 3  # 3sd1 bedeutet 3 Scale Deformable mit 1 Layer (nicht 9)
    cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS = 12544
    cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO = 0.75
    cfg.MODEL.MaskDINO.INITIAL_PRED = True
    cfg.MODEL.MaskDINO.TWO_STAGE = True
    cfg.MODEL.MaskDINO.DN = "seg"
    cfg.MODEL.MaskDINO.DN_NUM = 100
    cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE = "bitmask"
    
    # Dataset-Einstellungen
    cfg.DATASETS.TRAIN = ("coco_2017_train",)
    cfg.DATASETS.TEST = ("coco_2017_val",)
    
    # Test-Einstellungen entsprechend der YAML-Konfiguration
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.8
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.25  # Entspricht YAML (0.25 statt 0.8)
    cfg.MODEL.MaskDINO.TEST.SCORE_THRESH_TEST = 0.5
    
    # Input-Format
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    print("✅ Manual configuration applied with all required parameters")

# Lokale Gewichtungen verwenden (angepasster Pfad)
weights_path = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO/weights/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"
cfg.MODEL.WEIGHTS = weights_path
cfg.MODEL.DEVICE = "cpu"  # CPU-only

# Konfidenzschwelle für Instanzsegmentierung (angepasst an bessere Gewichte)
cfg.MODEL.MaskDINO.TEST.SCORE_THRESH_TEST = 0.25  # Niedrigere Schwelle für bessere Detektion

print("🔧 Konfiguration geladen")
print(f"📋 Anzahl Klassen: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
print(f"🔍 Hidden Dimension: {cfg.MODEL.MaskDINO.HIDDEN_DIM}")
print(f"❓ Object Queries: {cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES}")
print(f"🏗️ Decoder Layers: {cfg.MODEL.MaskDINO.DEC_LAYERS}")

# Predictor erstellen und Vorhersage durchführen
try:
    predictor = DefaultPredictor(cfg)
    print("✅ Predictor erfolgreich erstellt")
    
    outputs = predictor(im_rgb)
    print("✅ Vorhersage durchgeführt")
    
    # Visualisieren
    metadata = MetadataCatalog.get("coco_2017_val")  # Explizit COCO-Metadaten verwenden
    v = Visualizer(im_rgb[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Speichern
    output_path = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO/output_airplane_car.jpg"
    cv2.imwrite(output_path, out.get_image()[:, :, ::-1])
    print(f"✅ Segmentiertes Bild gespeichert als {output_path}")
    
    # Statistiken ausgeben
    instances = outputs["instances"]
    print(f"🎯 {len(instances)} Objekte erkannt")
    if len(instances) > 0:
        print(f"🏷️ Erkannte Klassen: {instances.pred_classes.tolist()}")
        print(f"📊 Konfidenzwerte: {[f'{score:.3f}' for score in instances.scores.tolist()]}")
        
        # Klassennamen ausgeben
        class_names = metadata.thing_classes
        detected_names = [class_names[cls_id] for cls_id in instances.pred_classes.tolist()]
        print(f"📝 Erkannte Objekte: {detected_names}")
    
except Exception as e:
    print(f"❌ Fehler beim Ausführen der Vorhersage: {e}")
    import traceback
    traceback.print_exc()