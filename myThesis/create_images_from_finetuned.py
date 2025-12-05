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

# NumPy Kompatibilität Fix für neuere NumPy-Versionen
try:
    import numpy as np
    # Fix für neuere NumPy-Versionen (np.bool wurde deprecated)
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

import torch
import os
import sys
import traceback
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import cv2

# MaskDINO zum Python-Pfad hinzufügen
sys.path.insert(0, "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO")

# Importiere MaskDINO-spezifische Module
try:
    from maskdino import add_maskdino_config
    print("✅ MaskDINO modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MaskDINO modules: {e}")
    print("   → Make sure MaskDINO is properly installed")
    print("   → Check if the path '/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO' exists")
    sys.exit(1)

def register_car_parts_datasets():
    """
    Registriere die Car Parts Datasets für das Fine-tuned Modell
    """
    # Pfade zu den COCO-Annotations und Bildern
    dataset_root = "/Users/nicklehmacher/Alles/MasterArbeit/ultralytics/datasets"
    
    # Register validation dataset (für Metadaten)
    try:
        register_coco_instances(
            "car_parts_val", 
            {},
            os.path.join(dataset_root, "annotations", "instances_val2017.json"),
            os.path.join(dataset_root, "images")
        )
        
        # Setze Class Names für Car Parts (23 Klassen)
        car_parts_classes = [
            'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
            'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
            'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
            'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
            'tailgate', 'trunk', 'wheel'
        ]
        
        MetadataCatalog.get("car_parts_val").set(thing_classes=car_parts_classes)
        print(f"✅ Car Parts Dataset registriert mit {len(car_parts_classes)} Klassen")
        return car_parts_classes
    except Exception as e:
        print(f"⚠️ Car Parts Dataset bereits registriert oder Fehler: {e}")
        return [
            'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
            'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
            'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
            'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
            'tailgate', 'trunk', 'wheel'
        ]

def setup_config(model_type="pretrained"):
    """
    Setup MaskDINO Konfiguration basierend auf test_maskdino.py
    model_type: "pretrained" oder "finetuned"
    """
    cfg = get_cfg()
    add_maskdino_config(cfg)
    
    # Manuelle Konfiguration - EXAKT wie in test_maskdino.py
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
    
    # Anzahl Klassen je nach Modell-Typ
    if model_type == "finetuned":
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 23  # Car Parts Klassen
    else:
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 80  # COCO Klassen
    
    # MaskDINO-spezifische Einstellungen (angepasst an die Gewichte) - EXAKT wie in test_maskdino.py
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
    
    # Dataset-Einstellungen je nach Modell-Typ
    if model_type == "finetuned":
        cfg.DATASETS.TRAIN = ("car_parts_train",)
        cfg.DATASETS.TEST = ("car_parts_val",)
    else:
        cfg.DATASETS.TRAIN = ("coco_2017_train",)
        cfg.DATASETS.TEST = ("coco_2017_val",)
    
    # Test-Einstellungen entsprechend der YAML-Konfiguration - EXAKT wie in test_maskdino.py
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.8
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.25  # Entspricht YAML (0.25 statt 0.8)
    cfg.MODEL.MaskDINO.TEST.SCORE_THRESH_TEST = 0.25  # Niedrigere Schwelle für bessere Detektion
    
    # Input-Format - EXAKT wie in test_maskdino.py
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    cfg.MODEL.DEVICE = "cpu"  # CPU-only
    
    return cfg

def process_image_with_model(image_path, weights_path, model_type, output_suffix):
    """
    Verarbeite ein Bild mit einem spezifischen Modell
    """
    print(f"\n🔧 Verarbeite mit {model_type} Modell...")
    print(f"📁 Weights: {os.path.basename(weights_path)}")
    
    # Bild laden
    im = cv2.imread(image_path)
    if im is None:
        raise FileNotFoundError(f"❌ Bild nicht gefunden: {image_path}")
    
    # Bild verkleinern für CPU-Inferenz (um Speicherproblem zu vermeiden)
    print(f"📏 Original Bildgröße: {im.shape}")
    height, width = im.shape[:2]
    if width > 800:
        scale = 800 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        im = cv2.resize(im, (new_width, new_height))
        print(f"📏 Verkleinertes Bild: {im.shape}")
    
    im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    
    # Konfiguration setup
    cfg = setup_config(model_type)
    cfg.MODEL.WEIGHTS = weights_path
    
    print(f"📋 Anzahl Klassen: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"🎯 Score Threshold: {cfg.MODEL.MaskDINO.TEST.SCORE_THRESH_TEST}")
    
    try:
        # Predictor erstellen und Vorhersage durchführen
        predictor = DefaultPredictor(cfg)
        print("✅ Predictor erfolgreich erstellt")
        
        outputs = predictor(im_rgb)
        print("✅ Vorhersage durchgeführt")
        
        # Passende Metadaten wählen
        if model_type == "finetuned":
            metadata = MetadataCatalog.get("car_parts_val")
        else:
            metadata = MetadataCatalog.get("coco_2017_val")
        
        # Visualisieren
        v = Visualizer(im_rgb[:, :, ::-1], metadata, scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Output-Pfad generieren - Benenne nach dem Modell, nicht nach dem Bild
        output_dir = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/model_comparison_outputs"
        output_path = os.path.join(output_dir, f"{output_suffix}.jpg")
        
        # Speichern
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
        
        return output_path, len(instances)
        
    except Exception as e:
        print(f"❌ Fehler beim Ausführen der Vorhersage: {e}")
        import traceback
        traceback.print_exc()
        return None, 0

def main():
    """
    Hauptfunktion - Verarbeite Bild mit allen 7 Modellen
    """
    print("🚀 MaskDINO Modell-Vergleich - 7 Modelle")
    print("=" * 60)
    
    # Eingabebild
    image_path = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/new_21_png_jpg.rf.d0c9323560db430e693b33b36cb84c3b.jpg"
    
    # Überprüfe ob Bild existiert
    if not os.path.exists(image_path):
        print(f"❌ Eingabebild nicht gefunden: {image_path}")
        return
    
    print(f"📸 Verarbeite Bild: {os.path.basename(image_path)}")
    
    # Erstelle Output-Ordner
    output_dir = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/model_comparison_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output-Ordner erstellt: {output_dir}")
    
    # Registriere Car Parts Datasets für Fine-tuned Modell
    register_car_parts_datasets()
    
    # Modell-Konfigurationen - Alle 7 Modelle
    models = [
        {
            "name": "Pre-trained COCO",
            "weights": "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/weights/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth",
            "type": "pretrained",
            "suffix": "pretrained_coco"
        },
        {
            "name": "Fine-tuned Iter 500",
            "weights": "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_0000499.pth",
            "type": "finetuned",
            "suffix": "finetuned_iter500"
        },
        {
            "name": "Fine-tuned Iter 1000", 
            "weights": "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_0000999.pth",
            "type": "finetuned",
            "suffix": "finetuned_iter1000"
        },
        {
            "name": "Fine-tuned Iter 1500",
            "weights": "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_0001499.pth",
            "type": "finetuned",
            "suffix": "finetuned_iter1500"
        },
        {
            "name": "Fine-tuned Iter 2000",
            "weights": "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_0001999.pth",
            "type": "finetuned",
            "suffix": "finetuned_iter2000"
        },
        {
            "name": "Fine-tuned Iter 2500",
            "weights": "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_0002499.pth",
            "type": "finetuned",
            "suffix": "finetuned_iter2500"
        },
        {
            "name": "Fine-tuned Final",
            "weights": "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_final.pth",
            "type": "finetuned",
            "suffix": "finetuned_final"
        }
    ]
    
    results = []
    
    # Verarbeite mit allen 7 Modellen
    for i, model in enumerate(models, 1):
        print(f"\n{'='*70}")
        print(f"🔍 Modell {i}/7: {model['name']}")
        print(f"{'='*70}")
        
        # Überprüfe ob Weights existieren
        if not os.path.exists(model["weights"]):
            print(f"❌ Gewichte nicht gefunden: {model['weights']}")
            print("   Überspringe dieses Modell...")
            continue
        
        try:
            output_path, num_detections = process_image_with_model(
                image_path, 
                model["weights"], 
                model["type"], 
                model["suffix"]
            )
            
            results.append({
                "model": model["name"],
                "output": output_path,
                "detections": num_detections,
                "weights_file": os.path.basename(model["weights"])
            })
            
        except Exception as e:
            print(f"❌ Fehler bei {model['name']}: {e}")
            continue
    
    # Zusammenfassung
    print(f"\n{'='*70}")
    print("📊 ERGEBNISSE ZUSAMMENFASSUNG - ALLE 7 MODELLE")
    print(f"{'='*70}")
    
    for i, result in enumerate(results, 1):
        if result["output"]:
            print(f"✅ {i}. {result['model']}:")
            print(f"   📁 Output: {os.path.basename(result['output'])}")
            print(f"   🎯 Erkannte Objekte: {result['detections']}")
            print(f"   🏋️ Weights: {result['weights_file']}")
            print()
        else:
            print(f"❌ {i}. {result['model']}: Fehler bei der Verarbeitung")
            print()
    
    print(f"🎉 Verarbeitung aller 7 Modelle abgeschlossen!")
    print(f"📸 Eingabebild: {os.path.basename(image_path)}")
    print(f"📁 Alle Outputs gespeichert in: {output_dir}")
    print(f"📊 Erfolgreiche Verarbeitungen: {len([r for r in results if r['output']])}/7")

if __name__ == "__main__":
    main()
