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

def register_butterfly_datasets():
    """
    Registriere die Butterfly Datasets für das Fine-tuned Modell
    """
    # Pfade zu den COCO-Annotations und Bildern
    dataset_root = "/Users/nicklehmacher/Alles/MasterArbeit/leedsbutterfly/coco"
    
    # Register validation dataset (für Metadaten)
    try:
        register_coco_instances(
            "butterfly_val", 
            {},
            os.path.join(dataset_root, "annotations", "instances_val2017.json"),
            os.path.join(dataset_root, "val2017")
        )
        
        # Setze Class Names für Butterfly (10 Klassen)
        butterfly_classes = [
            'Danaus plexippus', 'Heliconius charitonius', 'Heliconius erato', 
            'Junonia coenia', 'Lycaena phlaeas', 'Nymphalis antiopa', 
            'Papilio cresphontes', 'Pieris rapae', 'Vanessa atalanta', 'Vanessa cardui'
        ]
        
        MetadataCatalog.get("butterfly_val").set(thing_classes=butterfly_classes)
        print(f"✅ Butterfly Dataset registriert mit {len(butterfly_classes)} Klassen")
        return butterfly_classes
    except Exception as e:
        print(f"⚠️ Butterfly Dataset bereits registriert oder Fehler: {e}")
        return [
            'Danaus plexippus', 'Heliconius charitonius', 'Heliconius erato', 
            'Junonia coenia', 'Lycaena phlaeas', 'Nymphalis antiopa', 
            'Papilio cresphontes', 'Pieris rapae', 'Vanessa atalanta', 'Vanessa cardui'
        ]

def setup_config(model_type="pretrained", dataset_type="car_parts"):
    """
    Setup MaskDINO Konfiguration basierend auf test_maskdino.py
    model_type: "pretrained" oder "finetuned"
    dataset_type: "car_parts" oder "butterfly"
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
    
    # Anzahl Klassen je nach Modell-Typ und Dataset
    if model_type == "finetuned":
        if dataset_type == "butterfly":
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 10  # Butterfly Klassen
        else:
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
    
    # Dataset-Einstellungen je nach Modell-Typ und Dataset
    if model_type == "finetuned":
        if dataset_type == "butterfly":
            cfg.DATASETS.TRAIN = ("butterfly_train",)
            cfg.DATASETS.TEST = ("butterfly_val",)
        else:
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

def process_image_with_model(image_path, weights_path, model_type, output_suffix, dataset_type="car_parts"):
    """
    Verarbeite ein Bild mit einem spezifischen Modell
    dataset_type: "car_parts" oder "butterfly"
    """
    print(f"\n🔧 Verarbeite mit {model_type} Modell ({dataset_type})...")
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
    cfg = setup_config(model_type, dataset_type)
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
            if dataset_type == "butterfly":
                metadata = MetadataCatalog.get("butterfly_val")
            else:
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

def get_model_files_from_folder(folder_path):
    """
    Finde alle .pth Modell-Dateien in einem Ordner und sortiere sie
    """
    if not os.path.exists(folder_path):
        print(f"⚠️ Ordner nicht gefunden: {folder_path}")
        return []
    
    model_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.pth'):
            model_files.append(os.path.join(folder_path, f))
    
    # Sortiere nach Iteration (model_0000499.pth -> 499)
    def extract_iter(path):
        filename = os.path.basename(path)
        if filename == "model_final.pth":
            return float('inf')  # Final am Ende
        try:
            # Extrahiere Nummer aus model_0000499.pth
            num = int(filename.replace("model_", "").replace(".pth", ""))
            return num
        except:
            return 0
    
    model_files.sort(key=extract_iter)
    return model_files

def main():
    """
    Hauptfunktion - Verarbeite Bilder mit allen Modellen aus beiden Ordnern
    """
    print("🚀 MaskDINO Modell-Vergleich - Car Parts & Butterfly")
    print("=" * 70)
    
    # Basis-Pfade
    base_output_dir = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output"
    
    # Konfiguration für beide Datensätze
    datasets_config = {
        "car_parts": {
            "finetune_folder": os.path.join(base_output_dir, "car_parts_finetune"),
            "test_image": "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/new_21_png_jpg.rf.d0c9323560db430e693b33b36cb84c3b.jpg",
            "register_func": register_car_parts_datasets,
            "num_classes": 23,
            "metadata_name": "car_parts_val"
        },
        "butterfly": {
            "finetune_folder": os.path.join(base_output_dir, "butterfly_parts_finetune"),
            "test_image": "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/butterfly/1images/0010005.png",
            "register_func": register_butterfly_datasets,
            "num_classes": 10,
            "metadata_name": "butterfly_val"
        }
    }
    
    # Pre-trained Modell (gemeinsam für beide)
    pretrained_weights = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/weights/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"
    
    # Erstelle Output-Ordner
    output_dir = os.path.join(base_output_dir, "model_comparison_outputs")
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output-Ordner: {output_dir}")
    
    all_results = []
    
    # Verarbeite beide Datensätze
    for dataset_name, config in datasets_config.items():
        print(f"\n{'#'*70}")
        print(f"# DATENSATZ: {dataset_name.upper()}")
        print(f"{'#'*70}")
        
        # Überprüfe ob Testbild existiert
        if not os.path.exists(config["test_image"]):
            print(f"❌ Testbild nicht gefunden: {config['test_image']}")
            continue
        
        print(f"📸 Testbild: {os.path.basename(config['test_image'])}")
        
        # Registriere Dataset
        config["register_func"]()
        
        # Finde alle Modell-Dateien im Fine-tune Ordner
        finetune_models = get_model_files_from_folder(config["finetune_folder"])
        print(f"🔍 Gefundene Fine-tuned Modelle: {len(finetune_models)}")
        for m in finetune_models:
            print(f"   - {os.path.basename(m)}")
        
        # Erstelle Modell-Liste: Pre-trained + alle Fine-tuned
        models = []
        
        # Pre-trained Modell (nur einmal pro Dataset für Vergleich)
        if os.path.exists(pretrained_weights):
            models.append({
                "name": f"Pre-trained COCO ({dataset_name})",
                "weights": pretrained_weights,
                "type": "pretrained",
                "suffix": f"{dataset_name}_pretrained_coco",
                "folder": "pretrained",
                "dataset": dataset_name
            })
        
        # Alle Fine-tuned Modelle
        folder_name = os.path.basename(config["finetune_folder"])
        for weights_path in finetune_models:
            model_filename = os.path.basename(weights_path)
            # Extrahiere Iteration aus Dateiname
            if model_filename == "model_final.pth":
                iter_name = "final"
            else:
                iter_num = model_filename.replace("model_", "").replace(".pth", "")
                iter_name = f"iter{int(iter_num)+1}"  # +1 weil 0-indexed
            
            models.append({
                "name": f"{folder_name} - {model_filename}",
                "weights": weights_path,
                "type": "finetuned",
                "suffix": f"{dataset_name}_{folder_name}_{iter_name}",
                "folder": folder_name,
                "dataset": dataset_name
            })
        
        # Verarbeite alle Modelle für diesen Datensatz
        for i, model in enumerate(models, 1):
            print(f"\n{'='*70}")
            print(f"🔍 Modell {i}/{len(models)}: {model['name']}")
            print(f"📂 Ordner: {model['folder']}")
            print(f"🏷️ Dataset: {model['dataset']}")
            print(f"{'='*70}")
            
            # Überprüfe ob Weights existieren
            if not os.path.exists(model["weights"]):
                print(f"❌ Gewichte nicht gefunden: {model['weights']}")
                continue
            
            try:
                output_path, num_detections = process_image_with_model(
                    config["test_image"], 
                    model["weights"], 
                    model["type"], 
                    model["suffix"],
                    dataset_type=dataset_name
                )
                
                all_results.append({
                    "dataset": dataset_name,
                    "model": model["name"],
                    "folder": model["folder"],
                    "output": output_path,
                    "detections": num_detections,
                    "weights_file": os.path.basename(model["weights"])
                })
                
            except Exception as e:
                print(f"❌ Fehler bei {model['name']}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Zusammenfassung
    print(f"\n{'#'*70}")
    print("📊 ERGEBNISSE ZUSAMMENFASSUNG - ALLE MODELLE")
    print(f"{'#'*70}")
    
    # Gruppiere nach Dataset
    for dataset_name in datasets_config.keys():
        dataset_results = [r for r in all_results if r["dataset"] == dataset_name]
        if dataset_results:
            print(f"\n📁 {dataset_name.upper()} ({len(dataset_results)} Modelle):")
            print("-" * 60)
            for i, result in enumerate(dataset_results, 1):
                if result["output"]:
                    print(f"  ✅ {i}. {result['model']}:")
                    print(f"     📂 Ordner: {result['folder']}")
                    print(f"     📁 Output: {os.path.basename(result['output'])}")
                    print(f"     🎯 Erkannte Objekte: {result['detections']}")
                    print(f"     🏋️ Weights: {result['weights_file']}")
                else:
                    print(f"  ❌ {i}. {result['model']}: Fehler")
    
    print(f"\n{'='*70}")
    print(f"🎉 Verarbeitung abgeschlossen!")
    print(f"📊 Erfolgreiche Verarbeitungen: {len([r for r in all_results if r['output']])}/{len(all_results)}")
    print(f"📁 Alle Outputs gespeichert in: {output_dir}")

if __name__ == "__main__":
    main()
