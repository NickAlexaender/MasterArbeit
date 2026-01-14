# MaskDINO Fine-tuning Script für Car Parts Segmentation
# Basiert auf der funktionierenden test_maskdino.py Konfiguration

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
import yaml
import tempfile
import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch, HookBase
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.evaluation import COCOEvaluator
import logging

# MaskDINO zum Python-Pfad hinzufügen
maskdino_path = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO"
if not os.path.exists(maskdino_path):
    print(f"❌ MaskDINO path does not exist: {maskdino_path}")
    sys.exit(1)

sys.path.insert(0, maskdino_path)
print(f"✅ Added MaskDINO to Python path: {maskdino_path}")

# Importiere MaskDINO-spezifische Module
# In MaskDINO nur maskdino statt MaskDINO.maskdino
try:
    from maskdino import add_maskdino_config
    print("✅ MaskDINO modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MaskDINO modules: {e}")
    print("   → Make sure MaskDINO is properly installed")
    print("   → Check if the path '/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO' exists")
    sys.exit(1)


class EarlyStoppingHook(HookBase):
    """
    Early Stopping Hook - stoppt Training wenn Loss nicht mehr sinkt
    
    Args:
        patience: Anzahl Iterationen ohne Verbesserung bevor gestoppt wird
        min_delta: Minimale Verbesserung die als Fortschritt zählt
    """
    def __init__(self, patience=300, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_iter = 0
        
    def after_step(self):
        # Hole aktuellen Loss aus dem Storage
        storage = self.trainer.storage
        try:
            if storage.iter % 20 == 0:  # Nur alle 20 Iterationen prüfen
                current_loss = storage.history("total_loss").latest()
                
                if current_loss < self.best_loss - self.min_delta:
                    self.best_loss = current_loss
                    self.counter = 0
                    self.best_iter = self.trainer.iter
                else:
                    self.counter += 20
                    
                if self.counter >= self.patience:
                    print(f"\n⚠️ Early Stopping ausgelöst nach {self.trainer.iter} Iterationen")
                    print(f"   Bester Loss: {self.best_loss:.4f} bei Iteration {self.best_iter}")
                    print(f"   Keine Verbesserung seit {self.patience} Iterationen")
                    raise StopIteration("Early stopping triggered")
        except (KeyError, AttributeError):
            # Loss noch nicht verfügbar in frühen Iterationen
            pass


class CarPartsTrainer(DefaultTrainer):
    """
    Custom Trainer für Car Parts Segmentation
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Erstelle einen COCO Evaluator für Car Parts Dataset
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Custom train loader mit Segmentierungsmasken
        """
        from detectron2.data import transforms as T
        from detectron2.data.dataset_mapper import DatasetMapper
        
        # Augmentations für Training (256x256)
        augmentations = [
            T.ResizeShortestEdge(
                short_edge_length=(256,),
                max_size=256,
                sample_style="choice"
            ),
            T.RandomFlip(),
        ]
        
        # Custom mapper der die Segmentierungsmasken lädt
        mapper = DatasetMapper(
            is_train=True,
            augmentations=augmentations,
            image_format=cfg.INPUT.FORMAT,
            use_instance_mask=True,  # Wichtig: Instance-Masken aktivieren
            use_keypoint=False,
            instance_mask_format="bitmask",  # Bitmask für MaskDINO (nicht Polygon)
        )
        
        return build_detection_train_loader(cfg, mapper=mapper)

def register_car_parts_datasets():
    """
    Registriere die konvertierten Car Parts Datasets im COCO-Format
    """
    # Pfade zu den COCO-Annotations und Bildern
    dataset_root = "/Users/nicklehmacher/Alles/MasterArbeit/ultralytics/datasets"
    # COCO JSON already contains paths like "images_256/<split>/...". Using
    # dataset_root ensures detectron2 joins to the correct absolute path
    # (/.../datasets/images_256/...).
    images_root = dataset_root
    
    # Register train dataset
    register_coco_instances(
        "car_parts_train", 
        {},
        os.path.join(dataset_root, "annotations", "instances_train2017.json"),
        images_root
    )
    
    # Register validation dataset
    register_coco_instances(
        "car_parts_val", 
        {},
        os.path.join(dataset_root, "annotations", "instances_val2017.json"),
        images_root
    )
    
    # Setze Class Names für Car Parts (23 Klassen)
    car_parts_classes = [
        'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
        'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
        'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
        'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
        'tailgate', 'trunk', 'wheel'
    ]
    
    # Metadaten für beide Datasets setzen
    for dataset_name in ["car_parts_train", "car_parts_val"]:
        MetadataCatalog.get(dataset_name).set(thing_classes=car_parts_classes)
    
    print(f"✅ Car Parts Datasets registriert mit {len(car_parts_classes)} Klassen")
    return car_parts_classes

def setup_config():
    """
    Setup MaskDINO Konfiguration für Car Parts Fine-tuning
    Basiert exakt auf der funktionierenden test_maskdino.py Konfiguration
    """
    cfg = get_cfg()
    add_maskdino_config(cfg)
    
    # Base-Konfiguration laden und bereinigen (wie in test_maskdino.py)
    try:
        config_path = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"
        
        # YAML-Datei laden und problematische Schlüssel entfernen
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
        
        # Dataset-Einstellungen für Car Parts
        yaml_config['DATASETS'] = {
            'TRAIN': ("car_parts_train",),
            'TEST': ("car_parts_val",)
        }
        
        # Input-Einstellungen (256x256)
        yaml_config['INPUT'] = {
            'FORMAT': 'RGB',
            'MIN_SIZE_TRAIN': (256,),
            'MAX_SIZE_TRAIN': 256,
            'MIN_SIZE_TEST': 256,
            'MAX_SIZE_TEST': 256
        }
        
        # Erstelle temporäre bereinigte Konfigurationsdatei
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
    
    # SEM_SEG_HEAD Konfiguration (wichtig für MaskDINO) - EXAKT wie in test_maskdino.py
    cfg.MODEL.SEM_SEG_HEAD.NAME = "MaskDINOHead"
    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = 255
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 23  # 23 Car Parts Klassen (angepasst von 80)
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
    
    # Dataset-Einstellungen für Car Parts
    cfg.DATASETS.TRAIN = ("car_parts_train",)
    cfg.DATASETS.TEST = ("car_parts_val",)
    
    # Test-Einstellungen entsprechend der YAML-Konfiguration - EXAKT wie in test_maskdino.py
    cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
    cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
    cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.8
    cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.25  # Entspricht YAML (0.25 statt 0.8)
    cfg.MODEL.MaskDINO.TEST.SCORE_THRESH_TEST = 0.5
    
    # Input-Format (256x256 für schnelleres Training)
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MIN_SIZE_TRAIN = (256,)
    cfg.INPUT.MAX_SIZE_TRAIN = 256
    cfg.INPUT.MIN_SIZE_TEST = 256
    cfg.INPUT.MAX_SIZE_TEST = 256
    
    # Pre-trained weights vom funktionierenden Modell - EXAKT wie in test_maskdino.py
    weights_path = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/weights/maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"  # CPU-only (wie in test_maskdino.py)
    
    # Training-spezifische Parameter (für Fine-tuning angepasst)
    cfg.SOLVER.IMS_PER_BATCH = 1  # Sehr kleine Batch Size für CPU
    cfg.SOLVER.BASE_LR = 0.00001  # Sehr niedrige Learning Rate für Fine-tuning
    cfg.SOLVER.MAX_ITER = 15000  # Anzahl Training Iterationen
    cfg.SOLVER.STEPS = (2000, 2500)  # Learning rate decay steps
    cfg.SOLVER.GAMMA = 0.1  # Learning rate decay factor
    cfg.SOLVER.WARMUP_ITERS = 100  # Warmup iterations
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.CHECKPOINT_PERIOD = 100  # Save checkpoint every 500 iterations
    
    # Evaluation
    cfg.TEST.EVAL_PERIOD = 180  # Evaluate every 500 iterations
    
    # Data Loader
    cfg.DATALOADER.NUM_WORKERS = 0  # Setze auf 0 um Multiprocessing-Probleme zu vermeiden
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
    
    # Output Directory
    cfg.OUTPUT_DIR = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    print("✅ Manual configuration applied with all required parameters")
    
    return cfg

def main(args):
    """
    Main training function
    """
    # Setup logging
    setup_logger(name="maskdino")
    
    # Registriere Car Parts Datasets
    car_parts_classes = register_car_parts_datasets()
    
    # Setup configuration
    cfg = setup_config()
    cfg.freeze()
    
    print("🔧 Configuration Summary:")
    print(f"   📊 Classes: {cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES}")
    print(f"   🏋️ Batch Size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"   📈 Learning Rate: {cfg.SOLVER.BASE_LR}")
    print(f"   🔄 Max Iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"   💾 Output Directory: {cfg.OUTPUT_DIR}")
    print(f"   🎯 Pre-trained Weights: {cfg.MODEL.WEIGHTS}")
    
    # Create trainer
    trainer = CarPartsTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    # Add hooks for better monitoring
    trainer.register_hooks([
        hooks.EvalHook(cfg.TEST.EVAL_PERIOD, lambda: trainer.test(cfg, trainer.model)),
        hooks.PeriodicCheckpointer(trainer.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD),
        EarlyStoppingHook(patience=300, min_delta=0.001),  # Early Stopping nach 300 Iterationen ohne Verbesserung
    ])
    
    print("🚀 Starting MaskDINO Fine-tuning for Car Parts Segmentation...")
    print(f"   📚 Training samples: {len(DatasetCatalog.get('car_parts_train'))}")
    print(f"   🧪 Validation samples: {len(DatasetCatalog.get('car_parts_val'))}")
    
    # Start training
    trainer.train()
    
    print("✅ Training completed!")
    print(f"   💾 Models saved to: {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("🎯 MaskDINO Car Parts Fine-tuning Script")
    print("=" * 50)
    print(f"📁 Running from: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Launch training
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
