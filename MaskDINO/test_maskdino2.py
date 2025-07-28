import torch
import os
import sys
import traceback

print("🔍 MaskDINO Implementation Test")
print("=" * 50)

# Test 1: Python-Pfad und Imports
print("\n1️⃣ Testing Python Path and Imports...")
try:
    sys.path.insert(0, "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO")
    print("✅ Python path added successfully")
except Exception as e:
    print(f"❌ Failed to add Python path: {e}")
    sys.exit(1)

# PIL/Pillow Kompatibilität Fix
print("\n🔧 Fixing PIL/Pillow compatibility...")
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
    print(f"⚠️  PIL fix failed, but continuing: {e}")

# Test 2: Detectron2 Imports
print("\n2️⃣ Testing Detectron2 imports...")
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    print("✅ Detectron2 imports successful")
except Exception as e:
    print(f"❌ Detectron2 import failed: {e}")
    print("   → Try: pip install 'pillow<10.0.0' or conda install pillow=9.5.0")
    sys.exit(1)

# Test 3: MaskDINO-spezifische Imports
print("\n3️⃣ Testing MaskDINO-specific imports...")
try:
    from maskdino import add_maskdino_config
    print("✅ MaskDINO config import successful")
except Exception as e:
    print(f"❌ MaskDINO config import failed: {e}")
    print("   → Check if MaskDINO is properly installed")
    sys.exit(1)

# Test 4: Konfigurationsdatei
print("\n4️⃣ Testing configuration file...")
config_path = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO/configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s.yaml"
if os.path.exists(config_path):
    print("✅ Configuration file exists")
else:
    print(f"❌ Configuration file not found: {config_path}")
    print("   → Check if the config file exists")

# Test 5: Gewichtsdatei
print("\n5️⃣ Testing weights file...")
weights_path = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO/weights/maskdino_r50_50ep.pth"
if os.path.exists(weights_path):
    print("✅ Weights file exists")
    # Überprüfe Dateigröße
    file_size = os.path.getsize(weights_path) / (1024 * 1024)  # MB
    print(f"   File size: {file_size:.1f} MB")
    if file_size < 100:
        print("⚠️  Weights file seems too small - might be corrupted")
else:
    print(f"❌ Weights file not found: {weights_path}")
    print("   → Make sure you downloaded the weights correctly")

# Test 6: Testbild
print("\n6️⃣ Testing test image...")
image_path = "/Users/nicklehmacher/Alles/MasterArbeit/image/sherry-christian-8Myh76_3M2U-unsplash.jpg"
try:
    import cv2
    im = cv2.imread(image_path)
    if im is not None:
        print("✅ Test image loaded successfully")
        print(f"   Image shape: {im.shape}")
    else:
        print(f"❌ Failed to load test image: {image_path}")
        print("   → Check if the image file exists and is valid")
except Exception as e:
    print(f"❌ OpenCV import or image loading failed: {e}")

# Test 7: Konfiguration erstellen
print("\n7️⃣ Testing configuration setup...")
try:
    cfg = get_cfg()
    add_maskdino_config(cfg)
    print("✅ Basic configuration created")
    
    # Versuche Konfigurationsdatei zu laden mit Fehlerbehandlung
    try:
        # Versuche zuerst, nur die Hauptkonfiguration ohne Base zu laden
        import yaml
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        # Entferne _BASE_ um rekursive Abhängigkeiten zu vermeiden
        if '_BASE_' in yaml_config:
            print("⚠️  Removing _BASE_ dependency to avoid recursive loading")
            del yaml_config['_BASE_']
        
        # Entferne problematische Schlüssel
        if 'MODEL' in yaml_config and 'RESNETS' in yaml_config['MODEL']:
            if 'STEM_TYPE' in yaml_config['MODEL']['RESNETS']:
                print("⚠️  Removing deprecated STEM_TYPE from config")
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
        config_loaded = True
        
    except Exception as e:
        print(f"⚠️  Configuration file loading failed: {e}")
        print("   → Will use manual configuration")
        config_loaded = False
        
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
        cfg.MODEL.SEM_SEG_HEAD.DIM_FEEDFORWARD = 1024
        cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 3
        cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS = 3
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
        cfg.MODEL.MaskDINO.HIDDEN_DIM = 256  # Standard-Wert
        cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 300  # 300q
        cfg.MODEL.MaskDINO.NHEADS = 8
        cfg.MODEL.MaskDINO.DROPOUT = 0.0
        cfg.MODEL.MaskDINO.DIM_FEEDFORWARD = 2048  # hid2048
        cfg.MODEL.MaskDINO.ENC_LAYERS = 0
        cfg.MODEL.MaskDINO.PRE_NORM = False
        cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ = False
        cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY = 32
        cfg.MODEL.MaskDINO.DEC_LAYERS = 9  # 9 decoder layers + 1
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
        
        # Test-Einstellungen
        cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
        cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
        cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
        cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.8
        cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.8
        cfg.MODEL.MaskDINO.TEST.SCORE_THRESH_TEST = 0.5
        
        # Input-Format
        cfg.INPUT.FORMAT = "RGB"
        cfg.INPUT.MIN_SIZE_TRAIN = (800,)
        cfg.INPUT.MAX_SIZE_TRAIN = 1333
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333
        
        print("✅ Manual configuration applied with all required parameters")
    
    # Gewichte und Device setzen
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cpu"
    
    # Debug-Ausgabe
    print(f"   META_ARCHITECTURE: {cfg.MODEL.META_ARCHITECTURE}")
    print(f"   BACKBONE: {cfg.MODEL.BACKBONE.NAME}")
    print(f"   SEM_SEG_HEAD: {cfg.MODEL.SEM_SEG_HEAD.NAME}")
    print(f"   HIDDEN_DIM: {cfg.MODEL.MaskDINO.HIDDEN_DIM}")
    print(f"   NUM_QUERIES: {cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES}")
    print(f"   DEC_LAYERS: {cfg.MODEL.MaskDINO.DEC_LAYERS}")
    
except Exception as e:
    print(f"❌ Configuration setup failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 8: Model Building (ohne Gewichte)
print("\n8️⃣ Testing model building...")
try:
    # Temporär ohne Gewichte testen
    temp_cfg = cfg.clone()
    temp_cfg.MODEL.WEIGHTS = ""
    
    # Versuche das Modell zu bauen
    from detectron2.modeling import build_model
    model = build_model(temp_cfg)
    print("✅ Model architecture built successfully")
    print(f"   Model type: {type(model)}")
    
    # Teste ob das Modell die richtige Anzahl Parameter hat
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
except Exception as e:
    print(f"❌ Model building failed: {e}")
    print("   → Check if all MaskDINO components are properly implemented")
    traceback.print_exc()

# Test 9: Predictor Creation
print("\n9️⃣ Testing predictor creation...")
try:
    predictor = DefaultPredictor(cfg)
    print("✅ Predictor created successfully")
    print(f"   Model device: {predictor.model.device}")
    
except Exception as e:
    print(f"❌ Predictor creation failed: {e}")
    print("   → This might be due to incompatible weights or model architecture")
    traceback.print_exc()

# Test 10: Inference Test (wenn alles bisher funktioniert hat)
print("\n🔟 Testing inference...")
if 'predictor' in locals() and 'im' in locals() and im is not None:
    try:
        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        outputs = predictor(im_rgb)
        print("✅ Inference successful")
        
        # Überprüfe Ausgabe
        if "instances" in outputs:
            instances = outputs["instances"]
            print(f"   Detected {len(instances)} objects")
            
            if len(instances) > 0:
                print(f"   Classes: {instances.pred_classes.tolist()}")
                print(f"   Scores: {[f'{s:.3f}' for s in instances.scores.tolist()[:5]]}")
                print("✅ MaskDINO is working correctly!")
            else:
                print("⚠️  No objects detected (might be due to threshold settings)")
        else:
            print("❌ Unexpected output format")
            
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        traceback.print_exc()
else:
    print("⏭️  Skipping inference test (previous errors)")

# Zusammenfassung
print("\n" + "=" * 50)
print("📋 TEST SUMMARY")
print("=" * 50)

if 'predictor' in locals():
    print("🎉 MaskDINO appears to be working!")
    print("   You can now run your main inference script.")
else:
    print("❌ MaskDINO implementation has issues.")
    print("   Please fix the errors above before proceeding.")

print("\n💡 Next steps:")
print("   1. If successful: Run your main inference script")
print("   2. If failed: Check the specific error messages above")
print("   3. Common issues:")
print("      - Missing dependencies (detectron2, opencv, torch)")
print("      - Incorrect file paths")
print("      - Corrupted weight files")
print("      - PIL/Pillow version incompatibilities")

print("\n🔧 If PIL errors persist, try:")
print("   conda install pillow=9.5.0")
print("   or pip install 'pillow<10.0.0'")