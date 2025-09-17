# Network Dissection für MaskDINO Encoder Layer 1
# Analysiert den ersten Encoding Layer des finegetunten MaskDINO Modells

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
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.utils.logger import setup_logger

# ======================================================================
# MaskDINO Setup
# ----------------------------------------------------------------------
# Wir erweitern den Python-Pfad, damit das lokale MaskDINO-Repo importierbar ist.
# Das ist nötig, wenn MaskDINO nicht als Paket installiert wurde, sondern
# als Quellcode-Ordner vorliegt. sys.path.insert(0, ...) sorgt dafür,
# dass dieser Pfad in der Modulsuche ganz vorne steht.
# ======================================================================
maskdino_path = "/Users/nicklehmacher/Alles/MasterArbeit/MaskDINO"
sys.path.insert(0, maskdino_path)
from maskdino import add_maskdino_config  # stellt MaskDINO-spezifische CFG-Keys bereit

def register_datasets():
    """Registriere Car Parts Dataset
    - Erwartet COCO-Annotationen (instances_train2017.json)
    - Images liegen im Ordner 'images'
    - Setzt auch die Klassenbezeichnungen für Visualisierung/Evaluation
    """
    dataset_root = "/Users/nicklehmacher/Alles/MasterArbeit/ultralytics/datasets"
    
    # Registrierung robust prüfen
    if "car_parts_train" not in DatasetCatalog.list():
        register_coco_instances(
            "car_parts_train", {},
            os.path.join(dataset_root, "annotations", "instances_train2017.json"),
            os.path.join(dataset_root, "images")
        )
    
    # Klassenliste exakt in derselben Reihenfolge, wie im Training/Evaluations-Setup erwartet.
    # Achtung: NUM_CLASSES in der Config muss damit konsistent sein.
    car_parts_classes = [
        'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
        'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
        'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
        'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
        'tailgate', 'trunk', 'wheel'
    ]
    
    # Metadaten (z. B. Klassenlabels) an das Dataset hängen
    MetadataCatalog.get("car_parts_train").set(thing_classes=car_parts_classes)
    return car_parts_classes

# Registrierung direkt beim Import/Start — so stehen die Datasets später in cfg.DATASETS.* zur Verfügung.
register_datasets()


class RotFeatureAnalyzer:
    """Vereinfachte Analyse des Rot-Features im MaskDINO Encoder
    
    Kernidee:
    - Bild vorverarbeiten und durch das Modell leiten
    - Über einen Forward-Hook Feature-Maps (Encoder/Backbone) abgreifen
    - Für jede Kanal-Feature-Map prüfen, wie gut sie mit einer Rot-Maske (GT) überlappt (IoU)
    - Bestes Feature identifizieren, visualisieren und einen Kurzreport speichern
    """
    
    def __init__(self, model_path: str):
        # Pfad zur .pth/.pkl Gewichtsdatei
        self.model_path = model_path
        self.model = None
        self.cfg = None
        
        # Buffer für abgefangene Zwischenrepräsentationen (per Hook)
        self.features = {}
        
        # Output-Verzeichnis für Plots/Reports
        self.output_dir = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/rot_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Lade Modell & registriere Hooks sofort, damit die Instanz direkt nutzbar ist
        self._load_model()
        self._register_hooks()
    
    def _load_model(self):
        """Lade das MaskDINO Modell
        - Baut eine frische Config auf, friert sie ein, instanziiert das Modell
        - Lädt anschließend die Gewichte via DetectionCheckpointer
        """
        print(f"🔧 Loading model from: {self.model_path}")
        
        self.cfg = self._setup_config()
        # Sicherstellen, dass die Anzahl der Klassen zur Config passt
        assert len(MetadataCatalog.get("car_parts_train").thing_classes) == \
            self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES, "NUM_CLASSES ≠ Anzahl Labels"
        self.cfg.MODEL.WEIGHTS = ""   # wichtig: Checkpointer lädt später, also hier leer lassen
        self.cfg.freeze()               # verhindert versehentliche Änderungen zur Laufzeit
        
        # Modell entsprechend der Config konstruieren; eval() deaktiviert Dropout/BatchNorm-Training
        self.model = build_model(self.cfg)
        self.model.eval()
        
        # Checkpointer lädt state_dict aus gegebener Datei
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(self.model_path)
        print("✅ Model loaded successfully")
    
    def _setup_config(self):
        """Setup MaskDINO Config aligned with fine-tune settings to match checkpoint shapes"""
        cfg = get_cfg()
        add_maskdino_config(cfg)

        # Backbone/ResNet
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

        # Normalization
        cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
        cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

        # SemSeg Head / Pixel Decoder
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

        # MaskDINO core
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

        # Inference input settings
        cfg.INPUT.FORMAT = "RGB"
        cfg.INPUT.MIN_SIZE_TEST = 800
        cfg.INPUT.MAX_SIZE_TEST = 1333

        # Datasets (for metadata)
        cfg.DATASETS.TRAIN = ("car_parts_train",)

        # Test/Inference thresholds
        cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON = False
        cfg.MODEL.MaskDINO.TEST.INSTANCE_ON = True
        cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON = False
        cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD = 0.8
        cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.25
        cfg.MODEL.MaskDINO.TEST.SCORE_THRESH_TEST = 0.5

        # Device
        cfg.MODEL.DEVICE = "cpu"

        return cfg
    
    def _register_hooks(self):
        """Hook hart auf Encoder-Layer setzen"""
        TARGET = "sem_seg_head.pixel_decoder.transformer.encoder.layers.0"

        def hook_fn(m, i, o):
            self.features["encoder_features"] = o.detach().cpu()

        found = False
        for name, module in self.model.named_modules():
            if name.endswith(TARGET):
                module.register_forward_hook(hook_fn)
                print(f"🎯 Hook registered on: {name}")
                found = True
                break
        if not found:
            print(f"⚠️ No suitable layer found for hook (target: {TARGET})")
    
    def load_rot_mask(self, rot_path: str):
        """Lade die Rot-Maske (Binärbild)
        - Erwartet ein Bild, dessen Helligkeit/Alpha bereits die Rotfläche markiert
        - Wandelt in Grau und dann via Schwellwert in {0,1} um
        """
        rot_img = cv2.imread(rot_path)
        if rot_img is None:
            raise ValueError(f"Could not load rot mask: {rot_path}")
        
        # KISS-Ansatz: Wir interpretieren 'rot' nicht per Farbsegmentierung,
        # sondern nehmen an, dass die Eingabe selbst bereits die gewünschte Maske repräsentiert.
        rot_mask = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)
        rot_mask = (rot_mask > 128).astype(np.uint8)  # einfacher Threshold
        
        print(f"✅ Rot mask loaded: {rot_mask.shape}, {rot_mask.sum()} rot pixels")
        return rot_mask
    
    def preprocess_image(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(self.cfg.INPUT.MIN_SIZE_TEST / min(h, w),
                    self.cfg.INPUT.MAX_SIZE_TEST / max(h, w))
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float()  # KEINE Normierung
        return tensor, img_resized, (nh, nw)
    
    def extract_features(self, image_path: str):
        """Extrahiere Features
        - Führt einen Forward-Pass durch das Modell aus
        - Der zuvor registrierte Hook speichert die Zwischenrepräsentation in self.features
        - Achtung: Detectron2 erwartet bei 'model(inputs)' eine Liste von Dicts.
        """
        print(f"🔍 Extracting features from: {image_path}")
        
        image_tensor, original_image, new_size = self.preprocess_image(image_path)
        
        with torch.no_grad():  # keine Gradienten nötig -> schneller, weniger RAM
            inputs = [{
                "image": image_tensor,   # [C,H,W] FloatTensor
                "height": new_size[0],  # Originalhöhe nach Resize (für Postprocessing)
                "width": new_size[1]    # Originalbreite nach Resize
            }]
            
            self.features.clear()   # alten Hook-Output verwerfen
            outputs = self.model(inputs)  # Forward löst den Hook aus
        
        # Falls kein Hook feuert (z. B. Layername nicht gefunden), abbrechen
        if 'encoder_features' not in self.features:
            print("❌ No features extracted")
            return None
            
        return self.features['encoder_features'], original_image, new_size
    
    def calculate_iou(self, feature_map: np.ndarray, rot_mask: np.ndarray):
        """Berechne IoU zwischen Feature Map und Rot-Maske
        Vorgehen:
        - Resample Feature-Map auf Maskengröße (bilinear via OpenCV)
        - Binarisiere Feature-Map durch datenabhängigen Schwellwert (mean + std)
        - IoU = |Schnittmenge| / |Vereinigung|
        
        Anmerkung:
        - Der Schwellwert (mean+std) ist heuristisch. Je nach Verteilung kann ein
          quantilbasierter Threshold (z. B. 85. Perzentil) stabiler sein.
        """
        # Größen aneinander anpassen (Feature [Hf,Wf] -> Maske [Hm,Wm])
        if feature_map.shape != rot_mask.shape:
            feature_map = cv2.resize(feature_map, (rot_mask.shape[1], rot_mask.shape[0]))
        
        # Binärschwellwert aus Statistik der Feature-Map
        threshold = np.mean(feature_map) + np.std(feature_map)
        feature_binary = (feature_map > threshold).astype(np.uint8)
        
        # IoU-Berechnung auf Basis boolscher Mengen
        intersection = np.logical_and(feature_binary, rot_mask).sum()
        union = np.logical_or(feature_binary, rot_mask).sum()
        
        iou = intersection / union if union > 0 else 0.0
        return iou, feature_binary
    
    def find_best_rot_feature(self, features: torch.Tensor, rot_mask: np.ndarray):
        """Finde das Feature mit der besten IoU für Rot
        Unterstützt:
        - 4D Convolutional Maps: [B, C, H, W]
        - 3D Transformer Tokens: [B, N, D] (wird auf 2D-Gitter (H×W) gefaltet)
        """
        print("🎯 Analyzing features for rot correlation...")

        # Hilfsfunktion: N in (H, W) faktorisieren (nahe Quadrat, ohne Padding)
        def factor_hw(n: int):
            import math
            s = int(math.sqrt(n))
            # finde größtes h<=sqrt(n), das n teilt; fallback auf (s, ceil(n/s))
            for h in range(s, 0, -1):
                if n % h == 0:
                    return h, n // h
            return s, (n + s - 1) // s

        if len(features.shape) == 4:  # [B, C, H, W]
            features_np = features[0].numpy()  # [C, H, W]
            num_channels = features_np.shape[0]
            get_channel_map = lambda c: features_np[c]
        elif len(features.shape) == 3:  # [B, N, D] -> [D, H, W]
            tokens = features[0].numpy()  # [N, D]
            tokens = tokens.T  # [D, N]
            num_channels, n = tokens.shape
            h, w = factor_hw(n)
            # baue Zugriffsfunktion, ohne alle Kanäle upfront zu reshapen
            def get_channel_map(c):
                return tokens[c].reshape(h, w)
            print(f"ℹ️ Transformer tokens reshaped to grid: ({h}, {w}) from N={n}")
        else:
            print(f"❌ Unsupported feature shape: {features.shape}")
            return None, None, None, None

        print(f"📊 Analyzing {num_channels} feature channels")

        best_iou = 0.0
        best_channel = 0
        best_feature_binary = None
        ious = []  # (channel, iou)

        for c in range(num_channels):
            feature_map = get_channel_map(c)
            iou, feature_binary = self.calculate_iou(feature_map, rot_mask)
            ious.append((c, iou))
            if iou > best_iou:
                best_iou = iou
                best_channel = c
                best_feature_binary = feature_binary

        ious.sort(key=lambda x: x[1], reverse=True)
        print(f"✅ Best rot feature: Channel {best_channel} with IoU {best_iou:.4f}")
        return best_channel, best_iou, best_feature_binary, ious[:10]
    
    def visualize_results(self, original_image: np.ndarray, rot_mask: np.ndarray, 
                         best_channel: int, best_iou: float, best_feature_binary: np.ndarray,
                         features: torch.Tensor, top_ious: list):
        """Visualisiere Ergebnisse
        - 6-Panel Abbildung: Original, Maske, Raw-Feature, Binarisierung, Overlay, IoU-Barplot
        - Speichert Plot und JSON-Report im Output-Ordner
        
        Hinweise:
        - Für 'best_feature_raw' wird ggf. von der Original-Feature-Auflösung auf Maskengröße resampled.
        - Overlay nutzt Rotkanal für GT, Blaukanal für Feature (intuitive Gegenüberstellung).
        """
        print("🎨 Creating visualizations...")
        
        # 1. Haupt-Visualisierung (2x3 Subplots)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Rot-Feature Analyse - Bestes Feature: Channel {best_channel} (IoU: {best_iou:.4f})', fontsize=16)
        
        # (0,0) Originalbild
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # (0,1) Rot-Maske (Ground Truth)
        axes[0, 1].imshow(rot_mask, cmap='Reds')
        axes[0, 1].set_title('Rot Ground Truth')
        axes[0, 1].axis('off')
        
        # (0,2) Rohes Kanal-Feature (vor Binarisierung)
        if len(features.shape) == 4:
            best_feature_raw = features[0, best_channel].numpy()
        else:  # Transformer-Fall (bereits zu [C,H,W] umgeformt in find_best_rot_feature)
            features_np = features[0].numpy().T  # [D, N]
            spatial_size = int(np.sqrt(features_np.shape[1]))
            best_feature_raw = features_np[best_channel].reshape(spatial_size, spatial_size)
        
        # Auf Maskengröße bringen für visuelles Alignment
        if best_feature_raw.shape != rot_mask.shape:
            best_feature_raw = cv2.resize(best_feature_raw, (rot_mask.shape[1], rot_mask.shape[0]))
        
        axes[0, 2].imshow(best_feature_raw, cmap='viridis')
        axes[0, 2].set_title(f'Channel {best_channel} (Raw)')
        axes[0, 2].axis('off')
        
        # (1,0) Binarisierte Aktivierung dieses Kanals
        axes[1, 0].imshow(best_feature_binary, cmap='Blues')
        axes[1, 0].set_title(f'Channel {best_channel} (Binary)')
        axes[1, 0].axis('off')
        
        # (1,1) Overlay: Rot = GT, Blau = Feature -> Purple bei Überlappung
        overlay = np.zeros((rot_mask.shape[0], rot_mask.shape[1], 3))
        overlay[:, :, 0] = rot_mask        # R-Kanal für Ground Truth
        overlay[:, :, 2] = best_feature_binary  # B-Kanal für Feature
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (Rot=GT, Blau=Feature)')
        axes[1, 1].axis('off')
        
        # (1,2) Balkendiagramm der Top-10-IoUs
        channels, iou_values = zip(*top_ious)
        axes[1, 2].bar(range(len(channels)), iou_values, color='skyblue', edgecolor='navy')
        axes[1, 2].set_title('Top 10 Features IoU mit Rot')
        axes[1, 2].set_xlabel('Feature Rank')
        axes[1, 2].set_ylabel('IoU')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Kanal-Labels über die Balken schreiben (kompakt)
        for i, (ch, iou) in enumerate(top_ious):
            axes[1, 2].text(i, iou + 0.01, f'Ch{ch}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'rot_feature_analysis.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Visualization saved: {output_path}")
        
        # 2. Report als JSON (leicht weiterverarbeitbar, reproducible)
        report = {
            'best_channel': int(best_channel),
            'best_iou': float(best_iou),
            'top_10_features': [(int(ch), float(iou)) for ch, iou in top_ious],
            'analysis_summary': f"Channel {best_channel} shows best correlation with red color (IoU: {best_iou:.4f})"
        }
        
        report_path = os.path.join(self.output_dir, 'rot_analysis_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Report saved: {report_path}")
        
        # Erfolgskriterium: IoU > 0.3 (heuristisch). Je nach Use-Case anpassbar.
        return best_iou > 0.3  
    
    def run_analysis(self, image_path: str, rot_mask_path: str):
        """Führe komplette Rot-Analyse durch
        Pipeline:
        1) Rot-Maske laden
        2) Features extrahieren (Forward + Hook)
        3) Bestes 'Rot'-Feature suchen (max. IoU)
        4) Visualisieren & Report speichern
        """
        print("🚀 Starting Rot-Feature Analysis...")
        print(f"   📸 Image: {image_path}")
        print(f"   🔴 Rot Mask: {rot_mask_path}")
        print("=" * 60)
        
        # 1. Lade Rot-Maske
        rot_mask = self.load_rot_mask(rot_mask_path)
        
        # 2. Extrahiere Features
        result = self.extract_features(image_path)
        if result is None:
            print("❌ Feature extraction failed")
            return False
        
        features, original_image, new_size = result
        print(f"✅ Extracted features shape: {features.shape}")
        
        # 3. Finde bestes Rot-Feature
        best_channel, best_iou, best_feature_binary, top_ious = self.find_best_rot_feature(features, rot_mask)
        
        if best_channel is None:
            print("❌ Analysis failed")
            return False
        
        # 4. Visualisiere Ergebnisse
        success = self.visualize_results(original_image, rot_mask, best_channel, best_iou, 
                                       best_feature_binary, features, top_ious)
        
        print("=" * 60)
        if success:
            print(f"✅ Analysis successful! Best IoU: {best_iou:.4f}")
        else:
            print(f"⚠️ Low IoU detected: {best_iou:.4f}")
        
        print(f"💾 Results saved to: {self.output_dir}")
        
        return success


def main():
    """Main function für Rot-Feature Analyse
    - Prüft Pfade und setzt Logger
    - Führt die komplette Analyse in einem try/except-Block aus
    """
    print("🔴 MaskDINO Rot-Feature Analyse")
    print("=" * 60)
    
    # Pfade zu Modell, Eingangsbild und Rot-GT-Maske
    model_path = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/car_parts_finetune/model_final.pth"
    image_path = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/new_21_png_jpg.rf.d0c9323560db430e693b33b36cb84c3b.jpg"
    rot_mask_path = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/colours/rot.png"
    
    # Existenz-Check: frühzeitige Fehlererkennung, klare Fehlermeldungen
    for path, name in [(model_path, "Model"), (image_path, "Image"), (rot_mask_path, "Rot mask")]:
        if not os.path.exists(path):
            print(f"❌ {name} not found: {path}")
            return
    
    # Detectron2-Logger (gibt u. a. Config/Shape-Infos beim Forward aus)
    setup_logger(name="maskdino")
    
    try:
        # Analysiere Rot-Feature
        analyzer = RotFeatureAnalyzer(model_path)
        success = analyzer.run_analysis(image_path, rot_mask_path)
        
        if success:
            print("🎉 Rot-Feature analysis completed successfully!")
        else:
            print("⚠️ Analysis completed with warnings.")
            
    except Exception as e:
        # Harte Fehler ausführlich ausgeben (Stacktrace), um Debugging zu erleichtern
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
