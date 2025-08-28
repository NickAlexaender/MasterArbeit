# MaskDINO Fine-tuning für Car Parts Segmentation

## Überblick

Dieses Projekt implementiert ein Fine-tuning von MaskDINO für die Segmentierung von Autoteilen. Das ursprünglich auf COCO-Daten trainierte Modell wird auf einen spezialisierten Datensatz mit 23 Autoteil-Kategorien angepasst.

## Dataset-Konvertierung

### YOLOv8 zu COCO Format
- **Skript**: `ultralytics/yolo_to_coco_converter.py`
- **Zweck**: Konvertiert YOLOv8-Annotationen in das COCO-Format für MaskDINO
- **Output**: 
  - `annotations/instances_train2017.json`
  - `annotations/instances_val2017.json`

### Car Parts Klassen (23 Kategorien)
```
'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
'tailgate', 'trunk', 'wheel'
```

## Fine-tuning Process

### Training-Skript
- **Datei**: `MaskDINO/fine-tune.py`
- **Basis-Modell**: `maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth`
- **Architektur**: ResNet-50 Backbone mit MaskDINO Head

### Training-Parameter
```python
cfg.SOLVER.IMS_PER_BATCH = 1          # Batch Size (CPU-optimiert)
cfg.SOLVER.BASE_LR = 0.00001          # Learning Rate (sehr niedrig für Fine-tuning)
cfg.SOLVER.MAX_ITER = 3000            # Trainings-Iterationen
cfg.SOLVER.CHECKPOINT_PERIOD = 500    # Checkpoint alle 500 Iterationen
cfg.TEST.EVAL_PERIOD = 500            # Evaluation alle 500 Iterationen
```

### Modell-Konfiguration
- **Klassen-Anzahl**: 80 → 23 (COCO → Car Parts)
- **Hidden Dimension**: 256
- **Object Queries**: 300
- **Decoder Layers**: 3
- **Device**: CPU (für Apple Silicon optimiert)

## Modell-Checkpoints

Das Training generiert folgende Checkpoints:
1. `model_0000499.pth` - Nach 500 Iterationen
2. `model_0000999.pth` - Nach 1000 Iterationen
3. `model_0001499.pth` - Nach 1500 Iterationen
4. `model_0001999.pth` - Nach 2000 Iterationen
5. `model_0002499.pth` - Nach 2500 Iterationen
6. `model_final.pth` - Finales Modell nach 3000 Iterationen

## Evaluation & Vergleich

### Vergleichs-Skript
- **Datei**: `MaskDINO/create_images_from_finetuned.py`
- **Zweck**: Vergleicht alle 7 Modelle (1 Pre-trained + 6 Fine-tuned Checkpoints)
- **Output**: `model_comparison_outputs/` mit segmentierten Bildern

### Modell-Performance Vergleich
- **Pre-trained COCO**: Erkennt generische Objekte (car, motorcycle, person, etc.)
- **Fine-tuned Models**: Erkennen spezifische Autoteile (hood, front_bumper, wheel, etc.)

## Technische Details

### Kompatibilitäts-Fixes
```python
# PIL/Pillow Fix für neuere Versionen
PIL.Image.LINEAR = PIL.Image.Resampling.BILINEAR

# NumPy Fix für deprecated Attribute
np.bool = bool
```

### Dataset-Registrierung
```python
register_coco_instances(
    "car_parts_train", 
    {},
    "annotations/instances_train2017.json",
    "images/"
)
```

## Ergebnisse

Das Fine-tuning zeigt deutliche Verbesserungen:
- **Spezifität**: Erkennung spezifischer Autoteile statt generischer Objekte
- **Konfidenz**: Höhere Konfidenzwerte für Autoteil-spezifische Detektionen
- **Relevanz**: Direkt anwendbar für Automotive Computer Vision Tasks

## Verwendung

1. **Training starten**:
   ```bash
   python3 fine-tune.py
   ```

2. **Modelle vergleichen**:
   ```bash
   python3 create_images_from_finetuned.py
   ```

3. **Ergebnisse prüfen**:
   - Training-Logs: `output/car_parts_finetune/`
   - Vergleichs-Bilder: `model_comparison_outputs/`
