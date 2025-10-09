# Network Dissection für Encoder-Features

Dieses Modul implementiert die Network Dissection Methode zur Identifikation von konzept-spezifischen Features im Encoder.

## Übersicht

Das Skript `calculate_IoU_for_encoder.py` implementiert Network Dissection mit per-Feature Thresholding zur Identifikation von Features, die auf bestimmte visuelle Konzepte (z.B. rote Bildbereiche) reagieren.

## Verwendung

```bash
# Network Dissection ausführen
python myThesis/encoder/calculate_IoU_for_encoder.py
```

## Network Dissection Methode

### Kernprinzip

Network Dissection berechnet für jedes Feature einen **feature-spezifischen Threshold** über alle Bilder hinweg:

1. **Heatmap-Generierung**: Für jedes Feature wird eine kontinuierliche Aktivierungskarte (Heatmap) pro Bild erzeugt
2. **Per-Feature Threshold**: Der Threshold wird über **alle Pixel aller Bilder** eines Features berechnet
3. **Binarisierung**: Alle Heatmaps des Features werden mit dem **gleichen Threshold** binarisiert
4. **IoU-Berechnung**: Für jedes Bild wird die IoU zwischen binarisierter Heatmap und Ground-Truth-Maske berechnet
5. **mIoU-Aggregation**: Der mittlere IoU (mIoU) über alle Bilder gibt die Konzept-Spezifität des Features an

### Konfiguration

```python
NETWORK_DISSECTION_PERCENTILE = 90.0  # Perzentil für Threshold (0-100)
```

**Wichtig**: Ein niedrigeres Perzentil (z.B. 80) führt zu mehr positiven Pixeln, ein höheres (z.B. 95) zu weniger.

### Threshold-Berechnung

```python
# Flatten alle Heatmaps zu einem 1D-Array
all_values = np.concatenate([hm.flatten() for hm in heatmaps])
threshold = np.percentile(all_values, NETWORK_DISSECTION_PERCENTILE)
```

Dies entspricht der Original-Implementierung aus dem Network Dissection Paper.

## Ausgabe-Struktur

```
myThesis/output/encoder/network_dissection/
├── layer0/
│   ├── miou_network_dissection.csv       # Sortierte mIoU-Ergebnisse
│   └── network_dissection/
│       ├── Feature42/                     # Visualisierungen für bestes Feature
│       │   ├── image 1.png
│       │   ├── image 2.png
│       │   ├── image 3.png
│       │   └── ...
│       └── Feature17/                     # Bei Ties: mehrere beste Features
│           ├── image 1.png
│           └── ...
├── layer1/
│   └── ...
└── ...
```

## CSV-Formate

### mIoU-Ergebnisse (`miou_network_dissection.csv`)

| Spalte           | Beschreibung                                           |
|------------------|--------------------------------------------------------|
| layer_idx        | Layer-Index (0-basiert)                                |
| feature_idx      | Feature-Index (1-basiert)                              |
| miou             | Mittlerer IoU über alle Bilder                         |
| nd_threshold     | Berechneter Network Dissection Threshold               |
| n_images         | Anzahl der Bilder                                      |
| individual_ious  | Komma-separierte Liste individueller IoU-Werte         |
| overlay_dir      | Pfad zu Visualisierungen (nur für beste Features)      |

**Beispiel:**
```csv
layer_idx,feature_idx,miou,nd_threshold,n_images,individual_ious,overlay_dir
0,42,0.8523,0.4521,5,"0.850000,0.870000,0.810000,0.880000,0.850000",layer0/network_dissection/Feature42
```

## Visualisierungen

Alle Overlays verwenden folgende Farbcodierung:

| Farbe | BGR-Wert    | Bedeutung                                    |
|-------|-------------|----------------------------------------------|
| 🔵 Blau  | (255,0,0)   | Überschneidung (True Positive)              |
| 🔴 Rot   | (0,0,255)   | Nur Maske (False Negative)                  |
| 🟡 Gelb  | (0,255,255) | Nur Heatmap (False Positive)                |
| ⚫ Schwarz | (0,0,0)     | Rest (True Negative)                        |

**Interpretation:**
- Viel Blau: Feature erkennt Konzept gut
- Viel Rot: Feature übersieht Teile des Konzepts
- Viel Gelb: Feature feuert außerhalb des Konzepts

## Workflow

1. **Heatmap-Sammlung**: Alle Aktivierungskarten pro Feature werden gesammelt
   ```python
   for item in iter_iou_inputs():
       heatmap = generate_heatmap_only(item, combine="max")
       aggregator.add_heatmap(heatmap, mask, image_id)
   ```

2. **Threshold-Berechnung**: Pro Feature über alle Pixel
   ```python
   threshold = aggregator.compute_network_dissection_threshold()
   ```

3. **mIoU-Berechnung**: Über alle Bilder mit gemeinsamem Threshold
   ```python
   miou, ious = aggregator.compute_miou(threshold)
   ```

4. **Export**:
   - Sortierte CSV mit allen Features (absteigend nach mIoU)
   - Visualisierungen aller Bilder für beste Features

## Interpretation der Ergebnisse

### Hoher mIoU (> 0.7)
→ Feature ist hochspezifisch für das Konzept  
→ Kann als "Detektor" für dieses Konzept verwendet werden

### Mittlerer mIoU (0.3 - 0.7)
→ Feature reagiert teilweise auf das Konzept  
→ Kann zusammen mit anderen Features zur Detektion beitragen

### Niedriger mIoU (< 0.3)
→ Feature ist nicht spezifisch für das Konzept  
→ Reagiert möglicherweise auf andere visuelle Eigenschaften

## Technische Details

### FeatureHeatmapAggregator

Zentrale Klasse zur Sammlung und Verarbeitung von Heatmaps:

```python
class FeatureHeatmapAggregator:
    def __init__(self, layer_idx: int, feature_idx: int)
    def add_heatmap(self, heatmap, mask, image_id)
    def compute_network_dissection_threshold(self, percentile=None) -> float
    def compute_miou(self, threshold) -> Tuple[float, List[float]]
```

### Hauptfunktion

```python
def main_export_network_dissection():
    # 1. Sammle Heatmaps pro Feature
    # 2. Berechne per-Feature Threshold
    # 3. Berechne mIoU
    # 4. Exportiere Ergebnisse
    # 5. Visualisiere beste Features
```

## Abhängigkeiten

- `numpy`: Array-Operationen und Perzentil-Berechnung
- `cv2` (OpenCV): Bild-Resize und Overlay-Erstellung
- `iou_core.py`: Kern-Funktionen für Heatmap-Erzeugung und IoU-Berechnung

## Referenzen

- **Network Dissection Paper**: [Bolin et al. 2017](http://netdissect.csail.mit.edu/)
- **Original Code**: [netdissect GitHub](https://github.com/CSAILVision/NetDissect)

## Troubleshooting

### Keine Ergebnisse
→ Prüfe, ob `output/encoder/layer*/feature.csv` existiert  
→ Prüfe, ob Masken in `myThesis/image/rot/` vorhanden sind

### Niedrige mIoU-Werte
→ Passe `NETWORK_DISSECTION_PERCENTILE` an (z.B. 80 statt 90)  
→ Prüfe Maskenqualität und -größe

### Speicherprobleme
→ Verarbeite Layer einzeln  
→ Reduziere Anzahl der Features (nur beste Features exportieren)
