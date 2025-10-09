## Referenz: Datenflüsse, Strukturen und Speicherorte (Transformer-Decoder)

Diese Datei fasst alle relevanten Strukturen aus dem Decoder-Pipeline zusammen – von Setup über Hook-Registrierung bis zur IoU-Berechnung. Die vier Hauptmodule arbeiten zusammen, um Network Dissection auf MaskDINO's Transformer-Decoder durchzuführen.

### Einstiegspunkte

- Build & Handover: `nd_on_transformer_decoder.py`
	- Baut MaskDINO-Konfiguration und Modell, lädt finetuned Gewichte.
	- Findet automatisch den Transformer-Decoder im Modell.
	- Sammelt Bildpfade (aktuell 1 Beispielbild) und ruft
		`accept_weights_model_images(weights_path, model, image_list)` auf.

- Extraktion & Export: `weights_extraction_transformer_decoder.py`
	- Registriert Hooks auf Transformer-Decoder-Layern (z.B. `predictor.decoder.layers.0/1/2`).
	- Extrahiert Pixel-Embeddings aus Encoder/Backbone-Features.
	- Führt pro Bild einen Forward aus, sammelt Query-Hidden-States, exportiert CSV pro Layer.

- IoU-Kernlogik: `iou_core_decoder.py`
	- Berechnet Dot-Product zwischen Query-Features und Pixel-Embeddings.
	- Skaliert auf Input-Größe, binarisiert mit konfigurierbaren Schwellenwerten.
	- Berechnet IoU gegen Ground-Truth-Maske.

- IoU-Pipeline: `calculate_IoU_for_decoder.py`
	- Generator für IoU-Eingabedaten: kombiniert CSV-Queries mit Pixel-Embeddings.
	- Vollständige Export-Pipeline mit Heatmaps und Overlay-Visualisierungen.
	- Automatische Ergebnis-Sortierung und Best-Performance-Tracking.

## In-Memory-Datenstrukturen

### Modell-Attribute (zur Laufzeit gesetzt)

- `hidden_states_per_layer: Dict[str, List[torch.Tensor]]`
	- Gefüllt durch Forward-Hooks auf Decoder-Layern mit Name-Muster:
		`predictor.decoder.layers.<idx>` (nur Hauptlayer-Ebene)
	- Key = genauer Modulname (z. B. `...decoder.layers.0`), Value = Hidden-States-Tensor (detach, clone).
	- Format der Hidden-States: `[num_queries, batch_size, hidden_dim]` (typisch `[300, 1, 256]`)

- `hook_handles: List[HookHandle]`
	- Zum Entfernen der Decoder-Hooks am Ende.

- `encoder_features: Dict[str, torch.Tensor]`
	- Pixel-Embeddings aus Backbone/Encoder für Network Dissection.
	- Extrahiert aus Layern wie `backbone.res*`, `input_proj.*`, etc.
	- Priorität: `input_proj > res5 > res4 > res3 > res2`

### Datenstrukturen für IoU-Berechnung

- `DecoderIoUInput` (dataclass):
	- `layer_idx: int` – Layer-Index (0, 1, 2)
	- `image_idx: int` – Bild-Index (1-basiert in CSV)
	- `query_idx: int` – Query-Index (1-basiert in CSV)
	- `query_features: np.ndarray` – Shape `(256,)` Query-Embedding
	- `pixel_embedding: np.ndarray` – Shape `(256, H, W)` Pixel-Features
	- `input_size: Tuple[int, int]` – (H_in, W_in) Input-Bildgröße
	- `mask_input: np.ndarray` – bool, Shape `(H_in, W_in)` Ground-Truth-Maske

- `IoUResultDecoder` (dataclass):
	- `layer_idx, image_idx, query_idx: int`
	- `iou: float` – IoU-Wert zwischen 0 und 1
	- `threshold: float` – verwendeter Schwellenwert
	- `positives: int` – Anzahl positive Pixel nach Binarisierung
	- `heatmap: Optional[np.ndarray]` – skalierte Response-Map (falls `return_heatmap=True`)

### Standardisierung von Query-Formaten

Ziel: Query-Hidden-States nach standardisiertem Format extrahieren

- Input: `[num_queries, batch_size, hidden_dim]` (typisch `[300, 1, 256]`)
- Extraction: `queries = hidden_states[:, 0, :]` → `[300, 256]` (für `batch_size=1`)
- CSV-Format: Eine Zeile pro Query mit 256 Gewichten als Komma-separierte Werte

Layer-Nummer-Extraktion:
- `predictor.decoder.layers.0` → `layer0`
- `predictor.decoder.layers.1` → `layer1` 
- `predictor.decoder.layers.2` → `layer2`
- Fallback: Regex-Extraktion `layers[._](\d+)`

## Persistierte Artefakte (Speicherorte & Schemas)

Basisverzeichnis: `myThesis/output/decoder`

Pfadableitung:
- `project_root = <.../myThesis>` (Elternverzeichnis der Python-Datei)
- `base_out = project_root / "output" / "decoder"`

### 1) Query-CSV pro Layer (alle Bilder aggregiert)

- Speicherort: `output/decoder/layer<N>/Query.csv`
	- `<N>` = 0, 1, 2 für die drei Decoder-Layer
- Header: `Name, Gewicht 1, Gewicht 2, ..., Gewicht 256`
- Zeilen-Format: `"Bild<img_idx>, Query<query_idx>", weight1, weight2, ..., weight256`
	- `img_idx`, `query_idx` sind 1-basiert
	- Gewichte als float-Werte mit 6 Dezimalstellen

Interpretation:
- Jede Zeile beschreibt die Hidden-State-Aktivierung einer Query für ein bestimmtes Bild in diesem Layer.
- 300 Queries pro Bild → 300 Zeilen pro Bild im CSV.

### 2) Pixel-Embeddings pro Bild

- NPY-Datei: `output/decoder/pixel_embeddings/pixel_embed_Bild<NNNN>.npy`
	- `<NNNN>` = 4-stelliger Bild-Index (0-basiert, z.B. Bild0000)
	- Shape: `(channels, height, width)` (typisch `(256, 25, 25)`)
	- Datentyp: float32

- Metadaten: `output/decoder/pixel_embeddings/metadata_Bild<NNNN>.json`

Schema (Beispiel):
```json
{
	"image_index": 0,
	"layer_name": "sem_seg_head.pixel_decoder.input_proj.0.0",
	"shape": [256, 25, 25],
	"channels": 256,
	"height": 25,
	"width": 25,
	"npy_file": "pixel_embed_Bild0000.npy"
}
```

Verwendung für IoU-Berechnung:
- Pixel-Embedding wird als Base für Dot-Product mit Query-Features verwendet.
- Resulting Response-Map wird auf Input-Größe skaliert für räumliche IoU-Berechnung.

### 3) IoU-Ergebnisse (strukturiert nach Layern)

- CSV-Ergebnisse: `output/decoder/iou_results/layer<N>/iou_sorted.csv`
- Heatmaps: `output/decoder/iou_results/layer<N>/heatmaps/Bild<I>_Query<Q>.png`
- Vergleiche: `output/decoder/iou_results/layer<N>/comparisons/best_Bild<I>_Query<Q>.png`

CSV-Schema:
```csv
layer_idx,image_idx,query_idx,iou,threshold,positives,heatmap_path,overlay_path
0,1,1,0.234567,0.1234,1234,layer0/heatmaps/Bild1_Query1.png,layer0/comparisons/best_Bild1_Query1.png
```

Overlay-Farbcodierung:
- **Blau (BGR=255,0,0)**: Überschneidung (Maske ∧ Binär-Heatmap)
- **Rot (BGR=0,0,255)**: Ground-Truth-Maske nur
- **Gelb (BGR=0,255,255)**: Binär-Heatmap nur
- **Schwarz**: Hintergrund

## Bildvorverarbeitung (relevant für Größen)

- Quelle: `load_and_preprocess_image()` in `weights_extraction_transformer_decoder.py`
- Farbformat: OpenCV liest BGR → Konvertierung zu RGB
- Resize: Standard auf 800×800 für MaskDINO-Input
- ImageNet-Normalisierung: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- Tensor-Format: `[C, H, W]` mit `C=3, H=W=800`, dtype `float32`
- Input-Dictionary: `{"image": tensor, "height": orig_h, "width": orig_w, "file_name": basename}`

## Modell-/Config-Parameter (kurze Essentials)

- Arch: `MaskDINO` mit ResNet-50-Backbone
- Decoder-Layer: `cfg.MODEL.MaskDINO.DEC_LAYERS = 3` → Layer-Indices 0, 1, 2
- Num-Queries: `cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 300`
- Hidden-Dim: `cfg.MODEL.MaskDINO.HIDDEN_DIM = 256`
- Input-Format: `RGB`
- Device: `cpu` (Standard in diesem Setup)

## Namen/Regex für Hook-Registrierung

- Zielmodule für Decoder-Hooks:
	- Pattern: `predictor.decoder.layers.` mit Endung `.0`, `.1`, `.2`
	- Nur Hauptlayer-Ebene registrieren (nicht Sub-Module)

- Zielmodule für Pixel-Embedding-Hooks:
	- Patterns: `backbone.res`, `backbone.layer`, `input_proj`, `pixel_decoder.input_proj`
	- Spezifische Endungen: `.res2`, `.res3`, `.res4`, `.res5`, `input_proj.0/1/2`

## IoU-Berechnungs-Pipeline

### Input-Größen-Bestimmung (Priorität):

1. **Exakte Metadaten**: `input_h`, `input_w` aus Pixel-Embedding-Metadaten
2. **Stride-basiert**: `embed_h/w * stride` (falls stride in Metadaten)
3. **Standard-Heuristik**: `embed_h/w * 32` (z.B. 25 × 32 = 800)

### Schwellenwert-Strategien:

- `"percentile"`: `threshold_value`-tes Perzentil (Standard: 80.0)
- `"mean"`: Durchschnittswert der Response-Map
- `"median"`: Median der Response-Map  
- `"absolute"`: Fester Wert über `threshold_absolute`

### Response-Map-Berechnung:

1. **Dot-Product**: `query_features @ pixel_embedding.reshape(C, H*W)` → `(H*W,)`
2. **Reshape**: `(H*W,)` → `(H, W)` Response-Map
3. **Skalierung**: Bilineare Interpolation auf Input-Größe `(H_in, W_in)`
4. **Binarisierung**: Response-Map ≥ threshold → bool-Maske
5. **IoU**: Intersection-over-Union mit Ground-Truth-Maske

## Quick-Checkliste für manuelle IoU-Berechnung

1. **Daten laden**:
   - Query-Features aus `layer<N>/Query.csv`, Zeile für gewünschte `(Bild, Query)`
   - Pixel-Embedding aus `pixel_embeddings/pixel_embed_Bild<NNNN>.npy`
   - Ground-Truth-Maske aus `myThesis/image/colours/rot.png` (skaliert auf Input-Größe)

2. **Berechnung**:
   ```python
   response = np.dot(query_features, pixel_embedding.reshape(256, -1))
   response_map = response.reshape(H, W)
   response_scaled = cv2.resize(response_map, (W_in, H_in))
   binary_map = response_scaled >= threshold
   iou = intersection_area / union_area
   ```

3. **Export**: Nutze `save_heatmap_png()` und `save_overlay_comparison()` für Visualisierungen

## Pfade (ab Repo-Root)

- Setup/Handover: `myThesis/decoder/nd_on_transformer_decoder.py`
- Hook-Registrierung/Extraktion: `myThesis/decoder/weights_extraction_transformer_decoder.py`  
- IoU-Kernlogik: `myThesis/decoder/iou_core_decoder.py`
- IoU-Pipeline: `myThesis/decoder/calculate_IoU_for_decoder.py`
- Ausgaben:
	- `myThesis/output/decoder/layer<N>/Query.csv`
	- `myThesis/output/decoder/pixel_embeddings/pixel_embed_Bild<NNNN>.npy`
	- `myThesis/output/decoder/pixel_embeddings/metadata_Bild<NNNN>.json`
	- `myThesis/output/decoder/iou_results/layer<N>/iou_sorted.csv`
	- `myThesis/output/decoder/iou_results/layer<N>/heatmaps/*.png`
	- `myThesis/output/decoder/iou_results/layer<N>/comparisons/*.png`

## Edge Cases & Logs

- **Fehlende Pixel-Embeddings**: IoU-Berechnung wird übersprungen, Generator liefert keine Einträge
- **Batch-Größe > 1**: Nur `batch_index=0` wird verarbeitet (mit Warnung)
- **Input-Größe-Validierung**: Mindest-/Maximalgrößen werden geprüft und korrigiert
- **Unerwartete Hidden-State-Dimensionen**: 2D-Fallback oder Überspringen mit Warnung
- **Hook-Registrierung**: Automatische Cleanup bei Fehlern über `finally`-Block
