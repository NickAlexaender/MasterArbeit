## Referenz: Datenflüsse, Strukturen und Speicherorte (Transformer-Encoder)

Diese Datei fasst alle relevanten Strukturen aus `nd_on_transformer_encoder.py` (Setup/Handover) und `weights_extraction_transformer_encoder.py` (Hooks/Extraktion/Export) zusammen – optimiert für schnelle Umformung und spätere IoU-Berechnung.

### Einstiegspunkte

- Build & Handover: `nd_on_transformer_encoder.py`
	- Baut MaskDINO-Konfiguration und Modell, lädt finetuned Gewichte.
	- Sammelt Bildpfade (aktuell 1 Beispielbild) und ruft
		`accept_weights_model_images(weights_path, model, image_list)` auf.

- Extraktion & Export: `weights_extraction_transformer_encoder.py`
	- Registriert Hooks auf allen Transformer-Encoder-Layern und auf dem Pixel-Decoder-Transformer (für Shapes).
	- Führt pro Bild einen Forward aus, sammelt Layer-Outputs, speichert optionale `shapes.json`, exportiert CSV pro Layer.

## In-Memory-Datenstrukturen

### Modell-Attribute (zur Laufzeit gesetzt)

- `model._encoder_feature_buffers: Dict[str, torch.Tensor]`
	- Gefüllt durch Forward-Hooks auf Modulen mit Name-Muster:
		`sem_seg_head.pixel_decoder.transformer.encoder.layers.<idx>`
	- Key = genauer Modulname (z. B. `...encoder.layers.0`), Value = Output-Tensor (detach, cpu).

- `model._encoder_hook_handles: List[HookHandle]`
	- Zum Entfernen der Hooks am Ende.

- `model._last_encoder_shapes: Dict[str, List] | None`
	- Wird per Hook auf `...pixel_decoder.transformer` gesetzt.
	- Inhalt (falls verfügbar):
		- `spatial_shapes: List[List[int]]` – Liste von [H, W] pro Feature-Level.
		- `level_start_index: List[int]` – Startindex je Level im flach zusammengefügten Token-Array.

### Rückgabe der Extraktion

- `results: Dict[str, Dict[str, np.ndarray]]`
	- Top-Level-Key = `image_path` (str)
	- Value = `per_layer: Dict[layer_name (str), np.ndarray]`
		- Jede Ebene enthält den vom Hook abgegriffenen Output als `np.float16`.
		- Typische Feature-Dim D = 256 (MaskDINO Hidden-Dim).

### Standardisierung von Shapes (`to_bdn`)

Ziel: beliebige Layer-Outputs nach `[B, D, N]` transformieren, wobei

- B = Batch (typisch 1),
- D = Feature-Dimension (typisch 256),
- N = Anzahl Tokens (z. B. H*W oder Summe über Level).

Fälle, die erkannt werden:

- `[B, C, H, W]` → `[B, D=C, N=H*W]` (ggf. von `[B, H, W, C]` nach `[B, C, H, W]` transponiert, wenn letzte Achse 256 ist).
- `[B, N, D]` oder `[B, D, N]` → nach `[B, D, N]` transponiert.
- `[N, D]` oder `[D, N]` → zu `[1, D, N]` erweitert.
- Anderes → Fallback: flatten zu `[1, 1, N]`.

Hinweise:

- Wenn `D != 256`, wird gewarnt, aber dennoch exportiert.
- Bei `B > 1` wird im CSV nur `b=0` exportiert (Warnung in Log).

## Persistierte Artefakte (Speicherorte & Schemas)

Basisverzeichnis: `myThesis/output/encoder`

Pfadableitung:

- `project_root = <.../myThesis>` (Elternverzeichnis der Python-Datei)
- `base_out = project_root / "output" / "encoder"`

### 1) Shapes-JSON pro Bild

- Speicherort: `output/encoder/<image_id>/shapes.json`
	- `image_id = basename(img_path) ohne Erweiterung`
	- Verzeichnis wird bei Bedarf erstellt.
- Erzeugt nur, wenn Hook die Shapes erfassen konnte; sonst wird das JSON übersprungen.

Schema (Beispiel):

```json
{
	"image_id": "new_21_png_jpg.rf.d0c932...",
	"image_path": "/abs/path/to/image.jpg",
	"orig_size": [H0, W0],              // Originalbildgröße
	"input_size": [H_in, W_in],         // nach Resize (cfg.INPUT.MIN_SIZE_TEST/MAX_SIZE_TEST)
	"spatial_shapes": [[H1, W1], [H2, W2], ...],
	"level_start_index": [s1, s2, ...], // Startindex je Level im Token-Array
	"N_tokens": N                       // Sum(Hl*Wl)
}
```

Verwendung für Reassemblierung:

- Token-Reihenfolge entspricht Level-Paketen: Level i → Slice `[s_i : s_i + H_i*W_i]`.
- Formwiederherstellung pro Level:
	- Aus `[B, D, N]` → Tokens des Levels nach `[B, D, H_i, W_i]` reshapen.

### 2) CSV pro Layer (über alle Bilder aggregiert)

- Speicherort: `output/encoder/layer<lidx>/feature.csv`
	- `<lidx>` aus Layer-Namen extrahiert via Regex: `\.encoder\.layers\.(\d+)`.
- Header:
	- `Name, Gewicht 1, Gewicht 2, ..., Gewicht N`
	- `N` wird aus dem ersten gesehenen Sample des Layers bestimmt (`to_bdn`).
- Zeilen:
	- Name: `"Bild<img_idx>, Feature<fidx>"`
	- Werte: Sequenzlänge `N` (Tokens), Typ: float (aus `np.float16` konvertiert beim Schreiben).

Interpretation:

- Jede Zeile beschreibt die Token-Aktivierung eines einzelnen Features (Kanal fidx) für Bild `img_idx` in diesem Layer.
- Um räumliche Karten zurückzugewinnen, nutze `shapes.json` und reshaping wie oben beschrieben.

## Bildvorverarbeitung (relevant für Größen)

- Quelle: `_preprocess_image()` in `weights_extraction_transformer_encoder.py`.
- Farbformat: OpenCV liest BGR; wenn `input_format == "RGB"` → Konvertierung zu RGB.
- Resize-Skala: `scale = min(MIN_SIZE_TEST / min(h, w), MAX_SIZE_TEST / max(h, w))`.
- Standard aus Config: `MIN_SIZE_TEST = 800`, `MAX_SIZE_TEST = 1333`.
- Tensorform: `[C, H_in, W_in]`, dtype `float16` (später zu `float32` vor Modell-Forward gecastet), Wertebereich 0–255.
- Detectron2-Inputs: `[{"image": Tensor(float32, device), "height": H_in, "width": W_in}]`.

## Modell-/Config-Parameter (kurze Essentials)

- Arch: `MaskDINO` mit ResNet-50-Backbone, Pixel-Decoder `MaskDINOEncoder`.
- Encoder-Layer: `cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 6` → erwartet Layer-Indices 0..5.
- Feature-Level: `TOTAL_NUM_FEATURE_LEVELS = 3`, Encoder-In-Features: `[res3, res4, res5]`.
- Hidden-Dim: `cfg.MODEL.MaskDINO.HIDDEN_DIM = 256` (erwartete D).
- Input-Format: `RGB`.
- Gerät in diesem Setup: `cpu`.

## Namen/Regex

- Zielmodule für Hooks: Name beginnt mit
	`sem_seg_head.pixel_decoder.transformer.encoder.layers.` und endet mit `.<idx>` (reine Zahl).
- Layerindex-Regex: `\.encoder\.layers\.(\d+)`.

## Reassemblierung der Token → Karten (Kurzrezept)

Voraussetzungen:

- Layer-Output in `[B, D, N]` (über `to_bdn` herstellen).
- Shapes aus `output/encoder/<image_id>/shapes.json` laden.

Schritte pro Level i:

1. Hole `H_i, W_i = spatial_shapes[i]`, `s_i = level_start_index[i]`, `n_i = H_i * W_i`.
2. Slice Tokens: `T_i = out[:, :, s_i : s_i + n_i]` → `[B, D, n_i]`.
3. Reshape zu Karte: `T_i.reshape(B, D, H_i, W_i)`.

Damit erhältst du pro Feature-Kanal eine räumliche Karte auf dem Level-Gitter. Für IoU-Berechnungen zwischen zwei Features gleiche Level wählen und ggf. auf gemeinsame Auflösung bringen.

## Edge Cases & Logs

- Wenn `shapes.json` nicht erzeugt werden konnte (Hook liefert nichts), wird dies geloggt und Shapes fehlen.
- Bei `B > 1` wird nur `b=0` ins CSV exportiert (Loghinweis).
- `D != 256` wird geloggt, Export erfolgt dennoch.

## Quick-Checkliste für spätere IoU-Berechnung

- Lade `feature.csv` des gewünschten Layers und `shapes.json` des Bildes.
- Wähle Feature fidx → Zeile `BildX, Feature<fidx>`.
- Bringe die Zeilenwerte nach `[1, D, N]` (hier `D=1` für einzelnes Feature) und reassembliere mit `spatial_shapes`/`level_start_index` zu Karten je Level.
- Normalisiere/threshold ggf. die Karte, skaliere auf gemeinsame Größe, berechne IoU zu Referenzmaske oder anderem Feature.

## Pfade (ab Repo-Root)

- Extraktor: `myThesis/encoder/weights_extraction_transformer_encoder.py`
- Handover/Setup: `myThesis/encoder/nd_on_transformer_encoder.py`
- Ausgaben:
	- `myThesis/output/encoder/<image_id>/shapes.json`
	- `myThesis/output/encoder/layer<lidx>/feature.csv`

