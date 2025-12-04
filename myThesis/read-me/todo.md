# LRP für MaskDINO Transformer - Zusammenfassung & TODOs

*Stand: 3. Dezember 2025*

---

## 1. Architektur-Übersicht

### MaskDINO Transformer-Struktur
```
Backbone (ResNet50) → Pixel Decoder → Encoder (6 Layers) → Decoder (3 Layers) → Predictions
```

**Encoder:**
- 6 Transformer-Encoder-Layers
- Jeder Layer: `self_attn` (MSDeformAttn) + `norm1` + FFN + `norm2`
- Input: Multi-Scale Feature Maps (13.125 Tokens = 100×100 + 50×50 + 25×25)
- Aktivierungen-Shape: `(B, T, C) = (1, 13125, 256)`

**Decoder:**
- 3 Transformer-Decoder-Layers  
- Jeder Layer: `cross_attn` (MSDeformAttn) + `norm1` + `self_attn` (MHA) + `norm2` + FFN + `norm3`
- Input: 300 Object Queries
- **WICHTIG:** Aktivierungen-Shape ist `(T, B, C) = (300, 1, 256)` (nicht `(B, T, C)`!)

### LRP-fähige Module
| Modul-Typ | Anzahl | LRP-Klasse |
|-----------|--------|------------|
| LayerNorm | 23 | `LRP_LayerNorm` |
| MultiheadAttention | 3 | `LRP_MultiheadAttention` |
| MSDeformAttn | 9 | `LRP_MSDeformAttn` |

---

## 2. Gelöste Probleme

### ✅ OOM-Problem (Exit Code 137)
**Ursache:** Zu viel Memory-Verbrauch während der LRP-Rückpropagation

**Lösung (in `lrp_controller.py`):**
1. `gc.collect()` nach jedem propagierten Layer
2. Aktivierungen sofort nach Propagation löschen: `module.activations.clear()`
3. Nur die letzten 3 Layer-Relevanzen speichern statt alle
4. `torch.no_grad()` statt `torch.inference_mode()` für Forward-Pass

### ✅ Shape-Mismatch Decoder
**Ursache:** `R_start` hatte Shape `(B, T, C) = (1, 300, 256)`, aber Decoder-Aktivierungen haben Shape `(T, B, C) = (300, 1, 256)`

**Lösung (in `lrp_controller.py`, Methode `run_all_queries`):**
```python
# R_start im Format (T, B, C) = (300, 1, 256) erstellen
R_start_full = torch.zeros((num_total_queries, B, C), ...)
R_start_full[query_idx, 0, :] = decoder_output.squeeze()
```

### ✅ Shape-Mismatch in MSDeformAttn
**Ursache:** `R_out` kam im Format `(T, B, C)`, aber `msdeform_attn_lrp` erwartet `(B, T, C)`

**Lösung (in `lrp_propagators.py`, Funktion `propagate_msdeformattn`):**
```python
# Automatische Transposition wenn nötig
if R_out.shape[0] == T_loc and R_out.shape[1] == B_loc:
    R_out = R_out.transpose(0, 1).contiguous()
```

### ✅ Konservierungsfehler (~8x)
**Ursache:** In `splat_to_level` wurde Relevanz nicht durch Anzahl der Attention-Heads dividiert

**Lösung (in `lrp_deform_ops.py`):**
```python
R_expanded = R_expanded / H_heads  # Normalisierung für Konservierung
```

### ✅ None-Rückgabe bei LayerNorm
**Ursache:** `_align_shapes_for_layernorm` gab `None` zurück statt des Original-Tensors

**Lösung (in `lrp_propagators.py`):**
- Funktion gibt immer `R_out` zurück, niemals `None`

---

## 3. Gelöste Probleme (Update 03.12.2025)

### ✅ RuntimeError bei Decoder-Propagation durch mehrere Layers (GELÖST)
**Symptom:**
```
RuntimeError: The size of tensor a (13125) must match the size of tensor b (300) at non-singleton dimension 2
```

**Ursache:** 
- `cross_attn` propagiert Relevanz von 300 Queries → 13125 Encoder-Tokens
- Das Ergebnis hat Shape `(B, 13125, C)`
- Aber der nächste Layer (`norm1` von vorherigem Decoder-Layer) erwartet wieder `(300, 1, 256)`

**Lösung (in `lrp_controller.py`):**
Die Decoder-LRP wird jetzt bei `cross_attn` gestoppt, da die Shape-Transformation konzeptionell korrekt ist - die Relevanz fließt von den Decoder-Queries zu den Encoder-Tokens.

Neue Parameter in `backward_pass()`:
```python
stop_after_cross_attn: bool = True  # Stoppe nach cross_attn bei Decoder-LRP
clear_activations: bool = True       # Aktivierungen nach Propagation löschen
```

Der Decoder-LRP propagiert jetzt:
```
norm3 → norm2 → self_attn → norm1 → cross_attn → STOPP
```
Output: Relevanz auf Encoder-Tokens (Shape: `(1, 13125, 256)`)

### ✅ Aktivierungen verschwinden bei run_all_queries (GELÖST)
**Symptom:** Bei Query 1+ wurden Warnungen "Keine Aktivierungen" gezeigt.

**Ursache:** Aktivierungen wurden nach jedem Layer-Propagation gelöscht (`module.activations.clear()`), aber für `run_all_queries` brauchten wir sie für alle Queries.

**Lösung:** Neuer Parameter `clear_activations=False` in `backward_pass()`, der in `run_all_queries()` verwendet wird.

---

## 4. Code-Struktur

### Wichtige Dateien
```
myThesis/lrp/
├── calc_lrp.py              # Haupt-Einstiegspunkt für LRP-Analyse
├── do/
│   ├── lrp_controller.py    # Zentrale Steuerung (forward/backward pass)
│   ├── lrp_propagators.py   # Propagations-Funktionen je Modul-Typ
│   ├── lrp_param_modules.py # LRP-fähige Module (LRP_LayerNorm, LRP_MHA, LRP_MSDeformAttn)
│   ├── lrp_param_base.py    # Basis-Klassen (LRPActivations, LRPModuleMixin)
│   ├── lrp_deform_ops.py    # Bilineares Splatting für MSDeformAttn
│   ├── lrp_rules_deformable.py # msdeform_attn_lrp Implementierung
│   ├── lrp_attn_prop.py     # Attention-LRP (Value-Path, Q/K-Path)
│   └── model_graph_wrapper.py # Graph-Navigation für Propagations-Reihenfolge
```

### LRP-Fluss
```
1. prepare()         → Module durch LRP-Versionen ersetzen
2. forward_pass()    → Aktivierungen in jedem Modul speichern
3. backward_pass()   → Relevanz layer-by-layer zurückpropagieren
4. cleanup()         → Aktivierungen löschen, LRP-Modus deaktivieren
```

---

## 5. Nächste Schritte (TODO)

### Priorität 1: ✅ Decoder-LRP korrigieren - ERLEDIGT
- [x] Entscheiden: Decoder-LRP stoppt bei `cross_attn`
- [x] `which_module="decoder"` stoppt nach erstem `cross_attn`
- [x] Output = Relevanz auf Encoder-Tokens (13125 Dimensionen)

### Priorität 2: ✅ Ergebnis-Validierung - ERLEDIGT
- [x] Validierungsskript erstellt: `myThesis/lrp/validate_lrp_results.py`
- [x] Prüft: Normalisierung (sum ≈ 1), keine NaN/Inf, sinnvolle Verteilung
- [x] Alle 3 vorhandenen CSV-Dateien validiert und bestanden

### Priorität 3: ✅ Performance-Optimierung - ERLEDIGT
- [x] `BatchLRPProcessor` für Memory-optimierte Batch-Verarbeitung
- [x] `MemoryOptimizedLRP` Context Manager für einzelne Analysen
- [x] `LRPPerformanceConfig` für konfigurierbare Performance-Parameter
- [x] `estimate_memory_requirements()` zur Speicherschätzung
- [x] Generator-basierte Verarbeitung für niedrigen Speicherverbrauch


---

## 6. Bekannte Tensor-Shapes

| Kontext | Shape | Bedeutung |
|---------|-------|-----------|
| Encoder Input | `(1, 13125, 256)` | B=1, T=13125 Tokens, C=256 Features |
| Encoder Aktivierungen | `(1, 13125, 256)` | Gleich wie Input |
| Decoder Query Input | `(300, 1, 256)` | **T=300, B=1, C=256** (transponiert!) |
| Decoder Aktivierungen (norm) | `(300, 1, 256)` | T, B, C Format |
| Decoder self_attn Input | `(1, 300, 256)` | B, T, C Format (Tuple für Q,K,V) |
| cross_attn sampling_locations | `(1, 300, 8, 3, 4, 2)` | B, T, H_heads, L_levels, P_points, 2 |
| cross_attn Output | `(1, 13125, 256)` | Relevanz auf Encoder-Tokens |

---

## 7. Hilfreiche Debug-Befehle

```bash
# Nur Encoder testen
python -m myThesis.test 2>&1 | grep -E "(Encoder|Ergebnisse|Error)"

# Mit vollem Debug-Output
LOGLEVEL=DEBUG python -m myThesis.test 2>&1 | tail -100

# Memory-Verbrauch beobachten
python -m myThesis.test 2>&1 | grep -E "(Memory|MB)"
```

---

## 8. Referenzen

- MaskDINO Paper: [arXiv:2206.02777](https://arxiv.org/abs/2206.02777)
- LRP Tutorial: [Montavon et al., 2019](https://doi.org/10.1016/j.dsp.2017.10.011)
- Transformer LRP: [Chefer et al., 2021](https://arxiv.org/abs/2103.15679)
