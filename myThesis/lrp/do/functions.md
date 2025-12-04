# LRP-Modul-Dokumentation

Dieses Verzeichnis enthält alle Module für Layer-wise Relevance Propagation (LRP) zur Analyse von MaskDINO.

---

## Übersicht der Module

| Modul | Kategorie | Beschreibung |
|-------|-----------|--------------|
| `__init__.py` | Paket | Zentraler Einstiegspunkt mit allen Exporten |
| `cli.py` | CLI | Command-Line-Interface Parser |
| `compat.py` | Kompatibilität | Pillow/NumPy Monkey-Patches |
| `config.py` | Konfiguration | LRP-Konstanten und Defaults |
| `config_build.py` | Konfiguration | MaskDINO-Modellkonfiguration |
| `io_utils.py` | I/O | Bilddatei-Sammlung |
| `lrp_analysis.py` | High-Level | Batch-Verarbeitung und Context Manager |
| `lrp_attn_prop.py` | Attention | Lineare Algebra für Attention-LRP |
| `lrp_attn_structs.py` | Attention | AttnCache Datenstruktur |
| `lrp_controller.py` | Core | Zentraler LRP-Controller |
| `lrp_deform_capture.py` | Deformable | Monkey-Patching für MSDeformAttn |
| `lrp_deform_ops.py` | Deformable | Geometrische Operationen |
| `lrp_param_base.py` | Module | LRPActivations und Mixin-Klassen |
| `lrp_param_modules.py` | Module | LRP-fähige PyTorch-Module |
| `lrp_param_utils.py` | Module | Modul-Swap und Verwaltung |
| `lrp_propagators.py` | Propagation | Layer-Level Propagationsfunktionen |
| `lrp_rules_attention.py` | Regeln | Multi-Head Attention LRP |
| `lrp_rules_deformable.py` | Regeln | Deformable Attention LRP |
| `lrp_rules_standard.py` | Regeln | Basis-LRP-Regeln (ε, γ, α-β) |
| `lrp_softmax.py` | Regeln | Softmax-Rückpropagation |
| `lrp_structs.py` | Datenstrukturen | LRPResult und LayerRelevance |
| `model_graph_wrapper.py` | Graph | Modell-Linearisierung |
| `param_patcher.py` | Wrapper | Re-Export aller LRP-Module |
| `tensor_ops.py` | Utilities | Tensor-Transformationen |

---

## Detaillierte Modulbeschreibungen

### myThesis/lrp/do/__init__.py

**Kategorie:** Paket-Initialisierung

Zentraler Einstiegspunkt für das LRP-Paket. Importiert und re-exportiert alle wichtigen Klassen, Funktionen und Konstanten aus den Untermodulen. Stellt sicher, dass Kompatibilitäts-Patches (`compat.py`) zuerst geladen werden.

**Wichtige Exports:**
- Hauptklassen: `LRPController`, `LRPResult`, `LayerRelevance`, `LRPAnalysisContext`
- Propagatoren: `propagate_layer`, `propagate_linear`, `propagate_multihead_attention`
- LRP-Module: `LRP_Linear`, `LRP_LayerNorm`, `LRP_MultiheadAttention`, `LRP_MSDeformAttn`
- Regeln: `lrp_epsilon_rule`, `lrp_gamma_rule`, `lrp_full_attention`, `msdeform_attn_lrp`
- Utilities: `safe_divide`, `rearrange_activations`, `build_target_relevance`

---

### myThesis/lrp/do/cli.py

**Kategorie:** Command-Line-Interface

Stellt den CLI-Parser für `calc_lrp` bereit. Definiert alle Kommandozeilenargumente für die LRP-Analyse.

**Argumente:**
- `--images-dir`: Ordner mit Eingabebildern (jpg/png)
- `--layer-index`: 1-basierter Index des Encoder-/Decoder-Layers
- `--feature-index`: Kanalindex (Feature) im gewählten Layer
- `--target-norm`: Normierung der Zielrelevanz (`sum1`, `sumT`, `none`)
- `--lrp-epsilon`: Epsilon-Stabilisator für ε/z+-Regel
- `--output-csv`: Pfad zur CSV-Ausgabedatei
- `--which-module`: Wähle `encoder` oder `decoder`
- `--method`: Attributionsmethode (`gradinput` oder `lrp`)
- `--weights-path`: Optionaler Pfad zu Modellgewichten

---

### myThesis/lrp/do/compat.py

**Kategorie:** Kompatibilität

Enthält Monkey-Patches für Pillow und NumPy zur Sicherstellung der Rückwärtskompatibilität. Muss vor allen anderen schweren Imports geladen werden.

**Patches:**
- **Pillow:** Fügt fehlende Resampling-Konstanten hinzu (`LINEAR`, `CUBIC`, `LANCZOS`, `NEAREST`)
- **NumPy:** Fügt entfernte Typ-Aliase hinzu (`np.bool`, `np.int`, `np.float`, `np.complex`)

**Wichtig:** Dieses Modul importiert weder `torch` noch andere interne Module.

---

### myThesis/lrp/do/config.py

**Kategorie:** Konfiguration

Definiert Konstanten und Default-Werte für die lokale LRP-Analyse (L-1 → L). Alle Werte sind rückwärtskompatibel und erfordern keine CLI-Änderungen.

**Konstanten:**
- `TARGET_TOKEN_IDX = 0`: Ziel-Token t* (Decoder-Query bzw. Encoder-Token)
- `USE_SUBLAYER = "self_attn"`: Sublayer-Auswahl (`self_attn`, `cross_attn`, `ffn`)
- `MEASUREMENT_POINT = "post_res"`: Messpunkt (`pre_res`, `post_res`)
- `RESIDUAL_SPLIT = "zsign"`: Residual-Aufteilung (`energy`, `dotpos`, `zsign`)
- `LN_RULE = "zsign"`: LayerNorm-Regel (`zsign`, `abs-grad-xmu`, `xmu`)
- `ATTN_QK_SHARE = 0.0`: ρ-Anteil für Q/K (0.0 ⇒ nur Value)
- `SIGN_PRESERVING = True`: Vorzeichen über alle Regeln bewahren
- `DETERMINISTIC = True`: Deterministische Berechnung
- `SEED = 1234`: Random Seed

---

### myThesis/lrp/do/config_build.py

**Kategorie:** Modellkonfiguration

Baut die Detectron2/MaskDINO-Konfiguration für Inferenz. Importiert `add_maskdino_config` und definiert alle Modellparameter.

**Wichtige Konstanten:**
- `FINETUNED_WEIGHTS`: Pfad zu den finetuned Gewichten (23 Klassen)
- `COCO_WEIGHTS_FALLBACK`: Fallback auf COCO-Gewichte (80 Klassen)
- `DEFAULT_WEIGHTS`: Aktiver Gewichtspfad (via `MYTHESIS_WEIGHTS` überschreibbar)

**Hauptfunktion:**
- `build_cfg_for_inference(device, weights_path)`: Erzeugt eine vollständige Inferenz-Konfiguration inkl. Backbone (ResNet-50), SemSeg-Head (MaskDINOHead), Encoder (6 Layer), Decoder (3 Layer), und Test-Einstellungen.

---

### myThesis/lrp/do/io_utils.py

**Kategorie:** I/O-Hilfsfunktionen

Kleine Sammlung von Funktionen für Dateisystem-Operationen.

**Funktionen:**
- `collect_images(images_dir)`: Sammelt alle Bilddateien (jpg, jpeg, png, bmp) in einem Verzeichnis und gibt sie sortiert zurück.

---

### myThesis/lrp/do/lrp_controller.py

**Kategorie:** Core

Das zentrale Gehirn der LRP-Analyse für MaskDINO. Die Klasse `LRPController` vereint die zuvor getrennte Logik aus "Analyse" und "Engine" und löst State-Management-Probleme durch eine einheitliche Kontextverwaltung. Der **Forward Pass** führt das Modell im Evaluierungsmodus aus und speichert Aktivierungen ($a_j$) lokal in den LRP-Modul-Instanzen. Der **Backward Pass** iteriert durch den Modell-Graphen in umgekehrter topologischer Reihenfolge, ruft die entsprechenden Propagatoren auf und übergibt den Relevanz-Tensor an den nächsten Layer. Die **Aggregation** verwaltet die Summation von Relevanzwerten aus Skip-Verbindungen und stellt die Konservierungseigenschaft sicher.

---

### myThesis/lrp/do/tensor_ops.py

**Kategorie:** Utilities

Zustandslose Hilfsfunktionen für numerische Stabilität und Tensor-Transformationen.

**Hauptfunktionen:**
- `safe_divide(numerator, denominator, eps, signed_eps)`: Sichere Division zur Vermeidung von NaN/Inf in LRP-Berechnungen. Stabilisiert durch vorzeichenerhaltende Epsilon-Addition.
- `rearrange_activations(tensor, source_format, target_format, spatial_shape)`: Konvertiert zwischen Formaten:
  - `NCHW` ↔ `NLC` (CNN ↔ Transformer)
  - `NC` ↔ `NLC` (Flat ↔ Sequenz)
- `compute_jacobian(func, x)`: Berechnet die Jacobi-Matrix für GTI-Methode
- `gradient_times_input(func, x, target_indices)`: GTI-Relevanz für instabile Layer (z.B. LayerNorm)
- `build_target_relevance(layer_output, feature_index, token_reduce, target_norm, index_axis)`: Erzeugt Start-Relevanz $R_{out}$ für LRP
- `aggregate_channel_relevance(R_in)`: Aggregiert Eingangsrelevanz zu Vektor $(C_{in},)$

---

### myThesis/lrp/do/param_patcher.py

**Kategorie:** Wrapper

Ein Kompatibilitäts-Wrapper und zentraler Einstiegspunkt für das Patchen des Modells. Er re-exportiert alle LRP-fähigen Module (`LRP_Linear`, etc.) und Utilities (`swap_module_inplace`, `prepare_model_for_lrp`) aus den aufgeteilten Untermodulen. Dies ermöglicht das automatische In-Place-Ersetzen von Standard-Modulen durch LRP-fähige Varianten, ohne die Architekturdefinition ändern zu müssen.

---

### myThesis/lrp/do/model_graph_wrapper.py

**Kategorie:** Graph

Linearisiert die komplexe verschachtelte Struktur von MaskDINO/Detectron2-Modellen in einen flachen `ModelGraph`. Dies ermöglicht eine geordnete Iteration (insbesondere rückwärts für LRP) und das korrekte Routing von Relevanz zwischen Decodern, Encodern und dem Backbone. Der Graph erkennt Layer-Typen und Abhängigkeiten, um den Relevanzfluss auch über Skip-Connections und Cross-Attention hinweg korrekt zu steuern.

**Klassen:**
- `ModelGraph`: Container für die linearisierte Modellstruktur
- `LayerNode`: Knoten im Graphen mit Referenz auf das Modul
- `LayerType`: Enum für Layer-Kategorisierung

---

### myThesis/lrp/do/lrp_structs.py

**Kategorie:** Datenstrukturen

Definiert die zentralen Datenstrukturen für die LRP-Analyse als reine Dataclasses ohne Geschäftslogik.

**Klassen:**
- `LRPResult`: Kapselt das Gesamtergebnis einer Analyse
  - `R_input`: Relevanz auf der Eingabe (Pixel-Level) $(B, C, H, W)$
  - `R_per_layer`: Dict mit Relevanz pro Layer-Name
  - `conservation_errors`: Liste von Konservierungsfehlern pro Layer
  - `metadata`: Zusätzliche Metadaten der Analyse
- `LayerRelevance`: Container für die Relevanzwerte eines einzelnen Layers
  - `R_out`: Ausgabe-Relevanz (eingehend für Rückpropagation)
  - `R_in`: Eingabe-Relevanz (Ergebnis der Rückpropagation)
  - `R_skip`: Relevanz des Skip-Pfads (bei Residuals)
  - `R_transform`: Relevanz des Transform-Pfads

---

### myThesis/lrp/do/lrp_softmax.py

**Kategorie:** Regeln

Implementiert die mathematisch anspruchsvollen LRP-Regeln für Softmax-Layer ($A = \text{softmax}(S)$). Bietet verschiedene Rückpropagations-Strategien: Die Gradient-Regel (Taylor-Expansion 1. Ordnung), die $\epsilon$-Regel für stabilisierte Beiträge und eine exakte Jacobian-basierte Berechnung. Diese Funktionen ermöglichen den Fluss von Relevanz von den normalisierten Attention-Gewichten zurück zu den rohen Scores.

---

### myThesis/lrp/do/lrp_rules_standard.py

**Kategorie:** Regeln

Enthält die Standard-LRP-Regeln zur Rückpropagation durch lineare Schichten.

**Hauptfunktionen:**
- `lrp_epsilon_rule`: $\epsilon$-Regel für numerische Stabilität
- `lrp_gamma_rule`: $\gamma$-Regel zur Verstärkung positiver Beiträge
- `lrp_alpha_beta_rule`: $\alpha$-$\beta$-Regel für kontrollierte Zerlegung
- `lrp_linear`: LRP durch lineare Layer
- `residual_split`: Aufteilung der Relevanz auf Skip-Verbindungen
- `layernorm_lrp`: LRP durch LayerNorm (unter Berücksichtigung von Normalisierungsstatistiken)
- `layernorm_backshare`: Alternative LayerNorm-Strategie

---

### myThesis/lrp/do/lrp_rules_deformable.py

**Kategorie:** Regeln

Der Einstiegspunkt für LRP in Deformable Attention Layern (`MSDeformAttn`). Implementiert die High-Level-Logik zur Verteilung der Relevanz auf Sampling-Lokationen und Value-Features. Da MSDeformAttn Werte mittels bilinearer Interpolation sampelt, muss die Relevanz entsprechend auf die benachbarten Pixel verteilt werden ("Splatting").

**Hauptfunktionen:**
- `msdeform_attn_lrp`: Haupt-LRP-Funktion für MSDeformAttn
- `msdeform_attn_lrp_with_value`: LRP inkl. Value-Pfad
- `deform_value_path_lrp`: LRP für den Value-Pfad
- `bilinear_splat_relevance`: Relevanz-Splatting mit bilinearer Interpolation
- `compute_bilinear_weights`: Berechnung der Interpolationsgewichte
- `compute_pixel_relevance_map`: Pixel-Level Relevanz-Map
- `compute_multiscale_relevance_map`: Multi-Scale Relevanz-Map
- `attach_msdeformattn_capture`: Hook-Registrierung für Capture

---

### myThesis/lrp/do/lrp_rules_attention.py

**Kategorie:** Regeln

Orchestriert die LRP-Pipeline für Standard Multi-Head Attention (Self- und Cross-Attention). Zerlegt die Formel $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ in separate Pfade.

**Hauptfunktionen:**
- `lrp_full_attention`: Komplette Attention-LRP-Pipeline
- `lrp_attention_value_path`: LRP durch $O = A \cdot V$
- `lrp_attention_to_weights`: Relevanz zu Attention-Gewichten
- `lrp_softmax`: Softmax-Rückpropagation
- `lrp_attention_qk_path`: LRP durch $S = Q \cdot K^T$

**Datenstrukturen:**
- `AttnCache`: Container für Attention-Zwischenergebnisse

---

### myThesis/lrp/do/lrp_propagators.py

**Kategorie:** Propagation

Beinhaltet die "Arbeiter"-Funktionen für die Relevanz-Propagation auf Layer-Ebene. Jede Funktion kapselt die spezifische mathematische Logik für einen Modul-Typ.

**Hauptfunktionen:**
- `propagate_layer`: Dispatcher für verschiedene Layer-Typen
- `propagate_linear`: LRP durch lineare Layer
- `propagate_layernorm`: LRP durch LayerNorm
- `propagate_multihead_attention`: LRP durch MHA
- `propagate_msdeformattn`: LRP durch Deformable Attention
- `propagate_residual`: LRP durch Residual-Verbindungen

---

### myThesis/lrp/do/lrp_param_utils.py

**Kategorie:** Module

Stellt administrative Werkzeuge für die Vorbereitung und Verwaltung des Modells bereit.

**Hauptfunktionen:**
- `swap_module_inplace`: Ersetzt ein einzelnes Modul durch LRP-Variante
- `swap_all_modules`: Ersetzt alle Module im Modell
- `prepare_model_for_lrp`: Bereitet Modell für LRP vor
- `set_lrp_mode`: Aktiviert/Deaktiviert Aktivierungsspeicherung
- `clear_all_activations`: Löscht gespeicherte Aktivierungen
- `get_lrp_modules`: Gibt Liste aller LRP-Module zurück

---

### myThesis/lrp/do/lrp_param_modules.py

**Kategorie:** Module

Definiert die LRP-fähigen Varianten von Standard-PyTorch-Modulen. Diese Klassen erben von den Originalen, erweitern sie aber um die Fähigkeit, während des Forward-Passes interne Zustände zu speichern.

**Klassen:**
- `LRP_Linear`: LRP-fähiges `nn.Linear`
- `LRP_LayerNorm`: LRP-fähiges `nn.LayerNorm`
- `LRP_MultiheadAttention`: LRP-fähiges `nn.MultiheadAttention`
- `LRP_MSDeformAttn`: LRP-fähiges `MSDeformAttn`

---

### myThesis/lrp/do/lrp_param_base.py

**Kategorie:** Module

Bildet das Fundament für alle LRP-Module.

**Klassen:**
- `LRPActivations`: Strukturierte Datenhaltung für alle Arten von Tensoren (Q, K, V, Weights, etc.), die für LRP benötigt werden
- `LRPModuleMixin`: Injiziert die Speicherlogik und das `is_lrp`-Flag in die Modul-Klassen
- `LRPContext`: Kontext-Klasse für thread-sichere Aktivierungsspeicherung

---

### myThesis/lrp/do/lrp_deform_ops.py

**Kategorie:** Deformable

Implementiert die geometrischen und mathematischen Kernoperationen für die LRP-Berechnung in Deformable Attention. Dazu gehören die Berechnung bilinearer Interpolationsgewichte basierend auf Sampling-Lokationen und das "Splatting" (Scatter-Add) von Relevanzwerten zurück auf das Pixel-Grid. Diese Funktionen sind rein funktional und zustandslos.

---

### myThesis/lrp/do/lrp_deform_capture.py

**Kategorie:** Deformable

Beinhaltet die "Monkey Patching"-Logik, um in `MSDeformAttn`-Implementierungen (z.B. aus Detectron2) einzugreifen. Es registriert Hooks oder patcht Methoden zur Laufzeit, um Zugriff auf interne Sampling-Lokationen und Attention-Gewichte zu erhalten, die normalerweise nicht nach außen sichtbar sind, aber für die LRP-Berechnung unverzichtbar sind.

---

### myThesis/lrp/do/lrp_attn_structs.py

**Kategorie:** Attention

Definiert die `AttnCache`-Datenstruktur als spezialisierten Container für Attention-Layer. Sie speichert alle Zwischenergebnisse des Attention-Mechanismus: Queries, Keys, Values, rohe Scores, normalisierte Gewichte und Projektionsmatrizen. Diese Struktur dient als Schnittstelle zwischen dem Forward-Pass (Capture) und dem Backward-Pass (LRP).

---

### myThesis/lrp/do/lrp_attn_prop.py

**Kategorie:** Attention

Führt die linearen Algebra-Operationen für die Attention-LRP durch. Das Modul implementiert die Rückpropagation durch die drei Hauptpfade der Attention: den Value-Pfad ($O = A \cdot V$), den Query/Key-Pfad ($S = Q \cdot K^T$) und die linearen Projektions-Layer. Es nutzt effiziente Tensor-Operationen (wie `einsum`) für die korrekte Verteilung der Relevanz.

---

### myThesis/lrp/do/lrp_analysis.py

**Kategorie:** High-Level

Bietet High-Level-Schnittstellen für die Durchführung von LRP-Analysen.

**Hauptfunktionen:**
- `run_lrp_batch`: Verarbeitung mehrerer Bilder in einem Batch
- `run_lrp_analysis`: Verarbeitung ganzer Datensätze

**Klassen:**
- `LRPAnalysisContext`: Context Manager für Ressourcenverwaltung (Modell-Vorbereitung, LRP-Modus, Cleanup)
- `BatchLRPProcessor`: Memory-optimierter Batch-Processor mit Generator-basierter Verarbeitung
- `MemoryOptimizedLRP`: Context Manager für einzelne Analysen mit minimalem Speicherverbrauch
- `LRPPerformanceConfig`: Konfigurationsklasse mit Presets für Low-Memory und High-Throughput

**Utility-Funktionen:**
- `estimate_memory_requirements()`: Schätzt Speicherbedarf vor der Analyse

---

## Verwandte Dateien (außerhalb von /do)

### myThesis/lrp/validate_lrp_results.py

**Kategorie:** Validierung

Validierungsskript für LRP-Ergebnis-CSV-Dateien. Prüft ob gespeicherte Relevanzen sinnvoll sind:
- Normalisierung: `sum(|normalized_relevance|) ≈ 1`
- Keine NaN/Inf-Werte
- Sinnvolle Verteilung (nicht alle Werte gleich/null)
- Modul-spezifische Prüfungen (256 Features für Encoder, 300 Queries für Decoder)

**Verwendung:** `python -m myThesis.lrp.validate_lrp_results [--csv PATH] [--dir PATH]`

---

## Architektur-Diagramm

```
┌─────────────────────────────────────────────────────────────────┐
│                         calc_lrp.main()                         │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      lrp_analysis.py                            │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │ LRPAnalysisContext│ │ run_lrp_analysis │  │ run_lrp_batch  │ │
│  └─────────────────┘  └──────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      lrp_controller.py                          │
│                        LRPController                            │
│         (Forward Pass → Aktivierungen → Backward Pass)          │
└─────────────────────────────────────────────────────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│ lrp_propagators  │  │ model_graph_     │  │  param_patcher   │
│                  │  │ wrapper          │  │                  │
│ - propagate_*    │  │ - ModelGraph     │  │ - LRP_Linear     │
│ - dispatch logic │  │ - LayerNode      │  │ - LRP_LayerNorm  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         │
         ├─────────────────┬─────────────────┬─────────────────┐
         ▼                 ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ lrp_rules_      │ │ lrp_rules_      │ │ lrp_rules_      │ │ lrp_softmax     │
│ standard        │ │ attention       │ │ deformable      │ │                 │
│                 │ │                 │ │                 │ │ - gradient_rule │
│ - ε-Regel       │ │ - full_attention│ │ - msdeform_lrp  │ │ - ε-rule        │
│ - γ-Regel       │ │ - value_path    │ │ - bilinear_splat│ │ - jacobian      │
│ - layernorm_lrp │ │ - qk_path       │ │ - capture       │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Mathematische Grundlagen

### LRP-ε-Regel
$$R_j = \sum_k \frac{a_j w_{jk}}{z_k + \epsilon \cdot \text{sign}(z_k)} R_k$$

### LRP-γ-Regel
$$R_j = \sum_k \frac{a_j (w_{jk} + \gamma w_{jk}^+)}{z_k + \gamma z_k^+} R_k$$

### Attention-LRP (Value-Pfad)
$$R_V = A^T \cdot R_O$$

### Softmax-LRP (Gradient-Regel)
$$R_S = A \odot (R_A - \sum_j A_j R_{A_j})$$

---

*Zuletzt aktualisiert: 4. Dezember 2025*
