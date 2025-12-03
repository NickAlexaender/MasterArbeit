# Look into one of the files below, if you think the code could be relevant.

## myThesis/lrp/do/lrp_controller.py

Das zentrale Gehirn der LRP-Analyse für MaskDINO. Die Klasse `LRPController` vereint die zuvor getrennte Logik aus "Analyse" und "Engine" und löst State-Management-Probleme durch eine einheitliche Kontextverwaltung. Der **Forward Pass** führt das Modell im Evaluierungsmodus aus und speichert Aktivierungen ($a_j$) lokal in den LRP-Modul-Instanzen. Der **Backward Pass** iteriert durch den Modell-Graphen in umgekehrter topologischer Reihenfolge, ruft die entsprechenden Propagatoren auf und übergibt den Relevanz-Tensor an den nächsten Layer. Die **Aggregation** verwaltet die Summation von Relevanzwerten aus Skip-Verbindungen und stellt die Konservierungseigenschaft sicher.

## myThesis/lrp/do/tensor_ops.py

Dieses Modul stellt zustandslose Hilfsfunktionen für numerische Stabilität und Tensor-Transformation bereit. `safe_divide` verhindert Division-durch-Null-Fehler, was für LRP-Regeln essentiell ist. `rearrange_activations` konvertiert Aktivierungen zwischen CNN-Format (N,C,H,W) und Transformer-Format (N,L,C). `compute_jacobian` und `gradient_times_input` implementieren robuste Methoden für instabile Layer. Zudem gibt es Utilities wie `build_target_relevance` zur Erzeugung von Start-Relevanzen.

## myThesis/lrp/do/param_patcher.py

Ein Kompatibilitäts-Wrapper und zentraler Einstiegspunkt für das Patchen des Modells. Er re-exportiert alle LRP-fähigen Module (`LRP_Linear`, etc.) und Utilities (`swap_module_inplace`, `prepare_model_for_lrp`) aus den aufgeteilten Untermodulen. Dies ermöglicht das automatische In-Place-Ersetzen von Standard-Modulen durch LRP-fähige Varianten, ohne die Architekturdefinition ändern zu müssen.

## myThesis/lrp/do/model_graph_wrapper.py

Linearisiert die komplexe verschachtelte Struktur von MaskDINO/Detectron2-Modellen in einen flachen `ModelGraph`. Dies ermöglicht eine geordnete Iteration (insbesondere rückwärts für LRP) und das korrekte Routing von Relevanz zwischen Decodern, Encodern und dem Backbone. Der Graph erkennt Layer-Typen und Abhängigkeiten, um den Relevanzfluss auch über Skip-Connections und Cross-Attention hinweg korrekt zu steuern.

## myThesis/lrp/do/lrp_structs.py

Definiert die zentralen Datenstrukturen für die LRP-Analyse als reine Dataclasses ohne Geschäftslogik. `LRPResult` kapselt das Gesamtergebnis einer Analyse, inklusive Eingangsrelevanz (Heatmap), Layer-Relevanzen und Metadaten. `LayerRelevance` dient als Container für die Relevanzwerte eines einzelnen Layers und unterscheidet zwischen Eingangs-, Ausgangs-, Skip- und Transformations-Relevanz.

## myThesis/lrp/do/lrp_softmax.py

Implementiert die mathematisch anspruchsvollen LRP-Regeln für Softmax-Layer ($A = \text{softmax}(S)$). Bietet verschiedene Rückpropagations-Strategien: Die Gradient-Regel (Taylor-Expansion 1. Ordnung), die $\epsilon$-Regel für stabilisierte Beiträge und eine exakte Jacobian-basierte Berechnung. Diese Funktionen ermöglichen den Fluss von Relevanz von den normalisierten Attention-Gewichten zurück zu den rohen Scores.

## myThesis/lrp/do/lrp_rules_standard.py

Enthält die Standard-LRP-Regeln wie die $\epsilon$-Regel und $\gamma$-Regel zur Rückpropagation durch lineare Schichten. Zudem implementiert es spezifische Logik für LayerNorm (unter Berücksichtigung von Normalisierungsstatistiken) und Strategien für Residual-Splits (Aufteilung der Relevanz auf Skip-Verbindungen), um numerische Stabilität und Konservierung zu gewährleisten.

## myThesis/lrp/do/lrp_rules_deformable.py

Der Einstiegspunkt für LRP in Deformable Attention Layern (`MSDeformAttn`). Implementiert die High-Level-Logik zur Verteilung der Relevanz auf Sampling-Lokationen und Value-Features. Da MSDeformAttn Werte mittels bilinearer Interpolation sampelt, muss die Relevanz entsprechend auf die benachbarten Pixel verteilt werden ("Splatting"). Unterstützt auch den Value-Pfad für eine vollständigere Attribution.

## myThesis/lrp/do/lrp_rules_attention.py

Orchestriert die LRP-Pipeline für Standard Multi-Head Attention (Self- und Cross-Attention). Zerlegt die Formel $\text{Attention}(Q,K,V) = \text{softmax}(QK^T/\sqrt{d_k})V$ in separate Pfade. Verbindet die Softmax-LRP-Regeln mit der linearen Propagation durch die Q-, K- und V-Pfade und steuert den Fluss durch die Projektionsmatrizen.

## myThesis/lrp/do/lrp_propagators.py

Beinhaltet die "Arbeiter"-Funktionen für die Relevanz-Propagation auf Layer-Ebene. Jede Funktion (z.B. `propagate_linear`, `propagate_multihead_attention`) kapselt die spezifische mathematische Logik für einen Modul-Typ. Sie nehmen die gespeicherten Aktivierungen und die eingehende Relevanz entgegen und berechnen die Relevanz für die Eingänge, getrennt von der Iterationslogik des Controllers.

## myThesis/lrp/do/lrp_param_utils.py

Stellt administrative Werkzeuge für die Vorbereitung und Verwaltung des Modells bereit. Funktionen wie `swap_module_inplace` und `swap_all_modules` ermöglichen den Austausch von Standard-PyTorch-Modulen gegen ihre LRP-fähigen Pendants. Zudem steuert das Modul über `set_lrp_mode` global, ob Aktivierungen gespeichert werden sollen, und bietet Mechanismen zur Speicherbereinigung.

## myThesis/lrp/do/lrp_param_modules.py

Definiert die LRP-fähigen Varianten von Standard-PyTorch-Modulen wie `LRP_Linear`, `LRP_LayerNorm`, `LRP_MultiheadAttention` und `LRP_MSDeformAttn`. Diese Klassen erben von den Originalen, erweitern sie aber um die Fähigkeit, während des Forward-Passes interne Zustände (Inputs, Outputs, Attention-Weights, Sampling-Locations) in einem `LRPActivations`-Objekt zu speichern.

## myThesis/lrp/do/lrp_param_base.py

Bildet das Fundament für alle LRP-Module. Die Klasse `LRPActivations` definiert eine strukturierte Datenhaltung für alle Arten von Tensoren (Q, K, V, Weights, etc.), die für LRP benötigt werden. Das Mixin `LRPModuleMixin` injiziert die Speicherlogik und das `is_lrp`-Flag in die Modul-Klassen und sorgt für eine einheitliche Schnittstelle.

## myThesis/lrp/do/lrp_deform_ops.py

Implementiert die geometrischen und mathematischen Kernoperationen für die LRP-Berechnung in Deformable Attention. Dazu gehören die Berechnung bilinearer Interpolationsgewichte basierend auf Sampling-Lokationen und das "Splatting" (Scatter-Add) von Relevanzwerten zurück auf das Pixel-Grid. Diese Funktionen sind rein funktional und zustandslos.

## myThesis/lrp/do/lrp_deform_capture.py

Beinhaltet die "Monkey Patching"-Logik, um in `MSDeformAttn`-Implementierungen (z.B. aus Detectron2) einzugreifen. Es registriert Hooks oder patcht Methoden zur Laufzeit, um Zugriff auf interne Sampling-Lokationen und Attention-Gewichte zu erhalten, die normalerweise nicht nach außen sichtbar sind, aber für die LRP-Berechnung unverzichtbar sind.

## myThesis/lrp/do/lrp_attn_structs.py

Definiert die `AttnCache`-Datenstruktur als spezialisierten Container für Attention-Layer. Sie speichert alle Zwischenergebnisse des Attention-Mechanismus: Queries, Keys, Values, rohe Scores, normalisierte Gewichte und Projektionsmatrizen. Diese Struktur dient als Schnittstelle zwischen dem Forward-Pass (Capture) und dem Backward-Pass (LRP).

## myThesis/lrp/do/lrp_attn_prop.py

Führt die linearen Algebra-Operationen für die Attention-LRP durch. Das Modul implementiert die Rückpropagation durch die drei Hauptpfade der Attention: den Value-Pfad ($O = A \cdot V$), den Query/Key-Pfad ($S = Q \cdot K^T$) und die linearen Projektions-Layer. Es nutzt effiziente Tensor-Operationen (wie `einsum`) für die korrekte Verteilung der Relevanz.

## myThesis/lrp/do/lrp_analysis.py

Bietet High-Level-Schnittstellen für die Durchführung von LRP-Analysen. Funktionen wie `run_lrp_batch` ermöglichen die Verarbeitung mehrerer Bilder, während `run_lrp_analysis` ganze Datensätze verarbeitet. Der `LRPAnalysisContext` vereinfacht als Context Manager die Handhabung von Ressourcen (Modell-Vorbereitung, LRP-Modus, Cleanup).

## myThesis/lrp/do/io_utils.py

Eine kleine Sammlung von I/O-Hilfsfunktionen. Primär zuständig für das Auffinden und Auflisten von Bilddateien in Verzeichnissen (`collect_images`), um Batch-Verarbeitungsprozesse zu unterstützen.
