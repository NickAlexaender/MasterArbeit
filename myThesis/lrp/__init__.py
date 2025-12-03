"""
LRP-Analyse-Paket für MaskDINO.

Dieses Paket enthält die vollständige Layer-wise Relevance Propagation (LRP)
Implementierung für MaskDINO und ähnliche Detectron2-basierte Modelle.

Hauptmodule:
    - do: Kernfunktionalität (LRPController, Regeln, Graph-Wrapper)
    - lrp: Zusätzliche LRP-Utilities (falls vorhanden)
    - calc: Berechnungs-Utilities (falls vorhanden)
"""
from __future__ import annotations

# WICHTIG: Kompatibilitäts-Patches müssen zuerst geladen werden
from .do.compat import *  # Pillow/NumPy monkey patches - MUSS vor anderen Imports kommen

# Re-export der wichtigsten Komponenten aus do/
from .do import (
    LRPController,
    LRPResult,
    LRPAnalysisContext,
    run_lrp_analysis,
    ModelGraph,
    LayerNode,
    LayerType,
    prepare_model_for_lrp,
    set_lrp_mode,
)

__all__ = [
    "LRPController",
    "LRPResult",
    "LRPAnalysisContext",
    "run_lrp_analysis",
    "ModelGraph",
    "LayerNode",
    "LayerType",
    "prepare_model_for_lrp",
    "set_lrp_mode",
]
