"""
LRP Datenstrukturen - Reine Datencontainer für LRP-Analyse.

Dieses Modul enthält nur Datenklassen (dataclasses) ohne Logik.
Es definiert, wie Input und Output der LRP-Analyse strukturiert sind.

Verwendung:
    >>> from lrp_structs import LRPResult, LayerRelevance
    >>> result = LRPResult()
    >>> result.R_input = relevance_tensor
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from torch import Tensor


# =============================================================================
# Datenstrukturen
# =============================================================================


@dataclass
class LRPResult:
    """Ergebnis einer LRP-Analyse.
    
    Attributes:
        R_input: Relevanz auf der Eingabe (Pixel-Level) (B, C, H, W)
        R_per_layer: Dict mit Relevanz pro Layer-Name
        conservation_errors: Liste von Konservierungsfehlern pro Layer
        metadata: Zusätzliche Metadaten der Analyse
    """
    R_input: Optional[Tensor] = None
    R_per_layer: Dict[str, Tensor] = field(default_factory=dict)
    conservation_errors: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerRelevance:
    """Relevanz-Container für einen einzelnen Layer.
    
    Attributes:
        R_out: Ausgabe-Relevanz (eingehend für Rückpropagation)
        R_in: Eingabe-Relevanz (Ergebnis der Rückpropagation)
        R_skip: Relevanz des Skip-Pfads (bei Residuals)
        R_transform: Relevanz des Transform-Pfads
    """
    R_out: Tensor
    R_in: Optional[Tensor] = None
    R_skip: Optional[Tensor] = None
    R_transform: Optional[Tensor] = None


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "LRPResult",
    "LayerRelevance",
]
