from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from torch import Tensor


@dataclass
class LRPResult:
    R_input: Optional[Tensor] = None
    R_per_layer: Dict[str, Tensor] = field(default_factory=dict)
    conservation_errors: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LayerRelevance:
    R_out: Tensor
    R_in: Optional[Tensor] = None
    R_skip: Optional[Tensor] = None
    R_transform: Optional[Tensor] = None



__all__ = [
    "LRPResult",
    "LayerRelevance",
]
