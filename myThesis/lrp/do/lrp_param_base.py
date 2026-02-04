from __future__ import annotations
from typing import Dict, Optional
import torch
from torch import Tensor


# LRP benötigt einige Aktivierungen, die wir hier speichern. Interne, modulgebundene Speicherung der Zwischenwerte.

class LRPActivations:
    __slots__ = (
        'input', 'output', 'weights', 'bias',
        'Q', 'K', 'V', 'attn_weights', 'attn_scores',
        'W_Q', 'W_K', 'W_V', 'W_O', 'scale',
        'sampling_locations', 'deform_attention_weights',
        'spatial_shapes', 'level_start_index', 'value_proj',
        'gamma', 'beta', 'mean', 'var',
        'reference_points', 'input_flatten', 'query',
    )
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        """Setzt alle Aktivierungen zurück."""
        for attr in self.__slots__:
            setattr(self, attr, None)
    
    def to_dict(self) -> Dict[str, Optional[Tensor]]:
        """Exportiert alle nicht-None Aktivierungen als Dictionary."""
        return {k: getattr(self, k) for k in self.__slots__ if getattr(self, k) is not None}


# Mixin für LRP-fähige Module. Kann zu beliebigen Modulen hinzugefügt werden.

class LRPModuleMixin:
    _is_lrp: bool = False
    _activations: Optional[LRPActivations] = None
    
    @property
    def is_lrp(self) -> bool:
        return self._is_lrp
    
    @is_lrp.setter
    def is_lrp(self, value: bool):
        self._is_lrp = value
        if value and self._activations is None:
            self._activations = LRPActivations()
        elif not value and self._activations is not None:
            self._activations.clear()
    
    @property
    def activations(self) -> LRPActivations:
        if self._activations is None:
            self._activations = LRPActivations()
        return self._activations
    
    def clear_activations(self):
        if self._activations is not None:
            self._activations.clear()


__all__ = [
    "LRPActivations",
    "LRPModuleMixin",
]
