"""
LRP-Basis-Klassen: Aktivierungsspeicher und Modul-Mixin.

Dieses Modul enthält die primitiven Bausteine für LRP-fähige Module:
    - LRPActivations: Datenstruktur zum Speichern aller LRP-relevanten Aktivierungen
    - LRPModuleMixin: Mixin-Klasse, die is_lrp-Flag und activations-Speicher hinzufügt

Diese Klassen haben keine Abhängigkeiten zu anderen LRP-Teilen und bilden
die Grundlage für alle LRP-fähigen Module.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor


# =============================================================================
# Aktivierungsspeicher-Datenstruktur
# =============================================================================


class LRPActivations:
    """Speichert alle für LRP benötigten Aktivierungen eines Layers.
    
    Diese Klasse ersetzt das externe AttnCache durch eine interne,
    modulgebundene Speicherung der Zwischenwerte.
    
    Attributes:
        input: Eingabe-Tensor x (B, T, C) oder (B, C, H, W)
        output: Ausgabe-Tensor y
        weights: Gewichtsmatrix W für lineare Operationen
        bias: Bias-Vektor b
        
        # Attention-spezifisch:
        Q: Query-Projektionen (B, H, T, Dh)
        K: Key-Projektionen (B, H, S, Dh)
        V: Value-Projektionen (B, H, S, Dh)
        attn_weights: Attention-Gewichte nach Softmax (B, H, T, S)
        attn_scores: Rohe Scores vor Softmax (B, H, T, S)
        W_Q, W_K, W_V, W_O: Projektionsgewichte
        
        # MSDeformAttn-spezifisch:
        sampling_locations: (B, T, H, L, P, 2) normalisierte Sampling-Positionen
        deform_attention_weights: (B, T, H, L, P) Attention-Gewichte
        spatial_shapes: (L, 2) räumliche Dimensionen pro Level
        level_start_index: (L,) Start-Indizes pro Level
        value_proj: Projizierte Values
        
        # LayerNorm-spezifisch:
        gamma: LayerNorm-Gewicht (C,)
        beta: LayerNorm-Bias (C,)
        mean: Mittelwert für Normalisierung
        var: Varianz für Normalisierung
    """
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


# =============================================================================
# LRP-Basis-Mixin
# =============================================================================


class LRPModuleMixin:
    """Mixin für LRP-fähige Module.
    
    Fügt is_lrp-Flag und activations-Speicher zu beliebigen Modulen hinzu.
    
    Usage:
        >>> class MyLRPModule(nn.Module, LRPModuleMixin):
        ...     def forward(self, x):
        ...         if self._is_lrp:
        ...             self.activations.input = x.detach()
        ...         return x
    """
    _is_lrp: bool = False
    _activations: Optional[LRPActivations] = None
    
    @property
    def is_lrp(self) -> bool:
        """Gibt zurück, ob LRP-Modus aktiv ist."""
        return self._is_lrp
    
    @is_lrp.setter
    def is_lrp(self, value: bool):
        """Aktiviert/deaktiviert LRP-Modus."""
        self._is_lrp = value
        if value and self._activations is None:
            self._activations = LRPActivations()
        elif not value and self._activations is not None:
            self._activations.clear()
    
    @property
    def activations(self) -> LRPActivations:
        """Gibt die Aktivierungen zurück (lazy initialization)."""
        if self._activations is None:
            self._activations = LRPActivations()
        return self._activations
    
    def clear_activations(self):
        """Löscht alle gespeicherten Aktivierungen."""
        if self._activations is not None:
            self._activations.clear()


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    "LRPActivations",
    "LRPModuleMixin",
]
