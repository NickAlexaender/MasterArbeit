"""
Hilfsfunktionen und Puffer für Attention-Value-Pfad-Daten.

Diese Datei stellt schlanke Strukturen bereit, um während des Forwards
Attention-Zwischenwerte (z. B. Gewichte und Value-Projektionen) zwischenzuspeichern.
Die eigentliche Modellhook-Registrierung erfolgt in hooks_maskdino.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
from torch import Tensor


@dataclass
class AttnCache:
    attn_weights: Optional[Tensor] = None   # (B,H,T,S)
    Vproj: Optional[Tensor] = None          # (B,H,S,Dh)
    W_O: Optional[Tensor] = None            # (H,Dh,C)
    W_V: Optional[Tensor] = None            # (H,Dh,C) value-projection weights per head/dim
    # FFN-Gewichte (für MLP-Pfad):
    W_FFN1: Optional[Tensor] = None         # (Dhid, C_in)
    W_FFN2: Optional[Tensor] = None         # (C_out, Dhid)

    def clear(self):
        self.attn_weights = None
        self.Vproj = None
        self.W_O = None
        self.W_V = None
        self.W_FFN1 = None
        self.W_FFN2 = None


__all__ = ["AttnCache"]
