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
    # Deformable Attention (MSDeformAttn) Felder:
    deform_sampling_locations: Optional[Tensor] = None  # (B, T, H, L, P, 2) in [0,1]
    deform_attention_weights: Optional[Tensor] = None   # (B, T, H, L, P)
    deform_spatial_shapes: Optional[Tensor] = None      # (L, 2) -> (H_l, W_l)
    deform_level_start_index: Optional[Tensor] = None   # (L,)
    deform_im2col_step: Optional[int] = None            # optional metadata
    # Optional: projektionsgewichte aus umgebendem Block
    W_V_deform: Optional[Tensor] = None     # (H, Dh, C)
    W_O_deform: Optional[Tensor] = None     # (H, Dh, C)
    # LayerNorm-Capture (für klassische LRP-Zerlegung bei Post-LN)
    ln_x_in: Optional[Tensor] = None        # (B,T,C) Eingang der letzten LayerNorm im Block
    ln_gamma: Optional[Tensor] = None       # (C,) LayerNorm-Gewicht (weight)
    ln_beta: Optional[Tensor] = None        # (C,) LayerNorm-Bias (bias)

    def clear(self):
        self.attn_weights = None
        self.Vproj = None
        self.W_O = None
        self.W_V = None
        self.W_FFN1 = None
        self.W_FFN2 = None
        self.deform_sampling_locations = None
        self.deform_attention_weights = None
        self.deform_spatial_shapes = None
        self.deform_level_start_index = None
        self.deform_im2col_step = None
        self.W_V_deform = None
        self.W_O_deform = None
        self.ln_x_in = None
        self.ln_gamma = None
        self.ln_beta = None


__all__ = ["AttnCache"]
