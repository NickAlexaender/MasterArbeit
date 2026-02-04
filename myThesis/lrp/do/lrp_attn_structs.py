from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from torch import Tensor

# Zwischenwerte für die LRP-Rückpropagation durch Self-Attention und Cross-Attention Layer

@dataclass
class AttnCache:
    # Standard Multi-Head Attention Felder
    attn_weights: Optional[Tensor] = None   # (B,H,T,S) - nach Softmax
    attn_scores: Optional[Tensor] = None    # (B,H,T,S) - vor Softmax
    Q: Optional[Tensor] = None              # (B,H,T,Dh) Query-Projektionen
    K: Optional[Tensor] = None              # (B,H,S,Dh) Key-Projektionen
    Vproj: Optional[Tensor] = None          # (B,H,S,Dh) Value-Projektionen
    W_Q: Optional[Tensor] = None            # (H,Dh,C) Query-Projektionsgewichte
    W_K: Optional[Tensor] = None            # (H,Dh,C) Key-Projektionsgewichte
    W_V: Optional[Tensor] = None            # (H,Dh,C) Value-Projektionsgewichte
    W_O: Optional[Tensor] = None            # (H,Dh,C) Output-Projektionsgewichte
    scale: Optional[float] = None           # Skalierungsfaktor
    
    # FFN-Gewichte (für MLP-Pfad):
    W_FFN1: Optional[Tensor] = None
    W_FFN2: Optional[Tensor] = None
    
    # Deformable Attention (MSDeformAttn) Felder:
    deform_sampling_locations: Optional[Tensor] = None  # (B, T, H, L, P, 2) in [0,1]
    deform_attention_weights: Optional[Tensor] = None   # (B, T, H, L, P)
    deform_spatial_shapes: Optional[Tensor] = None      # (L, 2) -> (H_l, W_l)
    deform_level_start_index: Optional[Tensor] = None   # (L,)
    deform_im2col_step: Optional[int] = None
    W_V_deform: Optional[Tensor] = None     # (H, Dh, C)
    W_O_deform: Optional[Tensor] = None     # (H, Dh, C)
    
    # LayerNorm-Capture (für klassische LRP-Zerlegung bei Post-LN)
    ln_x_in: Optional[Tensor] = None        # (B,T,C) Eingang der letzten LayerNorm
    ln_gamma: Optional[Tensor] = None       # (C,) LayerNorm-Gewicht
    ln_beta: Optional[Tensor] = None        # (C,) LayerNorm-Bias
    # Zurücksetzen
    def clear(self):
        self.attn_weights = None
        self.attn_scores = None
        self.Q = None
        self.K = None
        self.Vproj = None
        self.W_Q = None
        self.W_K = None
        self.W_V = None
        self.W_O = None
        self.scale = None
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
