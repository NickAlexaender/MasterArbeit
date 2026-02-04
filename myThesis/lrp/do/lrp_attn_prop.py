from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor
from myThesis.lrp.do.tensor_ops import safe_divide


# LRP für Value-Pfad: R_O -> R_V
# Hier wollen wir die Relevanz von der Attention-Ausgabe zurück zu den Values propagieren.
# O = A · V -> proportional zum Beitrag jedes Value-Vektors verteilt

def lrp_attention_value_path(
    R_out: Tensor,
    attn_weights: Tensor,
    V: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    B, H, T, S = attn_weights.shape
    _, _, _, Dh = V.shape
    
    # O = A @ V -> (B, H, T, Dh)
    # O_t = sum_s A_ts * V_s
    O = torch.einsum("bhts,bhsd->bhtd", attn_weights, V)
    
    # Relevanz-Propagation: R_V[s] = sum_t A_ts * V_s / O_t * R_O[t]
    # Äquivalent: R_V = A^T @ (R_O / O) * V (broadcast über Dh)
    s = safe_divide(R_out, O, eps)  # (B, H, T, Dh)
    
    # R_V[s,d] = sum_t A[t,s] * V[s,d] * s[t,d]
    # = V[s,d] * sum_t A[t,s] * s[t,d]
    weighted_relevance = torch.einsum("bhts,bhtd->bhsd", attn_weights, s)
    R_V = V * weighted_relevance
    
    return R_V

# Diese Variante normalisiert die Relevanz, sodass die Summe erhalten bleibt.

def lrp_attention_value_path_conservative(
    R_out: Tensor,
    attn_weights: Tensor,
    V: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    R_V = lrp_attention_value_path(R_out, attn_weights, V, eps)
    
    # Konservative Normalisierung
    R_sum_out = R_out.sum()
    R_sum_V = R_V.sum()
    scale = safe_divide(R_sum_out, R_sum_V, eps)
    
    return R_V * scale


# Wir können die Attention-Gewichte aus der Output-Relevanz berechnen
# O_td = Σ_s A_ts · V_sd
# R_A[t,s] = Σ_d (A_ts · V_sd) / O_td · R_O[t,d]

def lrp_attention_to_weights(
    R_out: Tensor,
    attn_weights: Tensor,
    V: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    B, H, T, S = attn_weights.shape
    _, _, _, Dh = V.shape
    
    # O = A @ V -> (B, H, T, Dh)
    O = torch.einsum("bhts,bhsd->bhtd", attn_weights, V)
    
    # Relevanz pro (t,s)-Paar: aggregiere über Dh
    # R_A[t,s] = sum_d A[t,s] * V[s,d] / O[t,d] * R_O[t,d]
    # = A[t,s] * sum_d V[s,d] * R_O[t,d] / O[t,d]
    s = safe_divide(R_out, O, eps)  # (B, H, T, Dh)
    
    # Gewichtete Summe über Dh
    relevance_per_source = torch.einsum("bhtd,bhsd->bhts", s, V)
    R_A = attn_weights * relevance_per_source
    
    return R_A


# Propagiert Relevanz von Scores S zu Queries Q und Keys K
# S = Q · K^T / √d_k

def lrp_attention_qk_path(
    R_S: Tensor,
    Q: Tensor,
    K: Tensor,
    scale: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    B, H, T, Dh = Q.shape
    _, _, S_dim, _ = K.shape
    
    if scale is None:
        scale = 1.0 / (Dh ** 0.5)
    
    # S = Q @ K^T * scale -> (B, H, T, S)
    S = torch.einsum("bhtd,bhsd->bhts", Q, K) * scale
    
    # Relevanz-Signal pro (t,s)-Paar
    relevance_signal = safe_divide(R_S, S, eps)  # (B, H, T, S)
    
    # R_Q[t,d] = Q[t,d] * scale * Σ_s K[s,d] * relevance_signal[t,s]
    K_weighted = torch.einsum("bhts,bhsd->bhtd", relevance_signal, K)
    R_Q = Q * K_weighted * scale
    
    # R_K[s,d] = K[s,d] * scale * Σ_t Q[t,d] * relevance_signal[t,s]
    Q_weighted = torch.einsum("bhts,bhtd->bhsd", relevance_signal, Q)
    R_K = K * Q_weighted * scale
    
    return R_Q, R_K

# Symmetrische Q/K-Relevanz-Verteilung (50/50 Split)
#  Diese Variante verteilt die Relevanz gleichmäßig zwischen Q und K, was für Self-Attention oft sinnvoller ist

def lrp_attention_qk_path_symmetric(
    R_S: Tensor,
    Q: Tensor,
    K: Tensor,
    scale: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    R_Q, R_K = lrp_attention_qk_path(R_S, Q, K, scale, eps)
    
    # Berechne Gesamt-Relevanz
    R_total = R_S.sum()
    R_Q_sum = R_Q.sum()
    R_K_sum = R_K.sum()
    
    # Normalisiere auf 50/50 Split
    if R_Q_sum.abs() > eps and R_K_sum.abs() > eps:
        target = R_total / 2
        R_Q = R_Q * (target / R_Q_sum)
        R_K = R_K * (target / R_K_sum)
    
    return R_Q, R_K

# Propagiert Relevanz durch eine lineare Projektion: y = x @ W^T

def lrp_projection_layer(
    R_proj: Tensor,
    x_in: Tensor,
    W: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    # z = x @ W^T
    z = torch.einsum("...i,oi->...o", x_in, W)
    
    # ε-Regel
    s = safe_divide(R_proj, z, eps)
    c = torch.einsum("...o,oi->...i", s, W)
    
    return x_in * c

# Propagiert Relevanz durch die Output-Projektion, die alle Heads kombiniert

def lrp_multihead_output_projection(
    R_out: Tensor,
    context: Tensor,
    W_O: Tensor,
    num_heads: int,
    eps: float = 1e-6,
) -> Tensor:
    # Falls W_O in Head-Format (H, Dh, C), reshape zu (C, H*Dh)
    if W_O.dim() == 3:
        H, Dh, C = W_O.shape
        W_O = W_O.permute(2, 0, 1).reshape(C, H * Dh)
    
    return lrp_projection_layer(R_out, context, W_O.T, eps)


# Aggregieren der Encoder-Relevanz zu Token-Level Scores

def compute_encoder_token_relevance(
    R_encoder: Tensor,
    spatial_shape: Optional[Tuple[int, int]] = None,
) -> Tensor:
    if R_encoder.dim() == 4:
        # (B, H, S, Dh) -> (B, S)
        R_tokens = R_encoder.sum(dim=(1, 3))  # Summiere über Heads und Dh
    else:
        # (B, S, C) -> (B, S)
        R_tokens = R_encoder.sum(dim=-1)
    
    if spatial_shape is not None:
        H, W = spatial_shape
        B, S = R_tokens.shape
        if S == H * W:
            R_tokens = R_tokens.view(B, H, W)
    
    return R_tokens


__all__ = [
    "lrp_attention_value_path",
    "lrp_attention_value_path_conservative",
    "lrp_attention_to_weights",
    "lrp_attention_qk_path",
    "lrp_attention_qk_path_symmetric",
    "lrp_projection_layer",
    "lrp_multihead_output_projection",
    "compute_encoder_token_relevance",
]
