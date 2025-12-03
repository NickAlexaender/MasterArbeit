"""
LRP-Regeln für Self-Attention und Cross-Attention in Transformern.

Diese Datei ist der Orchestrator und Import-Punkt für alle Attention-LRP
Funktionalitäten. Sie verbindet die aufgeteilten Module:

    - lrp_attn_structs.py: AttnCache Datenstruktur
    - lrp_softmax.py: Softmax-LRP Regeln (Nicht-Linearität)
    - lrp_attn_prop.py: Lineare Propagation (Q, K, V Pfade)

Mathematischer Hintergrund:
    Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
    
    Sei A = softmax(S) mit S = Q·K^T / √d_k (Attention Scores).
    Die Ausgabe ist: O = A · V

    1. R_V: Rückpropagation durch A·V (lineare Kombination)
    2. R_A: Relevanz der Attention-Gewichte (welche Token wurden beachtet?)
    3. R_Q, R_K: Wie trugen Query und Key zur Attention-Verteilung bei?
"""
from __future__ import annotations

from typing import Tuple, Literal
from torch import Tensor

# Datenstruktur
from myThesis.lrp.do.lrp_attn_structs import AttnCache

# Softmax-LRP (Nicht-Linearität)
from myThesis.lrp.do.lrp_softmax import (
    lrp_softmax,
    lrp_softmax_jacobian,
)

# Lineare Propagation (Q, K, V Pfade, Projektionen)
from myThesis.lrp.do.lrp_attn_prop import (
    lrp_attention_value_path,
    lrp_attention_value_path_conservative,
    lrp_attention_to_weights,
    lrp_attention_qk_path,
    lrp_attention_qk_path_symmetric,
    lrp_projection_layer,
    lrp_multihead_output_projection,
    compute_encoder_token_relevance,
)


# =============================================================================
# Vollständige Attention-LRP Pipeline
# =============================================================================


def lrp_full_attention(
    R_out: Tensor,
    cache: AttnCache,
    softmax_rule: Literal["gradient", "epsilon", "jacobian"] = "gradient",
    qk_split: Literal["asymmetric", "symmetric"] = "asymmetric",
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Vollständige LRP durch einen Attention-Block.
    
    Propagiert Relevanz von der Attention-Ausgabe durch alle Komponenten:
    O -> V -> A -> S -> (Q, K)
    
    Args:
        R_out: Relevanz der Attention-Ausgabe (B, H, T, Dh)
        cache: AttnCache mit gespeicherten Zwischenwerten
        softmax_rule: Regel für Softmax-LRP ("gradient", "epsilon", "jacobian")
        qk_split: Verteilungsstrategie für Q/K ("asymmetric", "symmetric")
        eps: Stabilisierungsterm
        
    Returns:
        (R_Q, R_K, R_V, R_A): Relevanz für alle Attention-Komponenten
        
    Raises:
        ValueError: Falls notwendige Werte im Cache fehlen
    """
    # Validierung
    if cache.attn_weights is None or cache.Vproj is None:
        raise ValueError("AttnCache muss attn_weights und Vproj enthalten")
    if cache.Q is None or cache.K is None:
        raise ValueError("AttnCache muss Q und K für Q/K-Pfad enthalten")
    if softmax_rule != "epsilon" and cache.attn_scores is None:
        raise ValueError("AttnCache muss attn_scores für Softmax-LRP enthalten")
    
    attn_weights = cache.attn_weights
    V = cache.Vproj
    Q = cache.Q
    K = cache.K
    attn_scores = cache.attn_scores
    scale = cache.scale
    
    # 1. R_O -> R_V (Value-Pfad)
    R_V = lrp_attention_value_path(R_out, attn_weights, V, eps)
    
    # 2. R_O -> R_A (Attention-Gewichte)
    R_A = lrp_attention_to_weights(R_out, attn_weights, V, eps)
    
    # 3. R_A -> R_S (durch Softmax)
    if softmax_rule == "jacobian":
        R_S = lrp_softmax_jacobian(R_A, attn_weights, attn_scores, eps)
    else:
        R_S = lrp_softmax(R_A, attn_weights, attn_scores, rule=softmax_rule, eps=eps)
    
    # 4. R_S -> R_Q, R_K (Query/Key-Pfad)
    if qk_split == "symmetric":
        R_Q, R_K = lrp_attention_qk_path_symmetric(R_S, Q, K, scale, eps)
    else:
        R_Q, R_K = lrp_attention_qk_path(R_S, Q, K, scale, eps)
    
    return R_Q, R_K, R_V, R_A


# =============================================================================
# Cross-Attention spezifische Funktionen
# =============================================================================


def lrp_cross_attention(
    R_out: Tensor,
    cache: AttnCache,
    softmax_rule: Literal["gradient", "epsilon", "jacobian"] = "gradient",
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor, Tensor]:
    """LRP für Cross-Attention (Decoder Query, Encoder Key/Value).
    
    In Cross-Attention stammen Q aus dem Decoder und K, V aus dem Encoder.
    Diese Funktion berechnet die Relevanz für beide Seiten.
    
    Args:
        R_out: Relevanz der Attention-Ausgabe (B, H, T_dec, Dh)
        cache: AttnCache mit Q (Decoder), K, V (Encoder)
        softmax_rule: Regel für Softmax-LRP
        eps: Stabilisierungsterm
        
    Returns:
        (R_Q, R_encoder, R_A): 
            - R_Q: Query-Relevanz (Decoder) (B, H, T_dec, Dh)
            - R_encoder: Encoder-Relevanz (kombiniert K und V) (B, H, S_enc, Dh)
            - R_A: Attention-Gewichte-Relevanz (B, H, T_dec, S_enc)
            
    Note:
        R_encoder kombiniert R_K und R_V, da beide auf die Encoder-Tokens
        zurückgeführt werden.
    """
    R_Q, R_K, R_V, R_A = lrp_full_attention(
        R_out, cache, softmax_rule, qk_split="asymmetric", eps=eps
    )
    
    # Kombiniere Encoder-Relevanz (K und V stammen aus demselben Encoder-Output)
    # Beide tragen zur finalen Relevanz der Encoder-Tokens bei
    R_encoder = R_K + R_V
    
    return R_Q, R_encoder, R_A


__all__ = [
    # Datenstruktur
    "AttnCache",
    # Softmax-LRP
    "lrp_softmax",
    "lrp_softmax_jacobian",
    # Lineare Propagation
    "lrp_attention_value_path",
    "lrp_attention_value_path_conservative",
    "lrp_attention_to_weights",
    "lrp_attention_qk_path",
    "lrp_attention_qk_path_symmetric",
    "lrp_projection_layer",
    "lrp_multihead_output_projection",
    "compute_encoder_token_relevance",
    # Orchestrator-Funktionen
    "lrp_full_attention",
    "lrp_cross_attention",
]
