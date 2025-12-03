"""
LRP-Regeln für Softmax-Layer.

Diese Datei implementiert die mathematisch anspruchsvollste Komponente der
Attention-LRP: die Rückpropagation durch die Softmax-Nichtlinearität.

Mathematischer Hintergrund:
    Softmax: A = softmax(S) mit A_ts = exp(S_ts) / Σ_s' exp(S_ts')
    
    Die Jacobi-Matrix des Softmax ist:
        ∂A_ts/∂S_tk = A_ts · (δ_sk - A_tk)
    
    Dies führt zu spezifischen LRP-Regeln für den Softmax-Layer.

Implementierte Regeln:
    - "gradient": Gradient × Input (Taylor-Expansion 1. Ordnung)
    - "epsilon": ε-Regel mit Score-Beiträgen
    - "weighted": Gewichtete Attribution basierend auf Score-Magnitude
    - "jacobian": Exakte Berechnung via vollständige Jacobi-Matrix
"""
from __future__ import annotations

from typing import Literal
import torch
from torch import Tensor

from myThesis.lrp.do.tensor_ops import safe_divide


def lrp_softmax(
    R_A: Tensor,
    attn_weights: Tensor,
    attn_scores: Tensor,
    rule: Literal["gradient", "epsilon", "weighted"] = "gradient",
    eps: float = 1e-6,
) -> Tensor:
    """LRP-Regel für Softmax: Propagiert Relevanz von A zu Scores S.
    
    Softmax: A = softmax(S) mit A_ts = exp(S_ts) / Σ_s' exp(S_ts')
    
    Die Jacobi-Matrix des Softmax ist:
        ∂A_ts/∂S_tk = A_ts · (δ_sk - A_tk)
    
    Dies ergibt spezielle LRP-Regeln für die Softmax-Schicht.
    
    Implementierte Regeln:
        - "gradient": Gradient × Input (Taylor-Expansion 1. Ordnung)
        - "epsilon": ε-Regel mit Score-Beiträgen
        - "weighted": Gewichtete Attribution basierend auf Score-Magnitude
    
    Args:
        R_A: Relevanz der Attention-Gewichte (B, H, T, S)
        attn_weights: Attention-Gewichte A = softmax(S) (B, H, T, S)
        attn_scores: Rohe Attention-Scores S = QK^T/√d (B, H, T, S)
        rule: LRP-Regel für Softmax
        eps: Stabilisierungsterm
        
    Returns:
        R_S: Relevanz der Attention-Scores (B, H, T, S)
        
    Mathematik (Gradient-Regel):
        Die Softmax-Jacobi-Matrix für eine Zeile ist:
        J_jk = A_j(δ_jk - A_k) = A_j·δ_jk - A_j·A_k
        
        Für LRP berechnen wir:
        R_S[k] = S[k] · Σ_j J_jk · (R_A[j] / A[j])
               = S[k] · Σ_j (A_j·δ_jk - A_j·A_k) · (R_A[j] / A[j])
               = S[k] · (R_A[k] - A_k · Σ_j R_A[j])
    """
    B, H, T, S_dim = R_A.shape
    
    if rule == "gradient":
        # Gradient × Input Methode
        # ∂A/∂S = diag(A) - A ⊗ A (Outer Product)
        # Für jede Zeile t: grad_t = R_A_t - A_t · sum(R_A_t)
        
        R_A_sum = R_A.sum(dim=-1, keepdim=True)  # (B, H, T, 1)
        grad = R_A - attn_weights * R_A_sum  # (B, H, T, S)
        
        # Gradient × Input
        R_S = attn_scores * grad
        
    elif rule == "epsilon":
        # ε-Regel: Behandle Softmax als linearen Layer mit Beitrag exp(S)
        # z_k = exp(S_k), A_k = z_k / Σ z
        # R_S[k] = exp(S[k]) / (Σ exp(S) + ε) · R_A[k] (vereinfacht)
        
        # Berechne exp(S) mit numerischer Stabilität
        S_max = attn_scores.max(dim=-1, keepdim=True).values
        exp_S = torch.exp(attn_scores - S_max)
        sum_exp_S = exp_S.sum(dim=-1, keepdim=True)
        
        # ε-Regel
        contribution = safe_divide(exp_S, sum_exp_S, eps)
        R_S = contribution * R_A
        
        # Relevanz-Erhaltung
        scale = safe_divide(
            R_A.sum(dim=-1, keepdim=True),
            R_S.sum(dim=-1, keepdim=True),
            eps
        )
        R_S = R_S * scale
        
    elif rule == "weighted":
        # Gewichtete Attribution basierend auf Score-Magnitude
        # Höhere Scores (die zu höheren Attention-Gewichten führen)
        # erhalten proportional mehr Relevanz
        
        # Positive Score-Beiträge (Softmax ist monoton in S)
        S_pos = attn_scores.clamp(min=0)
        S_sum = S_pos.sum(dim=-1, keepdim=True) + eps
        
        # Proportionale Verteilung
        R_S = (S_pos / S_sum) * R_A.sum(dim=-1, keepdim=True)
        
        # Mische mit direkter Attribution
        R_S = 0.5 * R_S + 0.5 * attn_weights * R_A
        
    else:
        raise ValueError(f"Unbekannte Softmax-LRP-Regel: {rule}")
    
    return R_S


def lrp_softmax_jacobian(
    R_A: Tensor,
    attn_weights: Tensor,
    attn_scores: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Exakte Softmax-LRP via vollständige Jacobi-Matrix-Berechnung.
    
    Berechnet die exakte Relevanz-Rückpropagation durch Softmax unter
    Verwendung der analytischen Jacobi-Matrix.
    
    Args:
        R_A: Relevanz der Attention-Gewichte (B, H, T, S)
        attn_weights: Attention-Gewichte A (B, H, T, S)
        attn_scores: Attention-Scores S (B, H, T, S)
        eps: Stabilisierungsterm
        
    Returns:
        R_S: Relevanz der Scores (B, H, T, S)
        
    Note:
        Diese Methode ist rechenintensiver als die approximativen Regeln,
        aber garantiert exakte Relevanz-Erhaltung.
    """
    B, H, T, S_dim = R_A.shape
    
    # Jacobi-Matrix für Softmax: J_jk = A_j(δ_jk - A_k)
    # Für LRP: R_S[k] = S[k] · Σ_j (∂A_j/∂S_k) · (R_A[j] / A[j])
    
    # Relevanz-Signal: c_j = R_A[j] / A[j]
    c = safe_divide(R_A, attn_weights, eps)  # (B, H, T, S)
    
    # Summation Σ_j c_j
    c_sum = c.sum(dim=-1, keepdim=True)  # (B, H, T, 1)
    
    # Jacobi-Vektor-Produkt: Σ_j J_jk · c_j = A_k · c_k - A_k · (Σ_j A_j · c_j)
    # = A_k · (c_k - Σ_j A_j · c_j)
    # = A_k · (c_k - <A, c>)
    A_c_product = (attn_weights * c).sum(dim=-1, keepdim=True)  # (B, H, T, 1)
    jacobian_vec_product = attn_weights * (c - A_c_product)  # (B, H, T, S)
    
    # Gradient × Input für LRP
    R_S = attn_scores * jacobian_vec_product
    
    # Konservative Normalisierung pro Query
    R_A_sum = R_A.sum(dim=-1, keepdim=True)
    R_S_sum = R_S.sum(dim=-1, keepdim=True)
    scale = safe_divide(R_A_sum, R_S_sum, eps)
    R_S = R_S * scale
    
    return R_S


__all__ = [
    "lrp_softmax",
    "lrp_softmax_jacobian",
]
