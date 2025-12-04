"""
LRP Propagatoren - Mathematische Arbeiterschaft für Layer-Propagation.

Dieses Modul enthält die puren Funktionen für die Layer-spezifische
Relevanz-Propagation. Hier wird gerechnet, nicht verwaltet.

Jede Funktion nimmt Aktivierungen und Relevanz als Input und gibt
die propagierte Relevanz zurück.

Verwendung:
    >>> from lrp_propagators import propagate_linear, propagate_multihead_attention
    >>> R_in = propagate_linear(module, activations, R_out, eps=1e-6)
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from torch import Tensor

# LRP-Regeln
from .lrp_rules_attention import (
    lrp_attention_qk_path,
    lrp_attention_to_weights,
    lrp_attention_value_path,
    lrp_softmax,
)
from .lrp_rules_deformable import msdeform_attn_lrp
from .lrp_rules_standard import (
    layernorm_lrp,
    lrp_epsilon_rule,
    residual_split,
)
from .param_patcher import (
    LRPActivations,
    LRPModuleMixin,
    LRP_LayerNorm,
    LRP_Linear,
    LRP_MSDeformAttn,
    LRP_MultiheadAttention,
)


# =============================================================================
# Logging Setup
# =============================================================================

logger = logging.getLogger("lrp.propagators")


# =============================================================================
# MS Deformable Attention Propagation
# =============================================================================


def propagate_msdeformattn(
    module: LRP_MSDeformAttn,
    activations: LRPActivations,
    R_out: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """LRP für Multi-Scale Deformable Attention.
    
    Verwendet bilineares Splatting, um Relevanz auf die Multi-Scale
    Feature-Maps zurückzupropagieren.
    
    Args:
        module: Das LRP_MSDeformAttn Modul
        activations: Gespeicherte Aktivierungen vom Forward Pass
        R_out: Ausgabe-Relevanz
        eps: Epsilon für numerische Stabilität
        
    Returns:
        R_source: Relevanz auf den Source-Features
    """
    # Erforderliche Aktivierungen
    sampling_locations = activations.sampling_locations
    attention_weights = activations.deform_attention_weights
    spatial_shapes = activations.spatial_shapes
    level_start_index = activations.level_start_index
    
    if any(x is None for x in [sampling_locations, attention_weights, spatial_shapes]):
        logger.warning("MSDeformAttn: Fehlende Aktivierungen")
        return R_out
    
    # Optionale Projektionsgewichte
    W_V = activations.W_V
    W_O = activations.W_O
    value_features = activations.input_flatten
    
    # WICHTIG: R_out Shape muss zu sampling_locations passen!
    # sampling_locations hat Shape (B, T, H, L, P, 2)
    # R_out muss Shape (B, T, C) haben
    B_loc = sampling_locations.shape[0]
    T_loc = sampling_locations.shape[1]
    
    # Prüfe ob R_out transponiert werden muss
    if R_out.dim() == 3:
        if R_out.shape[0] == T_loc and R_out.shape[1] == B_loc:
            # R_out hat Shape (T, B, C) -> transponiere zu (B, T, C)
            logger.debug(f"MSDeformAttn: Transponiere R_out von {R_out.shape} zu (B, T, C)")
            R_out = R_out.transpose(0, 1).contiguous()
    
    logger.debug(f"MSDeformAttn: R_out.shape={R_out.shape}, sampling_locations.shape={sampling_locations.shape}")
    
    # Hauptpropagation via bilineares Splatting
    R_source = msdeform_attn_lrp(
        R_out=R_out,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        value_features=value_features,
        W_V=W_V,
        W_O=W_O,
        use_value_weighting=(W_V is not None and W_O is not None),
        eps=eps,
    )
    
    logger.debug(f"MSDeformAttn: R_source.shape={R_source.shape}")
    
    return R_source


# =============================================================================
# Multi-Head Attention Propagation
# =============================================================================


def propagate_multihead_attention(
    module: LRP_MultiheadAttention,
    activations: LRPActivations,
    R_out: Tensor,
    eps: float = 1e-6,
    attn_qk_share: float = 0.0,
) -> Tensor:
    """LRP für Multi-Head Attention.
    
    Propagiert durch den Value-Pfad (O = A·V) und optional
    durch den Q/K-Pfad für vollständige Attribution.
    
    Args:
        module: Das LRP_MultiheadAttention Modul
        activations: Gespeicherte Aktivierungen vom Forward Pass
        R_out: Ausgabe-Relevanz
        eps: Epsilon für numerische Stabilität
        attn_qk_share: Anteil der Q/K-Pfad Attribution (0.0-1.0)
        
    Returns:
        R_in: Relevanz auf den Eingabe-Features
    """
    # Erforderliche Aktivierungen
    attn_weights = activations.attn_weights
    V = activations.V
    
    if attn_weights is None or V is None:
        logger.warning("MHA: Fehlende Aktivierungen")
        return R_out
    
    # V hat Shape (B, H, S, Dh) - extrahiere Dimensionen
    B, H, S, Dh = V.shape
    C = H * Dh
    
    # Memory debug info
    tensor_size_mb = (attn_weights.numel() * 4) / (1024 * 1024)
    logger.debug(f"MHA: attn_weights size={tensor_size_mb:.1f}MB, shape={attn_weights.shape}")
    
    # R_out von (B, T, C) oder (T, B, C) zu (B, H, T, Dh) reshapen
    total_elements = R_out.numel()
    
    # Debug: Log R_out shape
    logger.debug(f"MHA: R_out.shape={R_out.shape}, V.shape={V.shape}, C={C}, total={total_elements}")
    
    # Prüfe ob R_out transponiert werden muss (T, B, C) -> (B, T, C)
    R_out_heads = _reshape_R_out_for_attention(R_out, B, H, Dh, C)
    if R_out_heads is None:
        return R_out
    
    # Value-Pfad LRP: R_O -> R_V
    R_V = lrp_attention_value_path(
        R_out=R_out_heads,
        attn_weights=attn_weights,
        V=V,
        eps=eps,
    )
    
    # Optional: Q/K-Pfad für vollständigere Attribution
    if attn_qk_share > 0:
        Q = activations.Q
        K = activations.K
        attn_scores = activations.attn_scores
        
        if all(x is not None for x in [Q, K, attn_scores]):
            # Relevanz zu Attention-Gewichten
            R_A = lrp_attention_to_weights(R_out_heads, attn_weights, V, eps)
            
            # Softmax LRP: R_A -> R_S
            R_S = lrp_softmax(R_A, attn_weights, attn_scores, rule="gradient", eps=eps)
            
            # Q/K-Pfad: R_S -> R_Q, R_K
            R_Q, R_K = lrp_attention_qk_path(R_S, Q, K, activations.scale, eps)
            
            # Mische V-Pfad mit Q/K-Pfad
            alpha = attn_qk_share
            R_in = (1 - alpha) * R_V + alpha * (R_Q + R_K) / 2
            return R_in
    
    # Nur Value-Pfad - propagiere durch Value-Projektion
    W_V = activations.W_V
    logger.debug(f"MHA: R_V.shape={R_V.shape}")
    
    if W_V is not None:
        # Reshape für Projektion: (B, H, S, Dh) -> (B, S, C)
        B, H, S, Dh = R_V.shape
        C = H * Dh
        R_V_flat = R_V.permute(0, 2, 1, 3).reshape(B, S, C)
        
        # LRP durch Linear-Projektion
        x_in = activations.input
        if isinstance(x_in, tuple):
            x_in = x_in[2]  # value input bei (q, k, v)
        
        if x_in is not None:
            logger.debug(f"MHA: x_in.shape={x_in.shape}, W_V.shape={W_V.shape}, R_V_flat.shape={R_V_flat.shape}")
            R_in = lrp_epsilon_rule(
                a_in=x_in,
                weight=W_V.view(C, C),
                R_out=R_V_flat,
                eps=eps,
            )
            logger.debug(f"MHA: R_in.shape={R_in.shape}")
            return R_in
    
    return R_V


def _reshape_R_out_for_attention(
    R_out: Tensor,
    B: int,
    H: int,
    Dh: int,
    C: int,
) -> Optional[Tensor]:
    """Hilfsfunktion zum Reshapen von R_out für Attention.
    
    Args:
        R_out: Relevanz-Tensor in verschiedenen möglichen Formaten
        B: Batch-Größe
        H: Anzahl Attention Heads
        Dh: Dimension pro Head
        C: Gesamt-Embedding-Dimension (H * Dh)
        
    Returns:
        R_out_heads: Reshaped zu (B, H, T, Dh) oder None bei Fehler
    """
    total_elements = R_out.numel()
    
    if R_out.dim() == 3:
        # Erkenne Format: (B, T, C) vs (T, B, C)
        if R_out.shape[2] == C:
            # Letzte Dimension ist C
            if R_out.shape[0] == B:
                # Format ist (B, T, C)
                T = R_out.shape[1]
                logger.debug(f"MHA: Format (B, T, C) mit T={T}")
                return R_out.view(B, T, H, Dh).permute(0, 2, 1, 3)  # (B, H, T, Dh)
            elif R_out.shape[1] == B:
                # Format ist (T, B, C) - transponiere zu (B, T, C)
                T = R_out.shape[0]
                logger.debug(f"MHA: Format (T, B, C) mit T={T} - transponiere zu (B, T, C)")
                R_out_btc = R_out.transpose(0, 1).contiguous()
                return R_out_btc.view(B, T, H, Dh).permute(0, 2, 1, 3)
            else:
                # Fallback: versuche aus Gesamtgröße T zu berechnen
                T = total_elements // (B * C)
                logger.debug(f"MHA: Unbekanntes Format, berechne T={T} aus Gesamtgröße")
                R_out_flat = R_out.view(B, T, C)
                return R_out_flat.view(B, T, H, Dh).permute(0, 2, 1, 3)
        else:
            logger.warning(f"MHA: Unerwartete R_out Shape: {R_out.shape}, letzte Dimension != C={C}")
            return None
    
    elif R_out.dim() == 4 and R_out.shape[1] == H:
        # Bereits im richtigen Format (B, H, T, Dh)
        return R_out
    
    elif total_elements % C == 0:
        # R_out ist flach oder hat andere Shape
        T = total_elements // (B * C)
        logger.debug(f"MHA: R_out Shape {R_out.shape} -> reshape zu ({B}, {T}, {C})")
        R_out_reshaped = R_out.view(B, T, C)
        return R_out_reshaped.view(B, T, H, Dh).permute(0, 2, 1, 3)
    
    else:
        logger.warning(f"MHA: Unerwartete R_out Shape: {R_out.shape} ({total_elements} Elemente)")
        return None


# =============================================================================
# LayerNorm Propagation
# =============================================================================


def propagate_layernorm(
    module: LRP_LayerNorm,
    activations: LRPActivations,
    R_out: Tensor,
    eps: float = 1e-6,
    ln_rule: str = "taylor",
) -> Tensor:
    """LRP für LayerNorm.
    
    Verwendet verschiedene Strategien je nach Konfiguration:
    - taylor: Taylor-Expansion 1. Ordnung
    - zsign: Vorzeichenbewahrende z-Regel
    
    Args:
        module: Das LRP_LayerNorm Modul
        activations: Gespeicherte Aktivierungen vom Forward Pass
        R_out: Ausgabe-Relevanz
        eps: Epsilon für numerische Stabilität
        ln_rule: LRP-Regel für LayerNorm ("taylor", "zsign")
        
    Returns:
        R_in: Relevanz auf den Eingabe-Features
    """
    x = activations.input
    gamma = activations.gamma
    beta = activations.beta
    
    if x is None or gamma is None:
        logger.warning("LayerNorm: Fehlende Aktivierungen")
        return R_out
    
    logger.debug(f"LayerNorm: x.shape={x.shape}, R_out.shape={R_out.shape}")
    
    # Prüfe ob Shapes kompatibel sind
    if x.shape != R_out.shape:
        R_out = _align_shapes_for_layernorm(x, R_out)
        # _align_shapes_for_layernorm gibt immer einen Tensor zurück (nie None)
        # Bei Shape-Mismatch wird R_out unverändert zurückgegeben
        if x.shape != R_out.shape:
            # Immer noch inkompatibel - Identität zurückgeben
            logger.warning(f"LayerNorm: Shapes weiterhin inkompatibel nach Alignment ({x.shape} vs {R_out.shape}) - Identität")
            return R_out
    
    R_in = layernorm_lrp(
        x=x,
        gamma=gamma,
        beta=beta if beta is not None else torch.zeros_like(gamma),
        R_out=R_out,
        rule=ln_rule,
        ln_eps=module.eps,
        lrp_eps=eps,
    )
    
    return R_in


def _align_shapes_for_layernorm(x: Tensor, R_out: Tensor) -> Optional[Tensor]:
    """Versucht Shapes zwischen x und R_out anzugleichen.
    
    Args:
        x: Input-Aktivierung
        R_out: Ausgabe-Relevanz
        
    Returns:
        R_out mit angepasster Shape oder Original R_out bei Fehler (nie None!)
    """
    # Versuche Transposition: (T, B, C) <-> (B, T, C)
    if x.dim() == 3 and R_out.dim() == 3:
        # Prüfe ob Transposition hilft
        R_out_transposed = R_out.transpose(0, 1)
        if R_out_transposed.shape == x.shape:
            logger.debug(f"LayerNorm: Transponiere R_out von {R_out.shape} zu {R_out_transposed.shape}")
            return R_out_transposed
        elif R_out.numel() == x.numel():
            # Gleiche Anzahl Elemente - reshape
            logger.debug(f"LayerNorm: Reshape R_out von {R_out.shape} zu {x.shape}")
            return R_out.view(x.shape)
        else:
            # Shapes sind inkompatibel - gib R_out unverändert zurück mit Warning
            logger.warning(f"LayerNorm: Kann Shapes nicht anpassen ({x.shape} vs {R_out.shape})")
            # Gib NICHT None zurück, sondern das Original!
            return R_out
    else:
        logger.warning(f"LayerNorm: Shape-Mismatch x={x.shape} vs R_out={R_out.shape}")
        # Gib NICHT None zurück, sondern das Original!
        return R_out


# =============================================================================
# Linear Layer Propagation
# =============================================================================


def propagate_linear(
    module: LRP_Linear,
    activations: LRPActivations,
    R_out: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """LRP für Linear Layer mit ε-Regel.
    
    Args:
        module: Das LRP_Linear Modul
        activations: Gespeicherte Aktivierungen vom Forward Pass
        R_out: Ausgabe-Relevanz
        eps: Epsilon für numerische Stabilität
        
    Returns:
        R_in: Relevanz auf den Eingabe-Features
    """
    x = activations.input
    weight = activations.weights
    bias = activations.bias
    
    if x is None or weight is None:
        logger.warning("Linear: Fehlende Aktivierungen")
        return R_out
    
    R_in = lrp_epsilon_rule(
        a_in=x,
        weight=weight,
        R_out=R_out,
        bias=bias,
        eps=eps,
    )
    
    return R_in


# =============================================================================
# Residual Connection Propagation
# =============================================================================


def propagate_residual(
    x: Tensor,
    Fx: Tensor,
    R_y: Tensor,
    mode: str = "proportional",
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Propagiert Relevanz durch eine Residual-Verbindung y = x + F(x).
    
    Args:
        x: Skip-Pfad Aktivierungen
        Fx: Transform-Pfad Aktivierungen
        R_y: Relevanz am Ausgang
        mode: Split-Modus ("proportional", "equal", "skip_only", "transform_only")
        eps: Epsilon für numerische Stabilität
        
    Returns:
        (R_x, R_Fx): Relevanz für Skip- und Transform-Pfad
    """
    R_x, R_Fx = residual_split(
        x=x,
        Fx=Fx,
        Ry=R_y,
        mode=mode,
        eps=eps,
    )
    
    return R_x, R_Fx


# =============================================================================
# Generic Layer Dispatch
# =============================================================================


def propagate_layer(
    module,
    activations: LRPActivations,
    R_out: Tensor,
    eps: float = 1e-6,
    ln_rule: str = "taylor",
    attn_qk_share: float = 0.0,
) -> Tensor:
    """Generische Dispatch-Funktion für Layer-Propagation.
    
    Wählt automatisch die passende Propagationsfunktion basierend auf
    dem Modul-Typ aus.
    
    Args:
        module: Das PyTorch-Modul (muss LRPModuleMixin implementieren)
        activations: Gespeicherte Aktivierungen vom Forward Pass
        R_out: Ausgabe-Relevanz
        eps: Epsilon für numerische Stabilität
        ln_rule: LRP-Regel für LayerNorm
        attn_qk_share: Anteil der Q/K-Pfad Attribution
        
    Returns:
        R_in: Propagierte Relevanz
    """
    # Prüfe ob es ein LRP-fähiges Modul ist
    if not isinstance(module, LRPModuleMixin):
        # Für nicht-LRP-Module: Identitäts-Propagation
        return R_out
    
    # Hole gespeicherte Aktivierungen falls nicht übergeben
    if activations is None:
        activations = module.activations
    
    if activations is None or activations.input is None:
        logger.warning(f"Keine Aktivierungen für {type(module).__name__} - Identität")
        return R_out
    
    # Dispatch basierend auf Modul-Typ
    if isinstance(module, LRP_MSDeformAttn):
        return propagate_msdeformattn(module, activations, R_out, eps)
    
    elif isinstance(module, LRP_MultiheadAttention):
        return propagate_multihead_attention(module, activations, R_out, eps, attn_qk_share)
    
    elif isinstance(module, LRP_LayerNorm):
        return propagate_layernorm(module, activations, R_out, eps, ln_rule)
    
    elif isinstance(module, LRP_Linear):
        return propagate_linear(module, activations, R_out, eps)
    
    else:
        # Fallback: Identität
        logger.debug(f"Unbekannter LRP-Modul-Typ: {type(module).__name__}")
        return R_out


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Layer-spezifische Propagation
    "propagate_msdeformattn",
    "propagate_multihead_attention",
    "propagate_layernorm",
    "propagate_linear",
    "propagate_residual",
    
    # Generische Dispatch-Funktion
    "propagate_layer",
]
