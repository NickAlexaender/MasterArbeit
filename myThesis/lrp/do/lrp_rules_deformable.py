"""
LRP-Regeln für MSDeformAttn (Multi-Scale Deformable Attention).

Dieses Modul ist der Einstiegspunkt für Deformable Attention LRP und enthält
die High-Level Regeln. Die mathematischen Operationen werden aus lrp_deform_ops
importiert, das Aktivierungs-Capturing aus lrp_deform_capture.

MSDeformAttn sampelt Werte an vorhergesagten Lokationen p_q + Δp_{mqk} mit
Attention-Gewichten A_{mqk}. Da bilineare Interpolation verwendet wird:
    x(p) = Σ_i g(p, p_i) · x_i

muss die Relevanz entsprechend auf die 4 nächsten Integer-Pixel verteilt werden:
    R_pixel = R_output · W_attn · W_bilinear

Hauptfunktionen:
    - msdeform_attn_lrp: Hauptfunktion zur Relevanz-Rückpropagation
    - msdeform_attn_lrp_with_value: Erweiterte LRP mit Value-Pfad
    - deform_value_path_lrp: Value-Path LRP für einen Kanal
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from .tensor_ops import safe_divide

# Import der mathematischen Operationen
from .lrp_deform_ops import (
    compute_bilinear_weights,
    compute_bilinear_corners,
    bilinear_splat_relevance,
    bilinear_splat_relevance_vectorized,
    splat_to_level,
    compute_pixel_relevance_map,
    compute_multiscale_relevance_map,
)

# Import der Capture-Funktionen
from .lrp_deform_capture import (
    attach_msdeformattn_capture,
    _resolve_msdeformattn_class,
    _ForwardPatchHandle,
)


# =============================================================================
# Haupt-LRP-Funktion für MSDeformAttn
# =============================================================================


def msdeform_attn_lrp(
    R_out: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    value_features: Optional[Tensor] = None,
    W_V: Optional[Tensor] = None,
    W_O: Optional[Tensor] = None,
    use_value_weighting: bool = False,
    eps: float = 1e-9,
) -> Tensor:
    """Führt LRP durch MSDeformAttn mit Bilinear-Interpolations-Dekomposition durch.
    
    Diese Funktion ist das Herzstück der Deformable Attention LRP. Sie verteilt
    Relevanz von den Ausgabe-Tokens zurück auf die Eingabe-Feature-Map-Pixel
    unter Berücksichtigung der:
    
    1. Attention-Gewichte A_{mqk}: Wie stark jeder Sampling-Punkt gewichtet wird
    2. Bilinearen Interpolation: Wie die Sampling-Positionen auf Integer-Pixel abgebildet werden
    3. Multi-Scale Struktur: Verschiedene Auflösungsebenen der Feature-Pyramide
    
    Die mathematische Formel lautet:
        R_pixel = R_output · W_attn · W_bilinear
    
    wobei:
        - R_output: Relevanz am Ausgang der Attention
        - W_attn: Attention-Gewichte A_{mqk}
        - W_bilinear: Bilineare Interpolationsgewichte basierend auf der Distanz
    
    Args:
        R_out: Ausgabe-Relevanz (B, T, C) wobei T die Query-Länge ist
        sampling_locations: Normierte Sampling-Positionen (B, T, H, L, P, 2)
            - H: Anzahl der Attention-Köpfe
            - L: Anzahl der Scale-Levels
            - P: Anzahl der Sampling-Punkte pro Level
            - 2: (x, y) Koordinaten in [0, 1]
        attention_weights: Attention-Gewichte (B, T, H, L, P), Summe über P ≈ 1
        spatial_shapes: (L, 2) Dimensionen (H_l, W_l) pro Level
        level_start_index: (L,) Start-Indizes pro Level in flacher Darstellung
        value_features: Optional (B, S, C) Value-Projektionen für gewichtete Verteilung
        W_V: Optional (H, D_h, C) Value-Projektionsgewichte
        W_O: Optional (H, D_h, C) Output-Projektionsgewichte
        use_value_weighting: Falls True, werden Value-Features einbezogen
        eps: Numerische Stabilisierung
    
    Returns:
        R_source: (B, S, C) Relevanz auf den Quell-Tokens der flachen Feature-Map
        
    Hinweis:
        S = Σ_l H_l × W_l ist die Gesamtzahl der Quell-Tokens über alle Levels
    """
    B, T, H_heads, L, P, _ = sampling_locations.shape
    device = sampling_locations.device
    dtype = R_out.dtype
    
    # Sicherstellen, dass R_out 3D ist
    if R_out.dim() == 2:
        R_out = R_out.unsqueeze(-1)
    C = R_out.shape[-1]
    
    # Gesamtzahl der Quell-Tokens berechnen
    spatial_shapes = spatial_shapes.long()
    level_start_index = level_start_index.long()
    last_H, last_W = spatial_shapes[-1].tolist()
    S_total = int(level_start_index[-1].item() + last_H * last_W)
    
    # Ergebnis-Tensor initialisieren
    R_source = torch.zeros((B, S_total, C), device=device, dtype=dtype)
    
    # Optional: Value-gewichtete Verteilung (TODO: Implementierung)
    if use_value_weighting and value_features is not None:
        pass
    
    # Über alle Levels iterieren und Relevanz splatten
    for l in range(L):
        H_l = int(spatial_shapes[l, 0].item())
        W_l = int(spatial_shapes[l, 1].item())
        base_idx = int(level_start_index[l].item())
        
        # Extrahiere Daten für dieses Level
        loc_l = sampling_locations[:, :, :, l]   # (B, T, H_heads, P, 2)
        attn_l = attention_weights[:, :, :, l]   # (B, T, H_heads, P)
        
        # Splatte Relevanz auf dieses Level
        splat_to_level(
            R_out=R_out,
            sampling_locations_level=loc_l,
            attention_weights_level=attn_l,
            H_l=H_l,
            W_l=W_l,
            base_idx=base_idx,
            R_target=R_source,
        )
    
    return R_source


def msdeform_attn_lrp_with_value(
    R_out: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    value_features: Tensor,
    W_V: Tensor,
    W_O: Tensor,
    eps: float = 1e-9,
) -> Tensor:
    """Erweiterte LRP mit Value-Pfad-Berücksichtigung.
    
    Diese Variante berücksichtigt nicht nur Attention und bilineare Gewichte,
    sondern auch den Beitrag der Value-Projektionen W_V und W_O.
    
    Formel:
        R_pixel = R_output · W_attn · W_bilinear · |V · W_O|
    
    Args:
        R_out: (B, T, C) Ausgabe-Relevanz
        sampling_locations: (B, T, H, L, P, 2) Sampling-Positionen
        attention_weights: (B, T, H, L, P) Attention-Gewichte
        spatial_shapes: (L, 2) Feature-Map-Größen
        level_start_index: (L,) Start-Indizes
        value_features: (B, S, C) Value-Projektionen
        W_V: (H, D_h, C) Value-Gewichte
        W_O: (H, D_h, C) Output-Gewichte
        eps: Stabilisierung
    
    Returns:
        R_source: (B, S, C) Relevanz auf Quell-Tokens
    """
    B, T, H_heads, L, P, _ = sampling_locations.shape
    device = sampling_locations.device
    dtype = R_out.dtype
    
    if R_out.dim() == 2:
        R_out = R_out.unsqueeze(-1)
    C = R_out.shape[-1]
    
    # Gesamtzahl der Quell-Tokens
    spatial_shapes = spatial_shapes.long()
    level_start_index = level_start_index.long()
    S_total = int(level_start_index[-1].item() + 
                  spatial_shapes[-1, 0].item() * spatial_shapes[-1, 1].item())
    
    # Berechne Value-Beitrag für jeden Quell-Token
    D_h = W_V.shape[1]
    
    # Value-Output-Score: |V| als Gewichtung
    V_magnitude = value_features.abs().mean(dim=-1, keepdim=True)  # (B, S, 1)
    
    # Basis-LRP durchführen
    R_base = msdeform_attn_lrp(
        R_out=R_out,
        sampling_locations=sampling_locations,
        attention_weights=attention_weights,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        eps=eps,
    )
    
    # Value-Gewichtung anwenden
    R_weighted = R_base * (V_magnitude + eps)
    
    # Normalisierung zur Relevanz-Erhaltung
    R_sum_out = R_out.sum()
    R_sum_weighted = R_weighted.sum()
    R_source = R_weighted * safe_divide(R_sum_out, R_sum_weighted, eps=eps)
    
    return R_source


# =============================================================================
# Value-Path LRP für Deformable Attention
# =============================================================================


def deform_value_path_lrp(
    R_channel: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    value_proj: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    W_V: Optional[Tensor] = None,
    W_O: Optional[Tensor] = None,
    eps: float = 1e-9,
) -> Tensor:
    """Verteilt Relevanz eines Output-Kanals zurück auf Quell-Tokens via Value-Pfad.
    
    Implementiert die vollständige LRP-Rückpropagation durch den Value-Pfad
    der Deformable Attention:
    
        out_c = Σ_{m,q,k} A_{mqk} × bilinear_sample(V_proj, p_{mqk}) × W_O[h,d,c]
    
    Die Relevanz R_c für Kanal c wird verteilt als:
        R_source = R_c × A × W_bilinear × |V × W_O_c|
    
    Args:
        R_channel: (B, T, 1) Relevanz für einen spezifischen Output-Kanal
        sampling_locations: (B, T, H, L, P, 2) Sampling-Positionen
        attention_weights: (B, T, H, L, P) Attention-Gewichte
        value_proj: (B, S, H, D_h) projizierte Values
        spatial_shapes: (L, 2) Feature-Map-Größen
        level_start_index: (L,) Start-Indizes
        W_V: (H, D_h, C) Value-Projektion
        W_O: (H, D_h, C) Output-Projektion
        eps: Stabilisierung
    
    Returns:
        R_source: (B, S, 1) Relevanz auf Quell-Tokens
    """
    B, T, H_heads, L, P, _ = sampling_locations.shape
    device = sampling_locations.device
    dtype = R_channel.dtype
    
    spatial_shapes = spatial_shapes.long()
    level_start_index = level_start_index.long()
    S_total = int(level_start_index[-1].item() + 
                  spatial_shapes[-1, 0].item() * spatial_shapes[-1, 1].item())
    
    # Ergebnis-Tensor
    R_source = torch.zeros((B, S_total, 1), device=device, dtype=dtype)
    
    # Berechne Value-Wichtigkeit falls W_O gegeben
    if W_O is not None and value_proj is not None:
        V_importance = torch.einsum("bshd,hdc->bshc", value_proj, W_O)
        V_importance = V_importance.abs().mean(dim=-1)  # (B, S, H)
    else:
        V_importance = None
    
    # Über Levels iterieren
    for l in range(L):
        H_l = int(spatial_shapes[l, 0].item())
        W_l = int(spatial_shapes[l, 1].item())
        base_idx = int(level_start_index[l].item())
        
        loc_l = sampling_locations[:, :, :, l]
        attn_l = attention_weights[:, :, :, l]
        
        # Pixel-Koordinaten
        x = loc_l[..., 0] * W_l - 0.5
        y = loc_l[..., 1] * H_l - 0.5
        
        # Bilineare Ecken
        corners = compute_bilinear_corners(x, y, H_l, W_l)
        
        # Basis-Relevanz
        R_expanded = R_channel.unsqueeze(2).unsqueeze(3).expand(-1, -1, H_heads, P, -1)
        attn_expanded = attn_l.unsqueeze(-1)
        
        # Optional: Value-Gewichtung
        if V_importance is not None:
            V_mean = V_importance[:, base_idx:base_idx+H_l*W_l].mean(dim=1)
            V_weight = V_mean.unsqueeze(1).unsqueeze(3).unsqueeze(4)
            base_R = R_expanded * attn_expanded * V_weight
        else:
            base_R = R_expanded * attn_expanded
        
        for xx, yy, bilin_w in corners:
            local_idx = yy * W_l + xx
            global_idx = local_idx + base_idx
            
            weighted_R = base_R * bilin_w.unsqueeze(-1)
            
            flat_idx = global_idx.reshape(B, -1)
            flat_R = weighted_R.reshape(B, -1, 1)
            
            R_source.scatter_add_(1, flat_idx.unsqueeze(-1), flat_R)
    
    return R_source


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Capture-Funktionen (re-export aus lrp_deform_capture)
    "attach_msdeformattn_capture",
    "_resolve_msdeformattn_class",
    "_ForwardPatchHandle",
    
    # Haupt-LRP-Funktionen
    "msdeform_attn_lrp",
    "msdeform_attn_lrp_with_value",
    "deform_value_path_lrp",
    
    # Operationen (re-export aus lrp_deform_ops)
    "bilinear_splat_relevance",
    "bilinear_splat_relevance_vectorized",
    "compute_bilinear_weights",
    "compute_pixel_relevance_map",
    "compute_multiscale_relevance_map",
]
