from __future__ import annotations
from typing import Tuple
import torch
from torch import Tensor
import torch.nn.functional as F


# Bei bilinearer Interpolation wird ein Wert an Position (x, y) aus den 4 nächsten Pixeln interpoliert

def compute_bilinear_weights(
    x: Tensor,
    y: Tensor,
    H: int,
    W: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # Floor-Koordinaten
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Clamp auf gültigen Bereich
    x0_clamped = x0.clamp(0, W - 1)
    x1_clamped = x1.clamp(0, W - 1)
    y0_clamped = y0.clamp(0, H - 1)
    y1_clamped = y1.clamp(0, H - 1)
    
    # Fraktionale Teile
    fx = (x - x0.float()).clamp(0, 1)
    fy = (y - y0.float()).clamp(0, 1)
    
    # Bilineare Gewichte
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    
    return x0_clamped, x1_clamped, y0_clamped, y1_clamped, w00, w10, w01, w11

# Wir berechnen alle 4 Ecken mit Koordinaten und Gewichten

def compute_bilinear_corners(
    x: Tensor,
    y: Tensor,
    H: int,
    W: int,
) -> list:
    x0 = torch.floor(x).long()
    y0 = torch.floor(y).long()
    
    fx = (x - x0.float()).clamp(0, 1)
    fy = (y - y0.float()).clamp(0, 1)
    
    return [
        (x0.clamp(0, W-1), y0.clamp(0, H-1), (1-fx)*(1-fy)),      # top-left
        ((x0+1).clamp(0, W-1), y0.clamp(0, H-1), fx*(1-fy)),      # top-right
        (x0.clamp(0, W-1), (y0+1).clamp(0, H-1), (1-fx)*fy),      # bottom-left
        ((x0+1).clamp(0, W-1), (y0+1).clamp(0, H-1), fx*fy),      # bottom-right
    ]

# Wir verteilen Relevanz auf die Feature-Map. Jeder Sampling-Punkt trägt zu den 4 nächsten Pixeln bei

def bilinear_splat_relevance(
    R_out: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    H: int,
    W: int,
    level_start_idx: int,
    eps: float = 1e-9,
) -> Tensor:
    B, T, H_heads, P, _ = sampling_locations.shape
    device = sampling_locations.device
    dtype = R_out.dtype
    
    # Ausgabe-Tensor für dieses Level
    C = R_out.shape[-1] if R_out.dim() == 3 else 1
    R_level = torch.zeros((B, H * W, C), device=device, dtype=dtype)
    
    # Normierte Koordinaten -> Pixel-Koordinaten
    x = sampling_locations[..., 0] * W - 0.5
    y = sampling_locations[..., 1] * H - 0.5
    
    # Bilineare Ecken berechnen
    corners = compute_bilinear_corners(x, y, H, W)
    
    # Relevanz vorbereiten
    if R_out.dim() == 2:
        R_out = R_out.unsqueeze(-1)
    R_broadcast = R_out.unsqueeze(2).unsqueeze(3).expand(-1, -1, H_heads, P, -1)
    attn_broadcast = attention_weights.unsqueeze(-1)
    base_weight = R_broadcast * attn_broadcast
    
    # Splatting für jede Ecke
    for b in range(B):
        for xx, yy, w_bilin in corners:
            flat_idx = (yy[b] * W + xx[b]).reshape(-1)
            flat_w = w_bilin[b].reshape(-1, 1)
            flat_R = base_weight[b].reshape(-1, C)
            weighted = flat_R * flat_w
            R_level[b].scatter_add_(0, flat_idx.unsqueeze(-1).expand(-1, C), weighted)
    
    return R_level

# Wir erstellen eine vektorisierte Version des bilinearen Splatting für bessere Performance

def bilinear_splat_relevance_vectorized(
    R_out: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    H: int,
    W: int,
    eps: float = 1e-9,
) -> Tensor:
    B, T, H_heads, P, _ = sampling_locations.shape
    device = sampling_locations.device
    dtype = R_out.dtype
    
    C = R_out.shape[-1] if R_out.dim() == 3 else 1
    if R_out.dim() == 2:
        R_out = R_out.unsqueeze(-1)
    
    # Koordinaten-Transformation
    x = sampling_locations[..., 0] * W - 0.5
    y = sampling_locations[..., 1] * H - 0.5
    
    # Bilineare Ecken
    corners = compute_bilinear_corners(x, y, H, W)
    
    # Basis-Gewicht: Attention * Relevanz
    R_expanded = R_out.unsqueeze(2).unsqueeze(3).expand(-1, -1, H_heads, P, -1)
    attn_expanded = attention_weights.unsqueeze(-1)
    base_w = R_expanded * attn_expanded
    
    # Ergebnis-Tensor
    R_level = torch.zeros((B, H * W, C), device=device, dtype=dtype)
    
    for xx, yy, bilin_w in corners:
        pixel_idx = yy * W + xx
        weighted = base_w * bilin_w.unsqueeze(-1)
        
        flat_pixel = pixel_idx.reshape(B, -1)
        flat_w = weighted.reshape(B, -1, C)
        
        R_level.scatter_add_(
            1,
            flat_pixel.unsqueeze(-1).expand(-1, -1, C),
            flat_w
        )
    
    return R_level


# Wir splatten Relevanz auf ein einzelnes Level in-place

def splat_to_level(
    R_out: Tensor,
    sampling_locations_level: Tensor,
    attention_weights_level: Tensor,
    H_l: int,
    W_l: int,
    base_idx: int,
    R_target: Tensor,
) -> None:
    B, T, H_heads, P, _ = sampling_locations_level.shape
    C = R_out.shape[-1]
    
    # Pixel-Koordinaten
    x = sampling_locations_level[..., 0] * W_l - 0.5
    y = sampling_locations_level[..., 1] * H_l - 0.5
    
    # Bilineare Ecken
    corners = compute_bilinear_corners(x, y, H_l, W_l)
    
    # Basis-Relevanz
    # Jeder Kopf trägt 1/H_heads zur Gesamtrelevanz bei
    R_expanded = R_out.unsqueeze(2).unsqueeze(3).expand(-1, -1, H_heads, P, -1)
    R_expanded = R_expanded / H_heads
    attn_expanded = attention_weights_level.unsqueeze(-1)
    base_R = R_expanded * attn_expanded
    
    # Splatting
    for xx, yy, bilin_w in corners:
        local_idx = yy * W_l + xx
        global_idx = local_idx + base_idx
        
        weighted_R = base_R * bilin_w.unsqueeze(-1)
        
        flat_idx = global_idx.reshape(B, -1)
        flat_R = weighted_R.reshape(B, -1, C)
        
        R_target.scatter_add_(
            1,
            flat_idx.unsqueeze(-1).expand(-1, -1, C),
            flat_R
        )


# Extrahiert Relevanz für ein einzelnes Level

def extract_level_relevance(
    R_source: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    target_level: int,
) -> Tuple[Tensor, int, int]:
    spatial_shapes = spatial_shapes.long()
    level_start_index = level_start_index.long()
    
    H_l = int(spatial_shapes[target_level, 0].item())
    W_l = int(spatial_shapes[target_level, 1].item())
    start = int(level_start_index[target_level].item())
    end = start + H_l * W_l
    
    R_level = R_source[:, start:end, :]
    return R_level, H_l, W_l

# Reshapet Level-Relevanz zu räumlicher Form

def reshape_to_spatial(
    R_level: Tensor,
    H: int,
    W: int,
    aggregation: str = "sum",
) -> Tensor:
    B, _, C = R_level.shape
    R_spatial = R_level.view(B, H, W, C).permute(0, 3, 1, 2)
    
    if aggregation == "sum":
        return R_spatial.sum(dim=1)
    elif aggregation == "mean":
        return R_spatial.mean(dim=1)
    elif aggregation == "max":
        return R_spatial.max(dim=1)[0]
    elif aggregation == "none":
        return R_spatial
    else:
        raise ValueError(f"Unbekannte Aggregation: {aggregation}")

# Formt flache Relevanz zu einer Heatmap um

def compute_pixel_relevance_map(
    R_source: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    target_level: int = 0,
    aggregation: str = "sum",
) -> Tensor:
    R_level, H_l, W_l = extract_level_relevance(
        R_source, spatial_shapes, level_start_index, target_level
    )
    return reshape_to_spatial(R_level, H_l, W_l, aggregation)

# Aggregiert Relevanz über alle Levels zu einer einheitlichen Heatmap

def compute_multiscale_relevance_map(
    R_source: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
    target_size: Tuple[int, int],
    aggregation: str = "sum",
    interpolation: str = "bilinear",
) -> Tensor:
    B = R_source.shape[0]
    device = R_source.device
    dtype = R_source.dtype
    L = spatial_shapes.shape[0]
    H_target, W_target = target_size
    
    R_final = torch.zeros((B, H_target, W_target), device=device, dtype=dtype)
    
    for l in range(L):
        R_level = compute_pixel_relevance_map(
            R_source=R_source,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            target_level=l,
            aggregation=aggregation,
        )
        
        R_upscaled = F.interpolate(
            R_level.unsqueeze(1),
            size=target_size,
            mode=interpolation,
            align_corners=False if interpolation == "bilinear" else None,
        ).squeeze(1)
        
        R_final = R_final + R_upscaled
    
    return R_final


__all__ = [
    "compute_bilinear_weights",
    "compute_bilinear_corners",
    "bilinear_splat_relevance",
    "bilinear_splat_relevance_vectorized",
    "splat_to_level",
    "extract_level_relevance",
    "reshape_to_spatial",
    "compute_pixel_relevance_map",
    "compute_multiscale_relevance_map",
]
