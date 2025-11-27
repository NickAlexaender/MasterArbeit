"""
LRP-Regeln für Residual, LayerNorm und Attention-Value-Pfad.
Hinweis: Diese Implementierungen sind robuste, numerisch stabile Varianten
und dienen als Bausteine für die LRPEngine.
"""
from __future__ import annotations

from typing import Tuple
import torch
from torch import Tensor


def _add_signed_eps(t: Tensor, eps: float) -> Tensor:
    """Numerisch stabile Denominator-Korrektur: t + eps * sign_or_one(t).

    Falls t==0 ⇒ addiere +eps. Für negative t ⇒ subtrahiere eps.
    """
    sign_or_one = torch.where(t >= 0, torch.ones_like(t), -torch.ones_like(t))
    return t + eps * sign_or_one


def residual_split(x: Tensor, Fx: Tensor, Ry: Tensor, mode: str = "energy", eps: float = 1e-6) -> Tuple[Tensor, Tensor]:
    """Teile Relevanz Ry zwischen Skip-Pfad x und Transformationspfad F(x) auf.

    - energy: proportional zu ||x||^2 vs. ||F(x)||^2 je Token.
    - dotpos: proportional zu positiven Skalarprodukten mit y = x + F(x).
    - zsign:  vorzeichenbewahrende z-Regel auf Tokenebene (Aggregation über Kanäle).
    """
    # Vereinheitliche Form (B, T, C)
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if Fx.dim() == 2:
        Fx = Fx.unsqueeze(1)
    if Ry.dim() == 2:
        Ry = Ry.unsqueeze(1)

    if mode == "energy":
        ex = (x.pow(2)).sum(dim=-1, keepdim=True) + 1e-12
        ef = (Fx.pow(2)).sum(dim=-1, keepdim=True) + 1e-12
        rx = ex / (ex + ef) * Ry
        rFx = Ry - rx
        return rx, rFx
    elif mode == "dotpos":
        def posdot(a: Tensor, b: Tensor) -> Tensor:
            return torch.clamp((a * b), min=0).sum(dim=-1, keepdim=True) + 1e-12
        y = x + Fx
        px = posdot(x, y)
        pf = posdot(Fx, y)
        rx = px / (px + pf) * Ry
        rFx = Ry - rx
        return rx, rFx
    elif mode == "zsign":
        # Signed z-rule auf Tokenebene: z_x = sum_c x, z_f = sum_c F(x)
        zx = x.sum(dim=-1, keepdim=True)
        zf = Fx.sum(dim=-1, keepdim=True)
        denom = _add_signed_eps(zx + zf, eps)
        rx = (zx / denom) * Ry
        rFx = Ry - rx
        return rx, rFx
    else:
        raise ValueError("unknown residual split")


def layernorm_backshare(x: Tensor, gamma: Tensor, beta: Tensor, Ry: Tensor, rule: str = "abs-grad-xmu", eps: float = 1e-6) -> Tensor:
    """Verteile Relevanz über LayerNorm-Eingang x.

    Zwei Varianten:
    - "xmu": rein proportional zu |x-μ| pro Token
    - "abs-grad-xmu": Absolutgradient gewichtet |x-μ| (benötigt Autograd-Kontext)
    - "zsign": vorzeichenbewahrende z-Regel mit z_k = (gamma_k/sigma)*(x_k - μ)
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if Ry.dim() == 2:
        Ry = Ry.unsqueeze(1)

    if rule == "xmu":
        xm = (x - x.mean(dim=-1, keepdim=True)).abs()
        w = xm / (xm.sum(dim=-1, keepdim=True) + 1e-12)
        return w * Ry
    elif rule == "zsign":
        # LayerNorm: y = gamma * (x - μ)/σ + beta
        xmu = x - x.mean(dim=-1, keepdim=True)
        sigma = xmu.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        # gamma Form: (C,) oder (1,C); broadcast auf (B,T,C)
        g = gamma.view(1, 1, -1)
        z = (g / sigma) * xmu  # (B,T,C)
        # z-Regel über Kanäle je Token
        zsum = _add_signed_eps(z.sum(dim=-1, keepdim=True), eps)
        # Summiere Ry über Kanäle (Relevanzskalare pro Token)
        Ry_tok = Ry.sum(dim=-1, keepdim=True)
        return (z / zsum) * Ry_tok
    else:  # "abs-grad-xmu"
        # Lokale Gradienten über die LN-Formel; x muss dazu requires_grad=True sein
        x_req = x.detach().requires_grad_(True)
        xmu = x_req - x_req.mean(dim=-1, keepdim=True)
        sigma = xmu.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        y = gamma * (xmu / sigma) + beta
        # Skalariere mit Ry und leite nach x_req ab
        L = (y * Ry).sum()
        g = torch.autograd.grad(outputs=L, inputs=x_req, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if g is None:
            g = torch.ones_like(x_req)
        g = g.abs()
        w = (g * xmu.abs())
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
        return w * Ry


def value_path_split(
    Ro_j: Tensor,
    attn_weights: Tensor,
    Vproj: Tensor,
    W_O_j: Tensor,
    use_abs: bool | None = None,
    eps: float = 1e-6,
    ) -> Tensor:
    """Verteilt Relevanz von Ausgabetoken t auf Quell-Token s über den Value-Pfad.

    Parameter:
    - Ro_j: (B,T,1) Relevanz für Zielkanal j pro Ziel-Token t
    - attn_weights: (B,H,T,S)
    - Vproj: (B,H,S,Dh)  Value-Projektionen je Head
    - W_O_j: (H,Dh,1)    Spaltenvektor der Output-Projektion für Kanal j
    - use_abs: None ⇒ vorzeichenbewahrende z-Regel; True ⇒ |.|-Gewichte; False ⇒ quadratische Gewichte

    Rückgabe:
    - Rs: (B,S,1) Relevanz auf die Quell-Token s aggregiert über t
    """
    B, H, T, S = attn_weights.shape
    # Effektiver signed Beitrag Z_{t,s} = sum_h a_{t,s,h} * <V_{s,h}, W_O_j[h]>
    # score_h: (B,S,1); nach Broadcast -> (B,T,S,1)
    scores = []
    for h in range(H):
        vs = Vproj[:, h]          # (B,S,Dh)
        wo = W_O_j[h]             # (Dh,1)
        score = torch.einsum("bsd,dk->bsk", vs, wo)  # (B,S,1)
        score = score.unsqueeze(1).expand(-1, T, -1, -1)  # (B,T,S,1)
        a = attn_weights[:, h].unsqueeze(-1)            # (B,T,S,1)
        scores.append(a * score)                        # signed
    Z = torch.stack(scores, dim=1).sum(dim=1)           # (B,T,S,1)

    if use_abs is True:
        C = Z.abs()
        Csum = C.sum(dim=2, keepdim=True) + 1e-12
        w = C / Csum
        Rs = (w * Ro_j).sum(dim=1)
        return Rs
    elif use_abs is False:
        C = Z.pow(2)
        Csum = C.sum(dim=2, keepdim=True) + 1e-12
        w = C / Csum
        Rs = (w * Ro_j).sum(dim=1)
        return Rs
    else:
        # Vorzeichenbewahrende z-Regel über s je t
        Zsum = _add_signed_eps(Z.sum(dim=2, keepdim=True), eps)  # (B,T,1,1)
        w = Z / Zsum                                      # (B,T,S,1)
        Rs = (w * Ro_j).sum(dim=1)                        # (B,S,1)
        return Rs


def value_path_split_deform(
    Ro_j: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
) -> Tensor:
    """Verteile Relevanz von Ziel-Tokens t zurück auf Quelle S der MSDeformAttn.

    Bilineares "Splatting" der attention_weights auf 4 Nachbarn je Sample-Punkt.

    Parameter:
    - Ro_j: (B,T,1) Relevanz auf Ziel-Tokens (z. B. aus rFx_scalar für gewählten Kanal)
    - sampling_locations: (B,T,H,L,P,2) normierte Koordinaten je Level in [0,1]
    - attention_weights: (B,T,H,L,P) zugehörige Gewichte (>=0, Summe über P typ. 1)
    - spatial_shapes: (L,2) je Level (H_l, W_l)
    - level_start_index: (L,) Start-Offsets pro Level in der flachen S-Dimension

    Rückgabe:
    - Rs: (B,S,1) Relevanz auf Quelle (über T/H/L/P aggregiert)
    """
    B, T, Hh, L, P, _ = sampling_locations.shape
    device = sampling_locations.device
    dtype = sampling_locations.dtype

    # Gesamt-Tokenzahl S bestimmen
    spatial_shapes = spatial_shapes.to(torch.long)
    level_start_index = level_start_index.to(torch.long)
    last_h, last_w = spatial_shapes[-1]
    S_total = int((level_start_index[-1] + last_h * last_w).item())

    Rs = torch.zeros((B, S_total, 1), device=device, dtype=Ro_j.dtype)

    # Broadcast Relevanz über Köpfe/Punkte
    Ro_broadcast = Ro_j.view(B, T, 1, 1)  # (B,T,1,1)

    for l in range(L):
        Hl = int(spatial_shapes[l, 0].item())
        Wl = int(spatial_shapes[l, 1].item())
        base = int(level_start_index[l].item())

        loc = sampling_locations[:, :, :, l]  # (B,T,Hh,P,2)
        aw = attention_weights[:, :, :, l]    # (B,T,Hh,P)

        # Normierte -> Pixelkoordinaten (wie bei grid_sample Bilinear)
        x = loc[..., 0] * Wl - 0.5  # (B,T,Hh,P)
        y = loc[..., 1] * Hl - 0.5

        x0 = torch.floor(x).to(torch.long)
        y0 = torch.floor(y).to(torch.long)
        x1 = x0 + 1
        y1 = y0 + 1

        # Clamp in gültigen Bereich
        x0c = x0.clamp(0, Wl - 1)
        x1c = x1.clamp(0, Wl - 1)
        y0c = y0.clamp(0, Hl - 1)
        y1c = y1.clamp(0, Hl - 1)

        fx = (x - x0.to(x.dtype)).clamp(0, 1)
        fy = (y - y0.to(y.dtype)).clamp(0, 1)

        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy

        # Basisgewicht aus Attention * Relevanz
        base_w = (aw * Ro_broadcast).to(Rs.dtype)  # (B,T,Hh,P)

        def _scatter(nei_w: Tensor, xx: Tensor, yy: Tensor):
            idx = (yy * Wl + xx).view(B, -1) + base
            vals = (base_w * nei_w).view(B, -1)
            # Summe über T/Hh/P -> add in S-Dimension
            for b in range(B):
                Rs[b, idx[b], 0] += vals[b]

        _scatter(w00, x0c, y0c)
        _scatter(w10, x1c, y0c)
        _scatter(w01, x0c, y1c)
        _scatter(w11, x1c, y1c)

    return Rs


__all__ = [
    "residual_split",
    "layernorm_backshare",
    "value_path_split",
    "value_path_split_deform",
]
