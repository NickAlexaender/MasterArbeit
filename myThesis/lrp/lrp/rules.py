"""
LRP-Regeln für Residual, LayerNorm und Attention-Value-Pfad.
Hinweis: Diese Implementierungen sind robuste, numerisch stabile Varianten
und dienen als Bausteine für die LRPEngine.
"""
from __future__ import annotations

from typing import Tuple
import torch
from torch import Tensor


def residual_split(x: Tensor, Fx: Tensor, Ry: Tensor, mode: str = "energy") -> Tuple[Tensor, Tensor]:
    """Teile Relevanz Ry zwischen Skip-Pfad x und Transformationspfad F(x) auf.

    - energy: proportional zu ||x||^2 vs. ||F(x)||^2 je Token.
    - dotpos: proportional zu positiven Skalarprodukten mit y = x + F(x).
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
    else:
        raise ValueError("unknown residual split")


def layernorm_backshare(x: Tensor, gamma: Tensor, beta: Tensor, Ry: Tensor, rule: str = "abs-grad-xmu") -> Tensor:
    """Verteile Relevanz über LayerNorm-Eingang x.

    Zwei Varianten:
    - "xmu": rein proportional zu |x-μ| pro Token
    - "abs-grad-xmu": Absolutgradient gewichtet |x-μ| (benötigt Autograd-Kontext)
    """
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if Ry.dim() == 2:
        Ry = Ry.unsqueeze(1)

    if rule == "xmu":
        xm = (x - x.mean(dim=-1, keepdim=True)).abs()
        w = xm / (xm.sum(dim=-1, keepdim=True) + 1e-12)
        return w * Ry
    else:  # "abs-grad-xmu"
        xmu = x - x.mean(dim=-1, keepdim=True)
        sigma = xmu.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        y = gamma * (xmu / sigma) + beta
        g = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=Ry, retain_graph=True, allow_unused=True)[0]
        if g is None:
            g = torch.ones_like(x)
        g = g.abs()
        w = (g * xmu.abs())
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
        return w * Ry


def value_path_split(
    Ro_j: Tensor,
    attn_weights: Tensor,
    Vproj: Tensor,
    W_O_j: Tensor,
    use_abs: bool = True,
    ) -> Tensor:
    """Verteilt Relevanz von Ausgabetoken t auf Quell-Token s über den Value-Pfad.

    Parameter:
    - Ro_j: (B,T,1) Relevanz für Zielkanal j pro Ziel-Token t
    - attn_weights: (B,H,T,S)
    - Vproj: (B,H,S,Dh)  Value-Projektionen je Head
    - W_O_j: (H,Dh,1)    Spaltenvektor der Output-Projektion für Kanal j
    - use_abs: Wenn True, werden Beiträge per |.| gewichtet (robust, vorzeichenignorierend).
               Wenn False, werden quadratische Beiträge verwendet (vermeidet Negative, erhält relative Stärke ohne Absolutbetrag).

    Rückgabe:
    - Rs: (B,S,1) Relevanz auf die Quell-Token s aggregiert über t
    """
    B, H, T, S = attn_weights.shape
    contrib = []
    for h in range(H):
        vs = Vproj[:, h]              # (B,S,Dh)
        wo = W_O_j[h]                 # (Dh,1)
        score = torch.einsum("bsd,dk->bsk", vs, wo)  # (B,S,1)
        # Bring score und attention auf gleiche Achsenreihenfolge (B,T,S,1)
        score = score.unsqueeze(1).expand(-1, T, -1, -1)   # (B,T,S,1)
        a = attn_weights[:, h]                 # (B,T,S)
        a = a.unsqueeze(-1)                    # (B,T,S,1)
        cs = a * score                         # (B,T,S,1)
        contrib.append(cs)
    C = torch.stack(contrib, dim=1).sum(dim=1)   # (B,T,S,1)
    if use_abs:
        C = C.abs()                              # SumAbs1-Variante (robust)
    else:
        C = C.pow(2)                             # quadratische Beiträge (nicht-negativ, vorzeichenfrei)
    Csum = C.sum(dim=2, keepdim=True) + 1e-12
    w = C / Csum                                 # Gewichte über S je (B,T)
    Rs = (w * Ro_j).sum(dim=1)                   # (B,S,1)
    return Rs


__all__ = [
    "residual_split",
    "layernorm_backshare",
    "value_path_split",
]
