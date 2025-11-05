"""
Tensor utility functions used across the LRP analysis code.
"""
from __future__ import annotations

from typing import Any, Iterable, List, Tuple
import warnings
import torch
import torch.nn as nn
from torch import Tensor


def _flatten(obj: Any) -> List[Any]:
    """Flacht verschachtelte Strukturen (tuple/list) ab, behält Reihenfolge."""
    if isinstance(obj, (list, tuple)):
        res: List[Any] = []
        for it in obj:
            res.extend(_flatten(it))
        return res
    return [obj]


def _iter_tensors(obj: Any) -> Iterable[Tensor]:
    for x in _flatten(obj):
        if isinstance(x, torch.Tensor):
            yield x


def _first_tensor(obj: Any) -> Tensor:
    for t in _iter_tensors(obj):
        return t
    raise ValueError("Keine Tensor-Ausgabe im Layer gefunden")


def _to_BTC(t: Tensor) -> Tensor:
    """Bringe Tensor robust in Form (B, T, C).

    Regeln:
    - 4D: (B, C, H, W) -> (B, H*W, C)
    - 3D: Unverändert zurückgeben (erwartet (B, T, C)). Keine heuristische Permutation.
      Falls tatsächlich (T,B,C) vorliegt, bitte explizit vor dem Aufruf transponieren.
      Bei Verdacht (erste Achse deutlich größer als zweite) wird eine Warnung ausgegeben.
    - 2D: (B, C) -> (B, 1, C)
    """
    if t.dim() == 4:  # (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = t.shape
        return t.permute(0, 2, 3, 1).reshape(B, H * W, C)
    if t.dim() == 3:
        if t.shape[0] > t.shape[1]:
            warnings.warn(
                "_to_BTC: 3D-Input wird unverändert angenommen (B,T,C). "
                "Die erste Achse ist größer als die zweite – prüfen, ob (T,B,C) vorliegt und ggf. explizit transponieren.",
                stacklevel=2,
            )
        return t
    if t.dim() == 2:
        return t.unsqueeze(1)
    raise ValueError(f"Unerwartete Tensorform: {tuple(t.shape)}")


def aggregate_channel_relevance(R_in: Tensor) -> Tensor:
    """Aggregiere Eingangsrelevanz zu einem Vektor (C_in,)."""
    if R_in.dim() == 4:  # (B, C, H, W)
        return R_in.sum(dim=(0, 2, 3)).detach().cpu()
    if R_in.dim() == 3:  # (B, L, C)
        return R_in.sum(dim=(0, 1)).detach().cpu()
    if R_in.dim() == 2:  # (B, C)
        return R_in.sum(dim=0).detach().cpu()
    raise ValueError(f"Unerwartete R_in-Form: {tuple(R_in.shape)}")


def build_target_relevance(
    layer_output: Tensor,
    feature_index: int,
    token_reduce: str,
    target_norm: str = "sum1",
    index_axis: str = "channel",
) -> Tensor:
    """Erzeuge eine Start-Relevanz R_out ohne Gradienten.

    index_axis:
    - "channel": feature_index adressiert den Kanal (C-Achse)
    - "token":   feature_index adressiert den Token/Query (T-Achse)

    token_reduce wirkt nur bei index_axis="channel" und steuert die Verteilung über Tokens.
    """
    y = _to_BTC(layer_output)  # (B, T, C)
    B, T, C = y.shape

    base = torch.zeros_like(y)

    if index_axis == "channel":
        if feature_index < 0 or feature_index >= C:
            raise IndexError(
                f"feature_index {feature_index} außerhalb [0, {C-1}] (axis=channel)"
            )
        feat = y[..., feature_index]
        if token_reduce == "mean":
            w = torch.ones_like(feat)
        elif token_reduce == "max":
            # Echte Max-Auswahl: One-Hot pro Batch auf den Token mit maximaler Aktivierung
            idx = torch.argmax(feat, dim=1, keepdim=True)  # (B,1)
            w = torch.zeros_like(feat).scatter_(1, idx, 1.0)
        else:
            raise ValueError("token_reduce muss 'mean' oder 'max' sein")
        # Normierung über alle Tokens/Batch
        s = w.sum().clamp_min(1e-12)
        if target_norm == "sum1":
            w = w / s
        elif target_norm == "sumT":
            w = w / s * float(T)
        elif target_norm == "none":
            pass
        else:
            raise ValueError("target_norm muss 'sum1', 'sumT' oder 'none' sein")
        base[..., feature_index] = w
    elif index_axis == "token":
        if feature_index < 0 or feature_index >= T:
            raise IndexError(
                f"feature_index {feature_index} außerhalb [0, {T-1}] (axis=token)"
            )
        # Verteile Gewicht gleichmäßig über Kanäle für den gewählten Token
        w_tok = torch.ones_like(y[:, feature_index, :])  # (B, C)
        s = w_tok.sum().clamp_min(1e-12)
        if target_norm == "sum1":
            w_tok = w_tok / s
        elif target_norm == "sumT":
            # für Token-Modus interpretieren wir 'T' als C-Anzahl
            w_tok = w_tok / s * float(C)
        elif target_norm == "none":
            pass
        else:
            raise ValueError("target_norm muss 'sum1', 'sumT' oder 'none' sein")
        base[:, feature_index, :] = w_tok
    else:
        raise ValueError("index_axis muss 'channel' oder 'token' sein")

    # Form zurück wie layer_output
    if layer_output.dim() == 4:
        if index_axis == "token":
            raise ValueError("index_axis='token' wird für 4D-Outputs nicht unterstützt")
        # (B,T,C)->(B,C,H,W)
        B2, C2, H, W = layer_output.shape
        assert B2 == B and C2 == C and H * W == T
        base = base.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return base
