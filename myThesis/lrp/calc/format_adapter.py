"""
Adapter, um LRPEngine-Ergebnisse in das bestehende Exportformat zu überführen.

Aktuell nutzt die Attribution-Pipeline in core_analysis.py einen einfachen CSV-Export
mit den Spalten:
- prev_feature_idx, relevance, layer_index, layer_name, feature_index, epsilon,
  module_role, target_norm, method

Dieser Adapter stellt Helfer bereit, um Tensoren konsistent zu kanalisieren.
"""
from __future__ import annotations

from typing import Iterable, Dict, Any
import numpy as np
import torch
from torch import Tensor


def channel_vector_from_Rprev(R_prev: Tensor) -> Tensor:
    """Aggregiere R_prev (B,T,C) zu (C,) konsistent zur bisherigen Pipeline."""
    if R_prev.dim() == 4:  # (B,C,H,W)
        B, C, H, W = R_prev.shape
        return R_prev.sum(dim=(0, 2, 3)).detach().cpu()
    if R_prev.dim() == 3:  # (B,T,C)
        return R_prev.sum(dim=(0, 1)).detach().cpu()
    if R_prev.dim() == 2:  # (B,C)
        return R_prev.sum(dim=0).detach().cpu()
    raise ValueError(f"Unexpected shape for R_prev: {tuple(R_prev.shape)}")


def to_existing_export_rows(R_prev: Tensor) -> Iterable[Dict[str, Any]]:
    """Liefere Iterator über (prev_feature_idx, relevance) konsistent zur Reihenfolge.

    Hinweis: Der vollständige CSV-Export (Header/Sorting) erfolgt wie gehabt
    in core_analysis.py. Diese Funktion bietet nur die Paare als Dict.
    """
    vec = channel_vector_from_Rprev(R_prev)
    for i, v in enumerate(vec.tolist()):
        yield {"prev_feature_idx": i, "relevance": float(v)}


__all__ = ["channel_vector_from_Rprev", "to_existing_export_rows"]
