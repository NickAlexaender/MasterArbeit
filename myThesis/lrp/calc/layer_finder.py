"""
Helpers to locate plausible encoder/decoder layers in the MaskDINO model.
"""
from __future__ import annotations

from typing import List, Tuple
import torch.nn as nn


def list_encoder_like_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Finde plausible Encoder-Layer-Module innerhalb MaskDINO.

    Strategie:
    - Bevorzuge Module, deren Pfadnamen Schlüsselwörter enthalten: 'encoder', 'transformer', 'pixel_decoder'
    - Filtere auf Blöcke, die selbst Unter-Module enthalten (Layer-Container), z.B. *EncoderLayer*
    - Ergebnisliste ist in DFS-Reihenfolge der Namen, stabil für Indexwahl
    """
    keywords = ("encoder", "transformer", "pixel_decoder")
    candidates: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        lname = name.lower()
        if any(k in lname for k in keywords):
            submods = list(module.children())
            if not submods:
                continue
            subnames = {type(m).__name__.lower() for m in submods}
            if any(s in subnames for s in ["multiheadattention", "msdeformattn", "selfattention", "selfattn"]):
                candidates.append((name, module))
            else:
                for cidx, child in enumerate(submods):
                    cname = f"{name}.{cidx}"
                    csub = list(child.children())
                    csubnames = {type(m).__name__.lower() for m in csub}
                    if any(s in csubnames for s in ["multiheadattention", "msdeformattn", "selfattention", "selfattn"]):
                        candidates.append((cname, child))
    uniq: List[Tuple[str, nn.Module]] = []
    seen = set()
    for n, m in candidates:
        if id(m) not in seen:
            uniq.append((n, m))
            seen.add(id(m))
    return uniq


def list_decoder_like_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Finde plausible Decoder-Layer-Module innerhalb MaskDINO/Detectron2.

    Heuristik wie Encoder, aber Keywords auf Decoder fokussiert.
    """
    keywords = ("decoder", "transformer_decoder")
    candidates: List[Tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        lname = name.lower()
        if any(k in lname for k in keywords):
            submods = list(module.children())
            if not submods:
                continue
            subnames = {type(m).__name__.lower() for m in submods}
            if any(s in subnames for s in ["multiheadattention", "msdeformattn", "selfattention", "selfattn"]):
                candidates.append((name, module))
            else:
                for cidx, child in enumerate(submods):
                    cname = f"{name}.{cidx}"
                    csub = list(child.children())
                    csubnames = {type(m).__name__.lower() for m in csub}
                    if any(s in csubnames for s in ["multiheadattention", "msdeformattn", "selfattention", "selfattn"]):
                        candidates.append((cname, child))
    uniq: List[Tuple[str, nn.Module]] = []
    seen = set()
    for n, m in candidates:
        if id(m) not in seen:
            uniq.append((n, m))
            seen.add(id(m))
    return uniq
