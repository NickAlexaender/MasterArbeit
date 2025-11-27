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
    # Wichtig: Enthält ein Modul zwar "pixel_decoder" im Namen, aber darunter
    #        explizit "encoder" (z. B. "...pixel_decoder.transformer.encoder.layers.X"),
    #        dann ist es KEIN Decoder-Layer für die 300 Objekt-Queries.
    #        Wir filtern solche Kandidaten konsequent heraus und bevorzugen explizite
    #        Pfade mit ".decoder.".
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
                # Top-Level-Kandidat nur akzeptieren, wenn der Name nicht offensichtlich Encoder ist
                if ".encoder." not in lname:
                    candidates.append((name, module))
            else:
                for cidx, child in enumerate(submods):
                    cname = f"{name}.{cidx}"
                    csub = list(child.children())
                    csubnames = {type(m).__name__.lower() for m in csub}
                    if any(s in csubnames for s in ["multiheadattention", "msdeformattn", "selfattention", "selfattn"]):
                        clname = cname.lower()
                        # Bevorzuge echte Decoder-Pfade und schließe Encoder-Pfade aus
                        if ".encoder." in clname:
                            continue
                        # Wenn möglich explizit Decoder referenzieren
                        if (".decoder." in clname) or ("transformer_decoder" in clname) or ("query" in clname):
                            candidates.append((cname, child))
                        # Falls der Parent lediglich "pixel_decoder" enthält, aber das Child
                        # weder ".decoder." noch ".encoder." enthält, nehmen wir es vorsichtig auf,
                        # um zumindest Transformer-Decoder-Layer zu erwischen.
                        elif ("pixel_decoder" in lname) and (".encoder." not in clname):
                            candidates.append((cname, child))
    uniq: List[Tuple[str, nn.Module]] = []
    seen = set()
    for n, m in candidates:
        if id(m) not in seen:
            uniq.append((n, m))
            seen.add(id(m))
    # Sortiere Kandidaten so, dass explizite ".decoder."-Pfade zuerst kommen
    def _score(name_mod: Tuple[str, nn.Module]) -> int:
        n = name_mod[0].lower()
        score = 0
        if ".decoder." in n:
            score += 2
        if "transformer_decoder" in n:
            score += 2
        if "pixel_decoder" in n:
            score += 1
        # Encoder-Pfade ganz nach hinten (sollte durch Filter ohnehin nicht enthalten sein)
        if ".encoder." in n:
            score -= 10
        return -score  # kleiner ist besser

    uniq.sort(key=_score)
    return uniq
