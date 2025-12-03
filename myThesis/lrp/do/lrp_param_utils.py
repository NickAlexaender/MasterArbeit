"""
LRP-Management-Utilities: Modul-Swap und LRP-Modus-Verwaltung.

Dieses Modul enthält die "Admin"-Funktionen für LRP:
    - swap_module_inplace: Ersetzt ein einzelnes Modul in-place
    - swap_all_modules: Ersetzt alle Module eines bestimmten Typs im Modell
    - swap_msdeformattn_modules: Spezialisierte Funktion für MSDeformAttn
    - prepare_model_for_lrp: Vollständige Modellvorbereitung für LRP-Analyse
    - set_lrp_mode: Aktiviert/deaktiviert LRP-Modus für alle Module
    - clear_all_activations: Löscht alle gespeicherten Aktivierungen
    - get_lrp_modules: Gibt alle LRP-fähigen Module zurück
    - LRPContext: Context Manager für LRP-Analyse
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type
import warnings

import torch.nn as nn

from .lrp_param_base import LRPModuleMixin
from .lrp_param_modules import (
    LRP_Linear,
    LRP_LayerNorm,
    LRP_MultiheadAttention,
    LRP_MSDeformAttn,
)


# =============================================================================
# Modul-Swap-Utilities
# =============================================================================


def swap_module_inplace(
    parent_module: nn.Module,
    child_name: str,
    new_module: nn.Module,
) -> None:
    """Ersetzt ein Kind-Modul in-place.
    
    Args:
        parent_module: Das Eltern-Modul, das das zu ersetzende Modul enthält
        child_name: Name des zu ersetzenden Kind-Moduls
        new_module: Das neue Modul, das eingesetzt werden soll
    
    Example:
        >>> model = SomeModel()
        >>> new_linear = LRP_Linear.from_linear(model.fc)
        >>> swap_module_inplace(model, 'fc', new_linear)
    """
    setattr(parent_module, child_name, new_module)


def _get_parent_and_name(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    """Findet das Elternmodul und den lokalen Namen eines Moduls."""
    parts = full_name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def swap_all_modules(
    model: nn.Module,
    source_type: Type[nn.Module],
    converter_fn,
    verbose: bool = False,
) -> int:
    """Ersetzt alle Module eines bestimmten Typs durch LRP-Versionen.
    
    Args:
        model: Das zu modifizierende Modell
        source_type: Der Typ der zu ersetzenden Module (z.B. nn.Linear)
        converter_fn: Funktion, die das Original-Modul konvertiert
        verbose: Wenn True, werden Ersetzungen geloggt
    
    Returns:
        Anzahl der ersetzten Module
    
    Example:
        >>> count = swap_all_modules(model, nn.Linear, LRP_Linear.from_linear)
        >>> print(f"Replaced {count} Linear layers")
    """
    # Sammle alle zu ersetzenden Module (um während Iteration zu ändern)
    to_replace: List[Tuple[str, nn.Module]] = []
    
    for name, module in model.named_modules():
        if type(module) == source_type:  # Exakter Typ-Check, keine Subklassen
            to_replace.append((name, module))
    
    count = 0
    for name, module in to_replace:
        try:
            new_module = converter_fn(module)
            parent, child_name = _get_parent_and_name(model, name)
            swap_module_inplace(parent, child_name, new_module)
            count += 1
            if verbose:
                print(f"Replaced {name}: {type(module).__name__} -> {type(new_module).__name__}")
        except Exception as e:
            if verbose:
                warnings.warn(f"Failed to replace {name}: {e}")
    
    return count


def swap_msdeformattn_modules(
    model: nn.Module,
    verbose: bool = False,
) -> int:
    """Ersetzt alle MSDeformAttn-Module durch LRP-Versionen.
    
    Diese Funktion sucht nach MSDeformAttn-Modulen verschiedener Implementierungen
    (MaskDINO, Deformable DETR, etc.) und ersetzt sie durch LRP_MSDeformAttn.
    
    Args:
        model: Das zu modifizierende Modell
        verbose: Wenn True, werden Ersetzungen geloggt
    
    Returns:
        Anzahl der ersetzten Module
    """
    # MSDeformAttn-Klassenkandidaten
    msdeform_candidates = [
        "maskdino.modeling.pixel_decoder.ops.modules.ms_deform_attn.MSDeformAttn",
        "models.ops.modules.ms_deform_attn.MSDeformAttn",
        "detectron2.layers.deformable_attention.MSDeformAttn",
    ]
    
    msdeform_cls: Optional[type] = None
    for fq in msdeform_candidates:
        try:
            mod_path, cls_name = fq.rsplit('.', 1)
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if isinstance(cls, type):
                msdeform_cls = cls
                break
        except Exception:
            continue
    
    if msdeform_cls is None:
        if verbose:
            warnings.warn("No MSDeformAttn class found to replace")
        return 0
    
    return swap_all_modules(
        model,
        msdeform_cls,
        LRP_MSDeformAttn.from_msdeformattn,
        verbose=verbose,
    )


def prepare_model_for_lrp(
    model: nn.Module,
    swap_linear: bool = False,
    swap_layernorm: bool = True,
    swap_mha: bool = True,
    swap_msdeform: bool = True,
    verbose: bool = False,
) -> Dict[str, int]:
    """Bereitet ein Modell vollständig für LRP-Analyse vor.
    
    Ersetzt die spezifizierten Module-Typen durch ihre LRP-fähigen Versionen.
    
    Args:
        model: Das zu modifizierende Modell
        swap_linear: Wenn True, ersetze alle nn.Linear (kann sehr viele sein!)
        swap_layernorm: Wenn True, ersetze alle nn.LayerNorm
        swap_mha: Wenn True, ersetze alle nn.MultiheadAttention
        swap_msdeform: Wenn True, ersetze alle MSDeformAttn
        verbose: Wenn True, werden Ersetzungen geloggt
    
    Returns:
        Dictionary mit Anzahl der Ersetzungen pro Modul-Typ
    
    Example:
        >>> stats = prepare_model_for_lrp(model, verbose=True)
        >>> print(stats)  # {'LayerNorm': 24, 'MultiheadAttention': 12, 'MSDeformAttn': 6}
    """
    stats: Dict[str, int] = {}
    
    if swap_linear:
        count = swap_all_modules(model, nn.Linear, LRP_Linear.from_linear, verbose)
        stats['Linear'] = count
    
    if swap_layernorm:
        count = swap_all_modules(model, nn.LayerNorm, LRP_LayerNorm.from_layernorm, verbose)
        stats['LayerNorm'] = count
    
    if swap_mha:
        count = swap_all_modules(model, nn.MultiheadAttention, LRP_MultiheadAttention.from_mha, verbose)
        stats['MultiheadAttention'] = count
    
    if swap_msdeform:
        count = swap_msdeformattn_modules(model, verbose)
        stats['MSDeformAttn'] = count
    
    return stats


# =============================================================================
# LRP-Modus-Verwaltung
# =============================================================================


def set_lrp_mode(model: nn.Module, enabled: bool = True) -> int:
    """Aktiviert oder deaktiviert den LRP-Modus für alle LRP-Module im Modell.
    
    Args:
        model: Das Modell mit LRP-fähigen Modulen
        enabled: True um LRP zu aktivieren, False zum Deaktivieren
    
    Returns:
        Anzahl der Module, deren Modus geändert wurde
    
    Example:
        >>> set_lrp_mode(model, True)  # Aktiviere LRP für Forward-Pass
        >>> output = model(input)
        >>> # Aktivierungen sind jetzt in module.activations gespeichert
        >>> set_lrp_mode(model, False)  # Deaktiviere für normale Inferenz
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LRPModuleMixin):
            module.is_lrp = enabled
            count += 1
    return count


def clear_all_activations(model: nn.Module) -> int:
    """Löscht alle gespeicherten Aktivierungen in LRP-Modulen.
    
    Args:
        model: Das Modell mit LRP-fähigen Modulen
    
    Returns:
        Anzahl der geleerten Module
    """
    count = 0
    for module in model.modules():
        if isinstance(module, LRPModuleMixin):
            module.clear_activations()
            count += 1
    return count


def get_lrp_modules(model: nn.Module) -> Dict[str, LRPModuleMixin]:
    """Gibt alle LRP-fähigen Module im Modell zurück.
    
    Args:
        model: Das Modell
    
    Returns:
        Dictionary mit Modul-Namen und LRP-Modulen
    """
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LRPModuleMixin)
    }


# =============================================================================
# Context Manager für LRP-Analyse
# =============================================================================


class LRPContext:
    """Context Manager für LRP-Analyse.
    
    Aktiviert automatisch den LRP-Modus beim Betreten und deaktiviert ihn
    beim Verlassen des Kontexts. Bereinigt auch alle Aktivierungen.
    
    Example:
        >>> with LRPContext(model):
        ...     output = model(input)
        ...     # Aktivierungen sind jetzt verfügbar
        ...     for name, module in get_lrp_modules(model).items():
        ...         print(f"{name}: {module.activations.to_dict().keys()}")
        >>> # LRP-Modus automatisch deaktiviert, Aktivierungen gelöscht
    """
    
    def __init__(self, model: nn.Module, clear_on_exit: bool = True):
        self.model = model
        self.clear_on_exit = clear_on_exit
    
    def __enter__(self) -> 'LRPContext':
        set_lrp_mode(self.model, True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_lrp_mode(self.model, False)
        if self.clear_on_exit:
            clear_all_activations(self.model)
        return False


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Swap-Utilities
    "swap_module_inplace",
    "swap_all_modules",
    "swap_msdeformattn_modules",
    "prepare_model_for_lrp",
    
    # LRP-Modus-Verwaltung
    "set_lrp_mode",
    "clear_all_activations",
    "get_lrp_modules",
    "LRPContext",
]
