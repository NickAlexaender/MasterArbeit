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


# Wir ersetzen Module in-place im Modell - durch das Elternmodul.

def swap_module_inplace(
    parent_module: nn.Module,
    child_name: str,
    new_module: nn.Module,
) -> None:
    setattr(parent_module, child_name, new_module)


def _get_parent_and_name(root: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split('.')
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


# Wir wollen die Module durch ihre LRP-Versionen ersetzen

def swap_all_modules(
    model: nn.Module,
    source_type: Type[nn.Module],
    converter_fn,
    verbose: bool = False,
) -> int:
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

# Spezielle Funktion zum Ersetzen von MSDeformAttn-Modulen

def swap_msdeformattn_modules(
    model: nn.Module,
    verbose: bool = False,
) -> int:
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

# Komplette Vorbereitung. Ersetzt die spezifizierten Module-Typen durch ihre LRP-fähigen Versionen

def prepare_model_for_lrp(
    model: nn.Module,
    swap_linear: bool = False,
    swap_layernorm: bool = True,
    swap_mha: bool = True,
    swap_msdeform: bool = True,
    verbose: bool = False,
) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    
    if swap_layernorm:
        count = swap_all_modules(model, nn.LayerNorm, LRP_LayerNorm.from_layernorm, verbose)
        stats['LayerNorm'] = count
    
    if swap_mha:
        count = swap_all_modules(model, nn.MultiheadAttention, LRP_MultiheadAttention.from_mha, verbose)
        stats['MultiheadAttention'] = count
    
    if swap_msdeform:
        count = swap_msdeformattn_modules(model, verbose)
        stats['MSDeformAttn'] = count
    
    if swap_linear:
        count = swap_all_modules(model, nn.Linear, LRP_Linear.from_linear, verbose)
        stats['Linear'] = count
    
    return stats

# Wir wollen hier noch was hinzufügen, was uns die aktivierung oder deaktivierung von LRP ermöglicht

def set_lrp_mode(model: nn.Module, enabled: bool = True) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, LRPModuleMixin):
            module.is_lrp = enabled
            count += 1
    return count

# Am Ende wollen wir hier wieder alles clearn

def clear_all_activations(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, LRPModuleMixin):
            module.clear_activations()
            count += 1
    return count

# Wir wollen eine Funktion, die uns alle LRP-fähigen Module im Modell zurückgibt

def get_lrp_modules(model: nn.Module) -> Dict[str, LRPModuleMixin]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LRPModuleMixin)
    }


# Manage LRP

class LRPContext:
    
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

__all__ = [
    "swap_module_inplace",
    "swap_all_modules",
    "swap_msdeformattn_modules",
    "prepare_model_for_lrp",
    "set_lrp_mode",
    "clear_all_activations",
    "get_lrp_modules",
    "LRPContext",
]
