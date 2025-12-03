"""
LRP-fähige Module, die PyTorch-Originale ersetzen und Aktivierungen intern speichern.

HINWEIS: Diese Datei ist ein Kompatibilitäts-Wrapper und re-exportiert alle
Komponenten aus den aufgeteilten Modulen:
    - lrp_param_base.py: LRPActivations, LRPModuleMixin
    - lrp_param_modules.py: LRP_Linear, LRP_LayerNorm, LRP_MultiheadAttention, LRP_MSDeformAttn
    - lrp_param_utils.py: Swap-Utilities und LRPContext

Für neue Imports empfohlen:
    from .lrp_param_base import LRPActivations, LRPModuleMixin
    from .lrp_param_modules import LRP_Linear, LRP_LayerNorm, ...
    from .lrp_param_utils import prepare_model_for_lrp, LRPContext, ...
"""
from __future__ import annotations

# Re-export aus lrp_param_base
from .lrp_param_base import (
    LRPActivations,
    LRPModuleMixin,
)

# Re-export aus lrp_param_modules
from .lrp_param_modules import (
    LRP_Linear,
    LRP_LayerNorm,
    LRP_MultiheadAttention,
    LRP_MSDeformAttn,
)

# Re-export aus lrp_param_utils
from .lrp_param_utils import (
    swap_module_inplace,
    swap_all_modules,
    swap_msdeformattn_modules,
    prepare_model_for_lrp,
    set_lrp_mode,
    clear_all_activations,
    get_lrp_modules,
    LRPContext,
)


# =============================================================================
# Exports
# =============================================================================


__all__ = [
    # Datenstrukturen (aus lrp_param_base)
    "LRPActivations",
    "LRPModuleMixin",
    
    # LRP-Module (aus lrp_param_modules)
    "LRP_Linear",
    "LRP_LayerNorm",
    "LRP_MultiheadAttention",
    "LRP_MSDeformAttn",
    
    # Swap-Utilities (aus lrp_param_utils)
    "swap_module_inplace",
    "swap_all_modules",
    "swap_msdeformattn_modules",
    "prepare_model_for_lrp",
    
    # LRP-Modus-Verwaltung (aus lrp_param_utils)
    "set_lrp_mode",
    "clear_all_activations",
    "get_lrp_modules",
    "LRPContext",
]
