"""
LRP-Analyse-Module für MaskDINO.

Dieses Paket enthält alle Module für Layer-wise Relevance Propagation (LRP):

- lrp_controller: Zentraler Controller für LRP-Analyse
- param_patcher: LRP-fähige Module, die Aktivierungen speichern
- model_graph_wrapper: Linearisiert Modellstruktur für LRP
- lrp_rules_standard: Basis-LRP-Regeln (ε, γ, α-β)
- lrp_rules_attention: LRP für Self/Cross-Attention
- lrp_rules_deformable: LRP für MSDeformAttn
- tensor_ops: Tensor-Utilities für numerische Stabilität
- config: Konfigurationskonstanten
- config_build: Model-Konfiguration
- io_utils: I/O-Hilfsfunktionen
"""
from __future__ import annotations

# WICHTIG: Kompatibilitäts-Patches müssen zuerst geladen werden
from .compat import *  # Pillow/NumPy monkey patches - MUSS vor anderen Imports kommen

# Datenstrukturen (aus lrp_structs.py)
from .lrp_structs import (
    LRPResult,
    LayerRelevance,
)

# Hauptklasse (aus lrp_controller.py)
from .lrp_controller import LRPController

# High-Level Tools (aus lrp_analysis.py)
from .lrp_analysis import (
    LRPAnalysisContext,
    run_lrp_analysis,
    run_lrp_batch,
)

# Propagatoren (aus lrp_propagators.py)
from .lrp_propagators import (
    propagate_layer,
    propagate_linear,
    propagate_layernorm,
    propagate_multihead_attention,
    propagate_msdeformattn,
    propagate_residual,
)

# Model Graph
from .model_graph_wrapper import (
    ModelGraph,
    LayerNode,
    LayerType,
)

# LRP-fähige Module
from .param_patcher import (
    LRPActivations,
    LRPModuleMixin,
    LRPContext,
    LRP_Linear,
    LRP_LayerNorm,
    LRP_MultiheadAttention,
    LRP_MSDeformAttn,
    prepare_model_for_lrp,
    set_lrp_mode,
    clear_all_activations,
    get_lrp_modules,
)

# LRP-Regeln
from .lrp_rules_standard import (
    lrp_epsilon_rule,
    lrp_gamma_rule,
    lrp_alpha_beta_rule,
    lrp_linear,
    residual_split,
    layernorm_lrp,
    layernorm_backshare,
)

from .lrp_rules_attention import (
    AttnCache,
    lrp_attention_value_path,
    lrp_attention_to_weights,
    lrp_softmax,
    lrp_attention_qk_path,
    lrp_full_attention,
)

from .lrp_rules_deformable import (
    msdeform_attn_lrp,
    msdeform_attn_lrp_with_value,
    deform_value_path_lrp,
    bilinear_splat_relevance,
    compute_bilinear_weights,
    compute_pixel_relevance_map,
    compute_multiscale_relevance_map,
    attach_msdeformattn_capture,
)

# Tensor-Utilities
from .tensor_ops import (
    safe_divide,
    rearrange_activations,
    build_target_relevance,
    aggregate_channel_relevance,
)

# Konfiguration
from .config import (
    TARGET_TOKEN_IDX,
    USE_SUBLAYER,
    MEASUREMENT_POINT,
    RESIDUAL_SPLIT,
    LN_RULE,
    ATTN_QK_SHARE,
    SIGN_PRESERVING,
    DETERMINISTIC,
    SEED,
)

from .config_build import (
    build_cfg_for_inference,
    DEFAULT_WEIGHTS,
)

from .io_utils import collect_images


__all__ = [
    # Hauptklassen
    "LRPController",
    "LRPResult",
    "LayerRelevance",
    "LRPAnalysisContext",
    "run_lrp_analysis",
    "run_lrp_batch",
    
    # Propagatoren
    "propagate_layer",
    "propagate_linear",
    "propagate_layernorm",
    "propagate_multihead_attention",
    "propagate_msdeformattn",
    "propagate_residual",
    
    # Model Graph
    "ModelGraph",
    "LayerNode",
    "LayerType",
    
    # LRP-Module
    "LRPActivations",
    "LRPModuleMixin",
    "LRPContext",
    "LRP_Linear",
    "LRP_LayerNorm",
    "LRP_MultiheadAttention",
    "LRP_MSDeformAttn",
    "prepare_model_for_lrp",
    "set_lrp_mode",
    "clear_all_activations",
    "get_lrp_modules",
    
    # LRP-Regeln Standard
    "lrp_epsilon_rule",
    "lrp_gamma_rule",
    "lrp_alpha_beta_rule",
    "lrp_linear",
    "residual_split",
    "layernorm_lrp",
    "layernorm_backshare",
    
    # LRP-Regeln Attention
    "AttnCache",
    "lrp_attention_value_path",
    "lrp_attention_to_weights",
    "lrp_softmax",
    "lrp_attention_qk_path",
    "lrp_full_attention",
    
    # LRP-Regeln Deformable
    "msdeform_attn_lrp",
    "msdeform_attn_lrp_with_value",
    "deform_value_path_lrp",
    "bilinear_splat_relevance",
    "compute_bilinear_weights",
    "compute_pixel_relevance_map",
    "compute_multiscale_relevance_map",
    "attach_msdeformattn_capture",
    
    # Tensor-Utilities
    "safe_divide",
    "rearrange_activations",
    "build_target_relevance",
    "aggregate_channel_relevance",
    
    # Konfiguration
    "TARGET_TOKEN_IDX",
    "USE_SUBLAYER",
    "MEASUREMENT_POINT",
    "RESIDUAL_SPLIT",
    "LN_RULE",
    "ATTN_QK_SHARE",
    "SIGN_PRESERVING",
    "DETERMINISTIC",
    "SEED",
    "build_cfg_for_inference",
    "DEFAULT_WEIGHTS",
    "collect_images",
]
