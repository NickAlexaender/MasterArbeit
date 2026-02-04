from __future__ import annotations
from .compat import *  # Pillow/NumPy monkey patches - MUSS vor anderen Imports kommen
from .lrp_structs import (
    LRPResult,
    LayerRelevance,
)
from .lrp_controller import LRPController
from .lrp_analysis import (
    LRPAnalysisContext,
    run_lrp_analysis,
    run_lrp_batch,
)
from .lrp_propagators import (
    propagate_layer,
    propagate_linear,
    propagate_layernorm,
    propagate_multihead_attention,
    propagate_msdeformattn,
    propagate_residual,
)
from .model_graph_wrapper import (
    ModelGraph,
    LayerNode,
    LayerType,
)
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
from .tensor_ops import (
    safe_divide,
    rearrange_activations,
    build_target_relevance,
    aggregate_channel_relevance,
)
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
