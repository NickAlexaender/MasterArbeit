from __future__ import annotations

from dataclasses import dataclass

TARGET_TOKEN_IDX: int = 0
USE_SUBLAYER: str = "self_attn"
MEASUREMENT_POINT: str = "post_res"
RESIDUAL_SPLIT: str = "zsign"
LN_RULE: str = "conservative"
ATTN_QK_SHARE: float = 0.0
DETERMINISTIC: bool = True
SEED: int = 1234
SIGN_PRESERVING: bool = True

__all__ = [
    "TARGET_TOKEN_IDX",
    "USE_SUBLAYER",
    "MEASUREMENT_POINT",
    "RESIDUAL_SPLIT",
    "LN_RULE",
    "ATTN_QK_SHARE",
    "SIGN_PRESERVING",
    "DETERMINISTIC",
    "SEED",
]
