"""
Konstanten und Defaults für die lokale LRP (L-1 -> L) Analyse.
Die Werte sind konservativ und deterministisch, ohne Änderung der CLI-Signatur.
"""
from __future__ import annotations

from dataclasses import dataclass

# Feste, rückwärtskompatible Defaults (kein Eingriff in main/CLI nötig)
TARGET_TOKEN_IDX: int = 0              # Ziel-Token t* (Decoder-Query bzw. Encoder-Token)
USE_SUBLAYER: str = "self_attn"        # {"self_attn", "cross_attn", "ffn"}
MEASUREMENT_POINT: str = "post_res"    # {"pre_res", "post_res"}
RESIDUAL_SPLIT: str = "zsign"          # {"energy", "dotpos", "zsign"}
LN_RULE: str = "zsign"                 # {"zsign", "abs-grad-xmu", "xmu"}
ATTN_QK_SHARE: float = 0.0             # ρ-Anteil für Q/K (0.0 ⇒ nur Value)
DETERMINISTIC: bool = True
SEED: int = 1234
SIGN_PRESERVING: bool = True           # Vorzeichen über alle Regeln bewahren

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
