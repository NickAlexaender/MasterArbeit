"""
Command line interface parser for calc_lrp.
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    """CLI-Parser für rückwärtskompatible Nutzung."""
    parser = argparse.ArgumentParser(description="LRP/Attribution für MaskDINO-Encoder/Decoder")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/1images",
        help="Ordner mit Eingabebildern (jpg/png)",
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=3,
        help="1-basierter Index des Encoder-/Decoder-Layers (z.B. 3)",
    )
    parser.add_argument(
        "--feature-index",
        type=int,
        default=214,
        help="Kanalindex (Feature) im gewählten Layer (z.B. 235)",
    )
    parser.add_argument(
        "--target-norm",
        type=str,
        default="sum1",
        choices=["sum1", "sumT", "none"],
        help="Norm der Zielrelevanz: sum1 (Summe=1), sumT (Summe=T), none (keine Norm)",
    )
    parser.add_argument(
        "--lrp-epsilon",
        type=float,
        default=1e-6,
        help="Epsilon-Stabilisator für ε/z+",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/lrp_result.csv",
        help="Pfad zur Ausgabedatei (CSV)",
    )
    parser.add_argument(
        "--which-module",
        type=str,
        default="encoder",
        choices=["encoder", "decoder"],
        help="Wähle Encoder oder Decoder für die LRP-Analyse",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="lrp",
        choices=["gradinput", "lrp"],
        help="Attributionsmethode: gradinput (Grad*Input am Layer-Eingang) oder lrp (LRP-Regeln)",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        default=None,
        help="Optionaler Pfad zu Modellgewichten (überschreibt Default)",
    )
    return parser.parse_args()
