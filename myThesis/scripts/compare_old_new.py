"""
Regression-Vergleich: alter (referenzierter) In-Memory-Flow vs. neue Streaming-Pipeline.

Voraussetzung: Ein kleines Test-Fixture liegt in encoder_out_dir (z. B. 2 Layer, 3 Features, ~5 Bilder).
Das Skript schreibt zwei Export-Roots und vergleicht die CSVs zeilenweise (np.allclose atol=1e-6).

Hinweis: Die "Referenz" (alt) wird hier lokal in diesem Skript nachgebildet,
indem alle Heatmaps eines Features im Speicher gehalten werden (nur für kleine Fixtures nutzen!).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Projektpfad einhängen
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from encoder.config import DEFAULT_COMBINE
from encoder.export_utils import resolve_export_root
from encoder.iou_core import compute_iou_from_heatmap, generate_heatmap_only
from encoder.iterators import iter_iou_inputs
from encoder.pipeline import main_export_network_dissection


@dataclass
class _Agg:
    hms: List[np.ndarray]
    msks: List[np.ndarray]


def _run_reference(encoder_out_dir: str, export_root: str, percentile: float = 90.0) -> None:
    # Sammle In-Memory pro Feature
    feats: Dict[Tuple[int, int], _Agg] = {}
    for item in iter_iou_inputs(encoder_out_dir=encoder_out_dir, mask_dir=None):
        key = (item.layer_idx, item.feature_idx)
        agg = feats.get(key)
        if agg is None:
            agg = _Agg(hms=[], msks=[])
            feats[key] = agg
        hm = generate_heatmap_only(item, combine=DEFAULT_COMBINE)
        agg.hms.append(hm)
        agg.msks.append(item.mask_input)

    # Schreibe einfache CSV-Struktur pro Layer
    rows_by_layer: Dict[int, List[Dict[str, object]]] = {}
    for (lidx, fidx), ag in feats.items():
        if not ag.hms:
            continue
        all_vals = np.concatenate([x.ravel() for x in ag.hms]).astype(np.float32, copy=False)
        thr = float(np.percentile(all_vals, float(percentile)))
        ious = [compute_iou_from_heatmap(h, m, thr) for h, m in zip(ag.hms, ag.msks)]
        miou = float(np.mean(ious)) if ious else 0.0
        active_ratios = []
        for hm in ag.hms:
            if hm.size == 0:
                active_ratios.append(0.0)
                continue
            active = float(np.count_nonzero(hm >= thr))
            total = float(hm.size)
            active_ratios.append((active / total) if total > 0 else 0.0)
        active_ratio_mean = float(np.mean(active_ratios)) if active_ratios else 0.0
        row = {
            "layer_idx": int(lidx),
            "feature_idx": int(fidx),
            "miou": float(miou),
            "nd_threshold": float(thr),
            "n_images": len(ious),
            "individual_ious": ",".join(f"{x:.6f}" for x in ious),
            "active_ratio_mean": f"{active_ratio_mean:.8f}",
            "overlay_dir": "",
        }
        rows_by_layer.setdefault(int(lidx), []).append(row)

    # Schreibe CSVs
    for lidx, rows in rows_by_layer.items():
        layer_dir = os.path.join(export_root, f"layer{lidx}")
        os.makedirs(layer_dir, exist_ok=True)
        rows = sorted(rows, key=lambda r: r.get("miou", 0.0), reverse=True)
        csv_path = os.path.join(layer_dir, "miou_network_dissection.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "layer_idx","feature_idx","miou","nd_threshold","n_images","individual_ious","active_ratio_mean","overlay_dir"
            ])
            writer.writeheader()
            writer.writerows(rows)


def _read_csv_rows(path: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    if not os.path.isfile(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            out.append(r)
    return out


def compare_roots(old_root: str, new_root: str) -> None:
    # Vergleiche pro Layer CSVs
    layers = []
    for name in os.listdir(old_root):
        if name.startswith("layer"):
            layers.append(int(name.replace("layer", "")))
    layers.sort()

    ok = True
    for l in layers:
        p_old = os.path.join(old_root, f"layer{l}", "miou_network_dissection.csv")
        p_new = os.path.join(new_root, f"layer{l}", "miou_network_dissection.csv")
        old_rows = _read_csv_rows(p_old)
        new_rows = _read_csv_rows(p_new)
        if len(old_rows) != len(new_rows):
            print(f"Layer {l}: verschiedene Zeilenzahlen: {len(old_rows)} vs {len(new_rows)}")
            ok = False
            continue
        # Sortiere nach Feature-Idx für deterministische Zuordnung
        old_rows = sorted(old_rows, key=lambda r: int(r["feature_idx"]))
        new_rows = sorted(new_rows, key=lambda r: int(r["feature_idx"]))
        for orow, nrow in zip(old_rows, new_rows):
            for k in ("miou", "nd_threshold", "active_ratio_mean"):
                a = float(orow[k])
                b = float(nrow[k])
                if not np.allclose(a, b, atol=1e-6):
                    print(f"Layer {l} Feature {orow['feature_idx']}: Feld {k} differiert: {a} vs {b}")
                    ok = False
    if ok:
        print("Vergleich OK: CSVs sind numerisch äquivalent (atol=1e-6)")
    else:
        print("Vergleich FEHLER: Unterschiede gefunden")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Vergleiche alten vs. neuen ND-Pipeline-Export")
    ap.add_argument("--encoder-out-dir", required=False, default=None)
    ap.add_argument("--percentile", type=float, default=90.0)
    ap.add_argument("--tmp-root", default=os.path.join(ROOT, "output", "_nd_compare"))
    args = ap.parse_args()

    encoder_out_dir = args.encoder_out_dir
    # zwei Export-Verzeichnisse
    ref_root = os.path.join(args.tmp_root, "ref")
    new_root = os.path.join(args.tmp_root, "new")
    os.makedirs(ref_root, exist_ok=True)
    os.makedirs(new_root, exist_ok=True)

    # 1) Referenzlauf (in-memory)
    _run_reference(encoder_out_dir=encoder_out_dir, export_root=ref_root, percentile=args.percentile)

    # 2) Neuer Lauf (streaming) – Overlays sind hier egal
    main_export_network_dissection(
        percentile=args.percentile,
        top_percent_active=None,
        mask_dir=None,
        encoder_out_dir=encoder_out_dir,
        export_root=new_root,
        export_mode="per-layer-best",
        approx_method="reservoir",
        reservoir_size=200_000,
        write_individual_ious=True,
        overlay_limit_per_feature=0,
    )

    # 3) Vergleich
    compare_roots(ref_root, new_root)
