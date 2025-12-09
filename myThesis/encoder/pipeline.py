"""Pipeline-Implementierung für Network Dissection Export (speichersparend, streaming-basiert).

Neues Design (M1-optimiert, bounded memory):
- Mehrpass-Verarbeitung ohne Speicherung kompletter Heatmap-Sammlungen
- Streaming-Perzentile je Feature (Reservoir oder Histogramm)
- CSV-Schreiben pro Layer mit minimalem Speicherbedarf
- Overlays werden in einem separaten Pass nur für Gewinner-Features erzeugt

Beinhaltet:
- resolve_percentile(...)
- main_export_network_dissection(...)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

# Threading-Limits früh setzen, bevor große Libs initialisiert werden
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")  # Apple Accelerate
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from .aggregator import FeatureStats
from .config import DEFAULT_COMBINE, NETWORK_DISSECTION_PERCENTILE
from .export_utils import CsvAppender, _write_network_dissection_csv, resolve_export_root
from .iterators import iter_iou_inputs, _iter_csv_rows
from .iou_core import compute_iou_from_heatmap, generate_heatmap_only
from .io_utils import _ensure_dir, _find_layer_csvs, _load_all_shapes, _select_shapes_for
from .mask_utils import _save_overlay_comparison

logger = logging.getLogger(__name__)


def resolve_percentile(percentile: Optional[float], top_percent_active: Optional[float]) -> float:
    """Ermittelt das Perzentil (0–100) für den Threshold.

    Regeln:
    - Wenn top_percent_active gesetzt ist (in %), nutze Perzentil = 100 - top_percent_active.
    - Wenn percentile None ist, nutze NETWORK_DISSECTION_PERCENTILE.
    - Wenn 0 < percentile <= 1 ist, interpretiere es als Anteil (0..1) -> Prozentil = percentile * 100.
    - Ansonsten clamp (0–100).
    """
    if top_percent_active is not None:
        try:
            val = float(top_percent_active)
        except Exception as e:
            raise ValueError(f"Ungültiger Wert für top_percent_active: {top_percent_active}") from e
        if not (0.0 < val <= 100.0):
            raise ValueError(f"top_percent_active muss in (0, 100] liegen, erhalten: {val}")
        return 100.0 - val

    if percentile is None:
        return float(NETWORK_DISSECTION_PERCENTILE)
    try:
        p = float(percentile)
    except Exception as e:
        raise ValueError(f"Ungültiger Wert für percentile: {percentile}") from e

    if 0.0 < p <= 1.0:
        pct = p * 100.0
        logger.info("percentile im Bereich (0,1] erkannt; interpretiere als Anteil: %.6f -> %.6f", p, pct)
        return pct

    if p < 0.0:
        p = 0.0
    elif p > 100.0:
        p = 100.0
    return p


def _export_per_layer_best(
    winners: Dict[int, List[Tuple[int, float]]],
    export_root: str,
    encoder_out_dir: Optional[str],
    mask_dir: Optional[str],
    combine: str,
    overlay_limit_per_feature: Optional[int] = None,
    gc_every_n: int = 250,
) -> None:
    """Erzeugt Overlays für beste Features pro Layer (ggf. mehrere bei Tie) on-the-fly."""
    if not winners:
        return
    # Dritter Pass: alle Inputs durchgehen und nur Winner schreiben
    written_per_feat: Dict[Tuple[int, int], int] = {}
    n_since_gc = 0
    for item in iter_iou_inputs(encoder_out_dir=encoder_out_dir, mask_dir=mask_dir):
        key = (item.layer_idx, item.feature_idx)
        feats = winners.get(item.layer_idx)
        if not feats:
            continue
        thr = None
        for fidx, t in feats:
            if fidx == item.feature_idx:
                thr = float(t)
                break
        if thr is None:
            continue
        # Limit pro Feature beachten (falls gesetzt)
        if overlay_limit_per_feature is not None and written_per_feat.get(key, 0) >= int(overlay_limit_per_feature):
            continue
        layer_dir = os.path.join(export_root, f"layer{item.layer_idx}")
        feat_dir = os.path.join(layer_dir, "network_dissection", f"Feature{item.feature_idx}")
        _ensure_dir(feat_dir)
        heatmap = generate_heatmap_only(item, combine=combine)
        out_path = os.path.join(feat_dir, f"{item.image_id}.png")
        _save_overlay_comparison(out_path, item.mask_input, heatmap, float(thr))
        written_per_feat[key] = written_per_feat.get(key, 0) + 1
        n_since_gc += 1
        if overlay_limit_per_feature is not None and written_per_feat[key] >= int(overlay_limit_per_feature):
            # keine weiteren Bilder für dieses Feature
            pass
        if n_since_gc >= int(gc_every_n):
            gc.collect()
            n_since_gc = 0


def _export_per_layer_best_rot(
    layer_best_single: Dict[int, Tuple[int, float]],
    mask_dir: str,
    encoder_out_dir: Optional[str],
    combine: str,
    overlay_limit_per_feature: Optional[int] = None,
    gc_every_n: int = 250,
) -> None:
    """Exportiert pro Layer exakt EIN bestes Feature nach mask_dir/layerX (on-the-fly)."""
    # Dritter Pass: Nur die besten je Layer schreiben
    written_per_layer: Dict[int, int] = {}
    n_since_gc = 0
    for item in iter_iou_inputs(encoder_out_dir=encoder_out_dir, mask_dir=None):
        sel = layer_best_single.get(item.layer_idx)
        if sel is None:
            continue
        fidx, thr = sel
        if item.feature_idx != fidx:
            continue
        # Limit beachten (falls gesetzt)
        if overlay_limit_per_feature is not None and written_per_layer.get(item.layer_idx, 0) >= int(overlay_limit_per_feature):
            continue
        layer_out_dir = os.path.join(mask_dir, f"layer{item.layer_idx}")
        _ensure_dir(layer_out_dir)
        heatmap = generate_heatmap_only(item, combine=combine)
        out_path = os.path.join(layer_out_dir, f"{item.image_id}.png")
        _save_overlay_comparison(out_path, item.mask_input, heatmap, float(thr))
        written_per_layer[item.layer_idx] = written_per_layer.get(item.layer_idx, 0) + 1
        n_since_gc += 1
        if overlay_limit_per_feature is not None and written_per_layer[item.layer_idx] >= int(overlay_limit_per_feature):
            # Kappung pro Layer (falls gewünscht)
            pass
        if n_since_gc >= int(gc_every_n):
            gc.collect()
            n_since_gc = 0


def _export_global_best(
    global_best: Optional[Dict[str, object]],
    export_root: str,
    encoder_out_dir: Optional[str],
    combine: str,
    mask_dir: Optional[str] = None,
) -> None:
    """Exportiert global_best: nur das beste Bild des global besten Features (on-the-fly)."""
    if global_best is None:
        logger.info("Keine globalen Bestwerte gefunden – Überspringe global_best-Export.")
        return
    gb_layer = int(global_best["layer_idx"])  # type: ignore
    gb_feat = int(global_best["feature_idx"])  # type: ignore
    gb_thr = float(global_best["threshold"])  # type: ignore
    gb_miou = float(global_best["miou"])  # type: ignore
    gb_best_img_id = str(global_best.get("best_image_id", ""))  # type: ignore
    gb_best_img_iou = float(global_best.get("best_image_iou", 0.0))  # type: ignore

    global_export_root = os.path.join(export_root, "global_best")
    _ensure_dir(global_export_root)

    # Falls best_image_id bekannt: nur dieses Bild rendern
    best_overlay_path = None
    for item in iter_iou_inputs(encoder_out_dir=encoder_out_dir, mask_dir=mask_dir):
        if item.layer_idx != gb_layer or item.feature_idx != gb_feat:
            continue
        if gb_best_img_id and item.image_id != gb_best_img_id:
            continue
        heatmap = generate_heatmap_only(item, combine=combine)
        best_filename = f"layer{gb_layer}_Feature{gb_feat}_{item.image_id}.png"
        best_overlay_path = os.path.join(global_export_root, best_filename)
        _ensure_dir(os.path.dirname(best_overlay_path))
        _save_overlay_comparison(best_overlay_path, item.mask_input, heatmap, gb_thr)
        gb_best_img_id = item.image_id
        break

    # optionale Metadatei
    try:
        info = {
            "layer_idx": gb_layer,
            "feature_idx": gb_feat,
            "miou_feature": gb_miou,
            "nd_threshold": gb_thr,
            "best_image_id": gb_best_img_id,
            "best_image_iou": gb_best_img_iou,
            "overlay_file": os.path.basename(best_overlay_path) if best_overlay_path else "",
        }
        with open(os.path.join(global_export_root, "info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    if best_overlay_path:
        logger.info(
            "Global_best gespeichert -> layer%s/Feature%s, Bild '%s' (IoU=%.6f) in %s",
            gb_layer,
            gb_feat,
            gb_best_img_id,
            gb_best_img_iou,
            best_overlay_path,
        )


def main_export_network_dissection(
    percentile: Optional[float] = None,
    top_percent_active: Optional[float] = None,
    mask_dir: Optional[str] = None,
    encoder_out_dir: Optional[str] = None,
    export_root: Optional[str] = None,
    export_mode: str = "per-layer-best",
    # neue Optionen
    approx_method: str = "reservoir",  # oder "histogram"
    reservoir_size: int = 200_000,
    num_bins: int = 1024,
    overlay_limit_per_feature: Optional[int] = None,
    write_individual_ious: bool = False,
    gc_every_n: int = 250,
) -> None:
    """Network Dissection Pipeline mit per-Feature Thresholding (streaming, bounded memory)."""
    t0 = time.time()
    # Prozentil bestimmen (robuste Interpretation)
    percentile_val = resolve_percentile(percentile, top_percent_active)
    # Export-Root auflösen
    export_root = export_root or resolve_export_root(encoder_out_dir)
    _ensure_dir(export_root)

    logger.info("Konfiguration:")
    logger.info("  Perzentil:          %s (größer -> weniger aktiv)", percentile_val)
    logger.info("  Erwartete Aktiv-Rate grob: ~%.6f%% der Pixel", max(0.0, 100.0 - float(percentile_val)))
    logger.info("  Masken-Ordner:      %s", mask_dir)
    logger.info("  Encoder-Output:     %s", encoder_out_dir)
    logger.info("  Export-Ziel:        %s", export_root)
    logger.info("  Export-Modus:       %s  (per-layer-best | global-best | per-layer-best-rot)", export_mode)
    logger.info("  Approximation:      %s (reservoir_size=%s, num_bins=%s)", approx_method, reservoir_size, num_bins)

    layer_csvs = _find_layer_csvs(encoder_out_dir)
    all_shapes = _load_all_shapes(encoder_out_dir)
    if not layer_csvs:
        logger.info("Keine Daten gefunden. Bitte zuvor die Extraktion ausführen.")
        return

    # ------------------
    # Pass 1: Threshold-Discovery (Streaming)
    # ------------------
    logger.info("Pass 1: Sammle Streaming-Statistiken je Feature (%s)…", approx_method)
    stats: Dict[Tuple[int, int], FeatureStats] = {}
    n_seen_rows = 0
    n_since_gc = 0
    for lidx, csv_path in layer_csvs:
        for image_id, feat_idx, tokens in _iter_csv_rows(csv_path):
            shapes = _select_shapes_for(image_id, tokens.size, all_shapes)
            if shapes is None:
                logger.warning("Keine shapes.json für image_id='%s' -> Zeile übersprungen", image_id)
                continue
            hm = generate_heatmap_only((lidx, image_id, feat_idx, tokens, shapes, np.zeros((1, 1), dtype=bool)), combine=DEFAULT_COMBINE)
            key = (lidx, int(feat_idx))
            st = stats.get(key)
            if st is None:
                st = FeatureStats(lidx, int(feat_idx), method=approx_method, reservoir_size=reservoir_size, num_bins=num_bins)
                stats[key] = st
            if approx_method == "histogram":
                st.ingest_heatmap(hm, stage="minmax")
            else:
                st.ingest_heatmap(hm)
            n_seen_rows += 1
            n_since_gc += 1
            if n_since_gc >= int(gc_every_n):
                gc.collect()
                n_since_gc = 0

    if approx_method == "histogram":
        logger.info("Pass 1b: Fülle Histogramme je Feature…")
        # Histogramm anlegen
        for st in stats.values():
            st.prepare_hist()
        # Werte in Histogramme streamen
        n_since_gc = 0
        for lidx, csv_path in layer_csvs:
            for image_id, feat_idx, tokens in _iter_csv_rows(csv_path):
                shapes = _select_shapes_for(image_id, tokens.size, all_shapes)
                if shapes is None:
                    continue
                hm = generate_heatmap_only((lidx, image_id, feat_idx, tokens, shapes, np.zeros((1, 1), dtype=bool)), combine=DEFAULT_COMBINE)
                stats[(lidx, int(feat_idx))].ingest_heatmap(hm, stage="hist")
                n_since_gc += 1
                if n_since_gc >= int(gc_every_n):
                    gc.collect()
                    n_since_gc = 0

    # Schwellen je Feature bestimmen und persistieren
    thresholds: Dict[Tuple[int, int], float] = {
        k: v.compute_threshold(percentile_val) for k, v in stats.items()
    }
    thr_json = {f"layer{k[0]}_feature{k[1]}": float(v) for k, v in thresholds.items()}
    with open(os.path.join(export_root, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump(thr_json, f, ensure_ascii=False, indent=2)

    # Zähle Bilder pro Feature (für Abschlusskriterium in Pass 2)
    images_per_feature: Dict[Tuple[int, int], int] = {k: 0 for k in thresholds.keys()}
    for lidx, csv_path in layer_csvs:
        for _image_id, feat_idx, _tokens in _iter_csv_rows(csv_path):
            key = (lidx, int(feat_idx))
            if key in images_per_feature:
                images_per_feature[key] += 1

    # ------------------
    # Pass 2: mIoU, Best-Feature pro Layer, CSV sammeln
    # ------------------
    logger.info("Pass 2: Berechne mIoU pro Feature und wähle Gewinner…")

    class _RunFeat:
        __slots__ = ("n_target", "count", "sum_iou", "active_ratio_sum", "ious", "threshold", "best_img_id", "best_img_iou")

        def __init__(self, n_target: int, threshold: float) -> None:
            self.n_target = int(n_target)
            self.count = 0
            self.sum_iou = 0.0
            self.active_ratio_sum = 0.0
            self.ious: Optional[List[float]] = ([] if write_individual_ious else None)
            self.threshold = float(threshold)
            self.best_img_id = ""
            self.best_img_iou = 0.0

    running: Dict[Tuple[int, int], _RunFeat] = {
        k: _RunFeat(n_target=images_per_feature.get(k, 0), threshold=t) for k, t in thresholds.items()
    }
    per_layer_rows: Dict[int, List[Dict[str, object]]] = {}
    per_layer_best: Dict[int, Dict[str, object]] = {}

    n_since_gc = 0
    for item in iter_iou_inputs(encoder_out_dir=encoder_out_dir, mask_dir=mask_dir):
        key = (item.layer_idx, item.feature_idx)
        rf = running.get(key)
        if rf is None or rf.n_target == 0:
            continue
        # Heatmap & IoU
        heatmap = generate_heatmap_only(item, combine=DEFAULT_COMBINE)
        thr = rf.threshold
        iou = compute_iou_from_heatmap(heatmap, item.mask_input, float(thr))
        active_ratio = float(np.count_nonzero(heatmap >= thr)) / float(heatmap.size if heatmap.size else 1)

        rf.count += 1
        rf.sum_iou += float(iou)
        rf.active_ratio_sum += float(active_ratio)
        if rf.ious is not None:
            rf.ious.append(float(iou))
        if float(iou) > rf.best_img_iou:
            rf.best_img_iou = float(iou)
            rf.best_img_id = str(item.image_id)

        n_since_gc += 1
        if n_since_gc >= int(gc_every_n):
            gc.collect()
            n_since_gc = 0

    # Abschluss: Zeilen pro Feature erzeugen, Bestwerte je Layer bestimmen
    for (lidx, fidx), rf in running.items():
        if rf.n_target <= 0:
            continue
        if rf.count != rf.n_target:
            # fehlende Daten tolerieren, aber berechenbar machen
            logger.debug("Feature (L%s,F%s): erwartet %s Bilder, gesehen %s", lidx, fidx, rf.n_target, rf.count)
        n_img = max(1, rf.count)
        miou = rf.sum_iou / float(n_img)
        active_ratio_mean = rf.active_ratio_sum / float(n_img)
        row = {
            "layer_idx": int(lidx),
            "feature_idx": int(fidx),
            "miou": float(miou),
            "nd_threshold": float(rf.threshold),
            "n_images": int(rf.count),
            "individual_ious": (",".join(f"{x:.6f}" for x in (rf.ious or [])) if write_individual_ious else ""),
            "active_ratio_mean": f"{active_ratio_mean:.8f}",
            "overlay_dir": "",
        }
        per_layer_rows.setdefault(int(lidx), []).append(row)

        # Bestes Feature pro Layer updaten (Tie -> sammeln)
        best = per_layer_best.get(int(lidx))
        if best is None:
            per_layer_best[int(lidx)] = {
                "best_miou": float(miou),
                "features": [
                    {"feature_idx": int(fidx), "threshold": float(rf.threshold), "best_image_id": rf.best_img_id, "best_image_iou": float(rf.best_img_iou)},
                ],
            }
        else:
            cur_best = float(best["best_miou"])  # type: ignore
            if float(miou) > cur_best + 1e-12:
                best["best_miou"] = float(miou)
                best["features"] = [
                    {"feature_idx": int(fidx), "threshold": float(rf.threshold), "best_image_id": rf.best_img_id, "best_image_iou": float(rf.best_img_iou)},
                ]
            elif abs(float(miou) - cur_best) <= 1e-12:
                best.setdefault("features", []).append(  # type: ignore
                    {"feature_idx": int(fidx), "threshold": float(rf.threshold), "best_image_id": rf.best_img_id, "best_image_iou": float(rf.best_img_iou)}
                )

    if export_mode == "per-layer-best-rot":
        if mask_dir is None:
            raise ValueError("mask_dir muss gesetzt sein für export_mode='per-layer-best-rot'")
        # Pro Layer genau ein bestes Feature wählen (bei Tie: kleinstes Feature)
        layer_best_single: Dict[int, Tuple[int, float]] = {}
        for lidx, best in per_layer_best.items():
            feats = best.get("features", [])  # type: ignore
            if not feats:
                continue
            chosen = sorted(feats, key=lambda fi: int(fi["feature_idx"]))[0]  # type: ignore
            layer_best_single[int(lidx)] = (int(chosen["feature_idx"]), float(chosen["threshold"]))  # type: ignore
        logger.info("Pass 3: Exportiere pro Layer NUR das jeweils beste Feature nach mask_dir/layerX …")
        _export_per_layer_best_rot(
            layer_best_single,
            mask_dir,
            encoder_out_dir=encoder_out_dir,
            combine=DEFAULT_COMBINE,
            overlay_limit_per_feature=overlay_limit_per_feature,
            gc_every_n=gc_every_n,
        )
        logger.info("Fertig: Overlays pro Layer unter mask_dir/layerX gespeichert.")
        return

    # Bestimme globale Gewinner und baue Overlay-Ziele
    global_best: Optional[Dict[str, object]] = None
    winners: Dict[int, List[Tuple[int, float]]] = {}
    for lidx, best in per_layer_best.items():
        feats = best.get("features", [])  # type: ignore
        if not feats:
            continue
        if export_mode == "global-best":
            candidate = sorted(feats, key=lambda fi: int(fi["feature_idx"]))[0]  # type: ignore
            # global_best wird später verglichen
            if (global_best is None) or (float(best["best_miou"]) > float(global_best["miou"])):
                global_best = {
                    "layer_idx": int(lidx),
                    "feature_idx": int(candidate["feature_idx"]),  # type: ignore
                    "miou": float(best["best_miou"]),  # type: ignore
                    "threshold": float(candidate["threshold"]),  # type: ignore
                    "best_image_id": str(candidate.get("best_image_id", "")),  # type: ignore
                    "best_image_iou": float(candidate.get("best_image_iou", 0.0)),  # type: ignore
                }
        else:  # per-layer-best
            winners[int(lidx)] = [(int(fi["feature_idx"]), float(fi["threshold"])) for fi in feats]  # type: ignore

    if export_mode == "global-best":
        winners = {}
        if global_best is not None:
            winners[int(global_best["layer_idx"]) ] = [(int(global_best["feature_idx"]), float(global_best["threshold"]))]  # type: ignore

    # CSVs pro Layer schreiben (overlay_dir für Gewinner ausfüllen)
    for lidx, rows in per_layer_rows.items():
        layer_dir = os.path.join(export_root, f"layer{lidx}")
        _ensure_dir(layer_dir)

        overlay_map: Dict[int, str] = {}
        if lidx in winners:
            for fidx, _thr in winners[lidx]:
                feat_dir = os.path.join(layer_dir, "network_dissection", f"Feature{fidx}")
                overlay_map[int(fidx)] = os.path.relpath(feat_dir, start=export_root)

        rows_sorted = sorted(rows, key=lambda r: r.get("miou", 0.0), reverse=True)
        for r in rows_sorted:
            fidx = int(r["feature_idx"])  # type: ignore
            if fidx in overlay_map:
                r["overlay_dir"] = overlay_map[fidx]

        csv_path = os.path.join(layer_dir, "miou_network_dissection.csv")
        _write_network_dissection_csv(csv_path, rows_sorted)

    # Pass 3: Overlays für Gewinner-Features
    logger.info("Pass 3: Erzeuge Overlays für Gewinner-Features…")
    _export_per_layer_best(
        winners=winners,
        export_root=export_root,
        encoder_out_dir=encoder_out_dir,
        mask_dir=mask_dir,
        combine=DEFAULT_COMBINE,
        overlay_limit_per_feature=overlay_limit_per_feature,
        gc_every_n=gc_every_n,
    )

    # Global_best-Overlay (nur ein Bild)
    _export_global_best(global_best, export_root=export_root, encoder_out_dir=encoder_out_dir, combine=DEFAULT_COMBINE, mask_dir=mask_dir)
    logger.info("Network Dissection Export abgeschlossen in %.2fs. Root: %s", time.time() - t0, export_root)


__all__ = [
    "main_export_network_dissection",
    "resolve_percentile",
]
