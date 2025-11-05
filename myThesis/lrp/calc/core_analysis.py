"""
Core analysis logic for running MaskDINO-specific LRP attributions over images.
Only the "lrp" method is supported; legacy LRP and Grad*Input paths were removed.
"""
from __future__ import annotations

import os
import gc
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import PIL.Image

import torch
from torch import Tensor

from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog

# (Legacy LRP engine imports removed)

from myThesis.lrp.calc.config_build import build_cfg_for_inference, DEFAULT_WEIGHTS
from myThesis.lrp.calc.layer_finder import (
    list_encoder_like_layers,
    list_decoder_like_layers,
)
from myThesis.lrp.calc.tensor_utils import (
    aggregate_channel_relevance,
)
from myThesis.lrp.calc.hooks_maskdino import register_cut_hooks_by_module
from myThesis.lrp.lrp.engine import LRPEngine
from myThesis.lrp.lrp.config import (
    TARGET_TOKEN_IDX,
    USE_SUBLAYER,
    MEASUREMENT_POINT,
    DETERMINISTIC,
    SEED,
)
from myThesis.lrp.lrp.value_path import AttnCache
from myThesis.lrp.calc.io_utils import collect_images



def run_analysis(
    images_dir: str,
    layer_index: int,
    feature_index: int,
    output_csv: str,
    target_norm: str = "sum1",
    lrp_epsilon: float = 1e-6,
    which_module: str = "encoder",
    method: str = "lrp",
    index_kind: str = "auto",
):
    logger = logging.getLogger("lrp")
    logger.info("Starte LRP/Attribution-Analyse…")

    if not os.path.exists(DEFAULT_WEIGHTS):
        raise FileNotFoundError(f"Gewichtsdatei nicht gefunden: {DEFAULT_WEIGHTS}")

    # Konfiguration erstellen (immer CPU)
    device = "cpu"
    cfg = build_cfg_for_inference(device=device)
    setup_logger()  # detectron2 logger

    # Minimalen Dataset-Eintrag registrieren (nur Metadaten/Classes)
    try:
        car_parts_classes = [
            'back_bumper', 'back_door', 'back_glass', 'back_left_door', 'back_left_light',
            'back_light', 'back_right_door', 'back_right_light', 'front_bumper', 'front_door',
            'front_glass', 'front_left_door', 'front_left_light', 'front_light', 'front_right_door',
            'front_right_light', 'hood', 'left_mirror', 'object', 'right_mirror',
            'tailgate', 'trunk', 'wheel'
        ]
        MetadataCatalog.get("car_parts_minimal").set(thing_classes=car_parts_classes)
    except Exception:
        pass

    # Modell direkt bauen (kein DefaultPredictor; LRP arbeitet ohne Gradienten)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    model.to(device)

    # Deterministische Seeds (optional gesteuert über config)
    try:
        import random
        if DETERMINISTIC:
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.use_deterministic_algorithms(False)
    except Exception:
        pass

    # Nur die spezialisierte LRP-Engine wird unterstützt.
    if method != "lrp":
        raise ValueError(
            "Nur method='lrp' wird unterstützt. Die Pfade für 'legacy-lrp' und 'Grad*Input' wurden entfernt."
        )

    # Encoder- oder Decoder-Layer finden
    if which_module == "decoder":
        enc_layers = list_decoder_like_layers(model)
        layer_role = "Decoder"
    else:
        enc_layers = list_encoder_like_layers(model)
        layer_role = "Encoder"
    if not enc_layers:
        # Fallback: alle Module mit passenden Keywords listen
        key = "decoder" if which_module == "decoder" else "encoder"
        names = [n for n, _ in model.named_modules() if key in n.lower()]
        raise RuntimeError(
            f"Konnte keine {layer_role}-Layer finden. Gefundene '{key}'-Module: " + ", ".join(names)
        )

    # layer_index als 1-basiert interpretieren (intuitiver für Nutzer)
    if layer_index <= 0 or layer_index > len(enc_layers):
        msg = (
            f"layer_index {layer_index} ungültig. Es gibt {len(enc_layers)} {layer_role}-Kandidaten.\n"
            + "Gefundene Layer:\n"
            + "\n".join([f"  {i+1}: {n} ({type(m).__name__})" for i, (n, m) in enumerate(enc_layers)])
        )
        raise IndexError(msg)

    chosen_name, chosen_layer = enc_layers[layer_index - 1]
    logger.info(f"Gewähltes {layer_role}-Layer [{layer_index}]: {chosen_name} ({type(chosen_layer).__name__})")

    # Index-Achse bestimmen: Decoder -> Token, Encoder -> Kanal (wenn auto)
    if index_kind not in ("auto", "channel", "token"):
        raise ValueError("index_kind muss 'auto', 'channel' oder 'token' sein")
    index_axis = (
        ("token" if which_module == "decoder" else "channel")
        if index_kind == "auto"
        else index_kind
    )
    logger.info(f"Index-Achse: {index_axis} (index_kind={index_kind})")

    # Bilder sammeln
    img_files = collect_images(images_dir)
    if not img_files:
        raise FileNotFoundError(f"Keine Bilder in {images_dir} gefunden")

    # Aggregation über Bilder
    agg_attr: Tensor | None = None
    agg_attr_plus_skip: Tensor | None = None
    processed = 0

    # Vorverarbeiter analog zum DefaultPredictor (feste Werte)
    resize_aug = T.ResizeShortestEdge(short_edge_length=320, max_size=512)

    logger.info(f"Verarbeite {len(img_files)} Bilder aus: {images_dir}")
    logger.debug("Dateiliste:\n" + "\n".join(img_files))

    # Wir registrieren Hooks erst im Bild-Loop

    for img_path in img_files:
        try:
            # Bild laden (BGR wie in DefaultPredictor, dann ggf. nach RGB drehen)
            pil_im = PIL.Image.open(img_path).convert("RGB")
            original_rgb = np.array(pil_im)  # RGB (H, W, 3)
            original_h, original_w = original_rgb.shape[:2]

            # Falls Modell RGB erwartet, lassen wir RGB; ansonsten drehen nach BGR
            if cfg.INPUT.FORMAT == "RGB":
                model_input = original_rgb
            else:
                model_input = original_rgb[:, :, ::-1]  # RGB -> BGR

            # Resize
            tfm = resize_aug.get_transform(original_rgb)
            model_input = tfm.apply_image(model_input)

            # Tensor (C,H,W)
            image_tensor = torch.as_tensor(model_input.astype("float32").transpose(2, 0, 1))
            image_tensor = image_tensor.to(device)

            # Batch zusammenstellen
            batched_inputs = [{
                "image": image_tensor,
                "height": original_h,
                "width": original_w,
            }]

            # Forward (nur LRP)
            # Neue MaskDINO-spezifische LRP-Pipeline: stabile Cut-Hooks + Engine
            attn_cache = AttnCache()
            cps, handles = register_cut_hooks_by_module(chosen_layer, attn_cache=attn_cache)
            try:
                with torch.inference_mode():
                    _ = model(batched_inputs)
                # Engine rechnen
                engine = LRPEngine(epsilon=lrp_epsilon)
                # 1) Variante ohne Skip-Relevanz (only_transform)
                res = engine.run_local(
                    cps,
                    feature_index=feature_index,
                    target_token_idx=TARGET_TOKEN_IDX,
                    norm=target_norm if target_norm in ("sum1", "sumAbs1", "none") else "sum1",
                    which_module=which_module,
                    use_sublayer=USE_SUBLAYER,
                    measurement_point=MEASUREMENT_POINT,
                    attn_cache=attn_cache,
                    conservative_residual=False,
                    index_axis=index_axis,
                    token_reduce="mean",
                )
                # 2) Variante mit konservativer Residual-Aufteilung (inkl. Skip)
                res_with_skip = engine.run_local(
                    cps,
                    feature_index=feature_index,
                    target_token_idx=TARGET_TOKEN_IDX,
                    norm=target_norm if target_norm in ("sum1", "sumAbs1", "none") else "sum1",
                    which_module=which_module,
                    use_sublayer=USE_SUBLAYER,
                    measurement_point=MEASUREMENT_POINT,
                    attn_cache=attn_cache,
                    conservative_residual=True,
                    index_axis=index_axis,
                    token_reduce="mean",
                )
                # Kanalaggregation wie gehabt
                attr = aggregate_channel_relevance(res.R_prev)
                attr_plus_skip = aggregate_channel_relevance(res_with_skip.R_prev)
                if not torch.isfinite(attr).all():
                    logger.warning("Nicht-endliche Relevanzwerte detektiert; Bild übersprungen.")
                    continue
                if agg_attr is None:
                    agg_attr = attr.clone()
                else:
                    agg_attr += attr
                if not torch.isfinite(attr_plus_skip).all():
                    logger.warning("Nicht-endliche Relevanzwerte (plus_skip) detektiert; Bild übersprungen.")
                    continue
                if agg_attr_plus_skip is None:
                    agg_attr_plus_skip = attr_plus_skip.clone()
                else:
                    agg_attr_plus_skip += attr_plus_skip
                processed += 1
            finally:
                for h in handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
                cps.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.exception(f"Fehler bei {img_path}: {e}")
        finally:
            # Nothing extra to drop beyond per-image cleanup
            pass


    if agg_attr is None or agg_attr_plus_skip is None or processed == 0:
        raise RuntimeError("Keine Attributionen konnten berechnet werden.")

    # Mittelwert über Bilder
    agg_attr = agg_attr / float(processed)
    agg_attr_plus_skip = agg_attr_plus_skip / float(processed)

    # Export nach CSV
    df = pd.DataFrame(
        {
            "prev_feature_idx": list(range(len(agg_attr))),
            "relevance": agg_attr.numpy().tolist(),
            "relevance_plus_skip": agg_attr_plus_skip.numpy().tolist(),
            "layer_index": layer_index,
            "layer_name": chosen_name,
            "feature_index": feature_index,
            "epsilon": lrp_epsilon,
            "module_role": layer_role,
            "target_norm": target_norm,
            "method": method,
        }
    ).sort_values("relevance", ascending=False)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Logging der Top-10 Beiträge
    topk = df.head(10)
    logger.info("Top-10 vorherige Features nach Relevanz (ohne Skip):")
    for _, row in topk.iterrows():
        logger.info(f"  idx={int(row.prev_feature_idx):4d}  rel={row.relevance:.6f}")

    # Logging der Top-10 (mit Skip)
    topk_plus = df.sort_values("relevance_plus_skip", ascending=False).head(10)
    logger.info("Top-10 vorherige Features nach Relevanz (mit Skip):")
    for _, row in topk_plus.iterrows():
        logger.info(
            f"  idx={int(row.prev_feature_idx):4d}  rel+skip={row.relevance_plus_skip:.6f}"
        )

    logger.info(
        f"Fertig. Auswertung über {processed} Bilder. Ergebnis gespeichert in: {output_csv}"
    )
