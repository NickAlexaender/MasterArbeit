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
    LN_RULE,
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
    weights_path: str | None = None,
):
    logger = logging.getLogger("lrp")
    logger.info("Starte LRP/Attribution-Analyse…")

    # Gewichte wählen: explizit übergeben > Default
    chosen_weights = weights_path if weights_path else DEFAULT_WEIGHTS
    if not os.path.exists(chosen_weights):
        raise FileNotFoundError(f"Gewichtsdatei nicht gefunden: {chosen_weights}")

    # Konfiguration erstellen (immer CPU)
    device = "cpu"
    cfg = build_cfg_for_inference(device=device, weights_path=chosen_weights)
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
            torch.use_deterministic_algorithms(True)
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        logger.info(f"Determinismus aktiv: {DETERMINISTIC}, Seed={SEED}")
    except Exception:
        logger.warning("Determinismus konnte nicht vollständig aktiviert werden.", exc_info=True)

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

    # Zusätzliche Robustheit: Falls fälschlich ein Encoder-Pfad als Decoder gewählt wurde,
    # versuche automatisch auf einen echten Decoder-Kandidaten umzuschalten.
    if which_module == "decoder" and ".encoder." in chosen_name.lower():
        try:
            dec_only = [(n, m) for (n, m) in enc_layers if ".encoder." not in n.lower()]
            if dec_only:
                chosen_name, chosen_layer = dec_only[min(layer_index - 1, len(dec_only) - 1)]
                logger.warning(
                    f"Decoder-Auswahl enthielt Encoder-Pfad. Umschalten auf Decoder-Kandidat: {chosen_name}"
                )
        except Exception:
            pass

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
    agg_attr_skip_only: Tensor | None = None
    agg_attr_skip_component: Tensor | None = None
    agg_attr_skip_component_raw: Tensor | None = None
    processed = 0

    # Vorverarbeiter analog zum DefaultPredictor (Werte aus cfg übernehmen)
    resize_aug = T.ResizeShortestEdge(
        short_edge_length=getattr(cfg.INPUT, "MIN_SIZE_TEST", 800),
        max_size=getattr(cfg.INPUT, "MAX_SIZE_TEST", 1333),
    )

    logger.info(f"Verarbeite {len(img_files)} Bilder aus: {images_dir}")
    logger.debug("Dateiliste:\n" + "\n".join(img_files))

    # Wir registrieren Hooks erst im Bild-Loop

    def _aggregate_relevance(vec: Tensor, axis: str) -> Tensor:
        """
        Aggregiert Relevanz vektor- oder matrixförmig entsprechend der Index-Achse.
        - channel: nutzt aggregate_channel_relevance (bestehend)
        - token: summiert über Nicht-Token-Dimension(en) und liefert pro Token einen Wert
        """
        if axis == "channel":
            return aggregate_channel_relevance(vec)
        # Token-Aggregation: bevorzuge die Engine-Form (B,T,C)
        with torch.no_grad():
            if vec.dim() == 3:
                # (B,T,C) -> pro Token summieren über Batch und Kanäle
                return vec.sum(dim=(0, 2))
            if vec.dim() == 2:
                # (T,C) oder (B,C) -> über Kanäle summieren; ergibt (T,) bzw. (B,)
                return vec.sum(dim=1)
            if vec.dim() == 1:
                # bereits Vektor
                return vec
            # Generischer Fallback: Token-Achse an Position 1 annehmen
            try:
                Tdim = vec.shape[1]
                return vec.movedim(1, 0).reshape(Tdim, -1).sum(dim=1)
            except Exception:
                # letzte Rettung: alles auf Skalar und als 1D ausgeben
                return vec.reshape(-1)

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

            # Resize (Transform von der tatsächlich ans Modell gehenden Darstellung ableiten)
            tfm = resize_aug.get_transform(model_input)
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
            # Präferenz setzen: im Decoder je nach USE_SUBLAYER, im Encoder i. d. R. Self-Attention
            try:
                prefer = (USE_SUBLAYER if which_module == "decoder" else "self")
                setattr(attn_cache, "prefer_kind", prefer)
            except Exception:
                pass
            cps, handles = register_cut_hooks_by_module(chosen_layer, attn_cache=attn_cache)
            try:
                with torch.inference_mode():
                    _ = model(batched_inputs)
                # Engine rechnen: einmal, ohne Normierung, nur Transform-Pfad
                engine = LRPEngine(epsilon=lrp_epsilon)
                norm_value = target_norm if target_norm in ("sum1", "sumAbs1", "none") else "sum1"
                if norm_value != target_norm:
                    logger.warning(f"Ungültiges target_norm='{target_norm}'. Fallback auf '{norm_value}'.")

                res = engine.run_local(
                    cps,
                    feature_index=feature_index,
                    target_token_idx=TARGET_TOKEN_IDX,
                    norm="none",
                    which_module=which_module,
                    use_sublayer=USE_SUBLAYER,
                    measurement_point=MEASUREMENT_POINT,
                    attn_cache=attn_cache,
                    conservative_residual=False,
                    index_axis=index_axis,
                    token_reduce="mean",
                )

                # Normierungshelper
                def _norm_tensor(R: Tensor, mode: str) -> Tensor:
                    if mode == "sum1":
                        s = (R.sum() + 1e-12)
                        return R / s
                    if mode == "sumAbs1":
                        s = (R.abs().sum() + 1e-12)
                        return R / s
                    return R

                R_transform = res.R_transform_raw if isinstance(res.R_transform_raw, torch.Tensor) else res.R_prev
                Rx_skip = res.Rx_skip_raw if isinstance(res.Rx_skip_raw, torch.Tensor) else torch.zeros_like(R_transform)

                R_no_skip_n = _norm_tensor(R_transform, norm_value)
                R_with_skip_n = _norm_tensor(R_transform + Rx_skip, norm_value)
                R_skip_n = _norm_tensor(Rx_skip, norm_value)

                # Kanal-/Tokenaggregation
                attr = _aggregate_relevance(R_no_skip_n, index_axis)
                attr_plus_skip = _aggregate_relevance(R_with_skip_n, index_axis)
                # "skip_only": als echte (normalisierte) Skip-Komponente reporten, nicht als Differenz
                attr_skip_comp = _aggregate_relevance(R_skip_n, index_axis)
                attr_skip_only = attr_skip_comp.clone()
                attr_skip_comp_raw = _aggregate_relevance(Rx_skip, index_axis)
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
                if attr_skip_only is not None:
                    if not torch.isfinite(attr_skip_only).all():
                        logger.warning("Nicht-endliche Relevanzwerte (skip_only) detektiert; Skip-Spalte wird für dieses Bild ausgelassen.")
                    else:
                        if agg_attr_skip_only is None:
                            agg_attr_skip_only = attr_skip_only.clone()
                        else:
                            agg_attr_skip_only += attr_skip_only
                # Skip-Komponente separat aggregieren (normalisiert und roh)
                if not torch.isfinite(attr_skip_comp).all():
                    logger.warning("Nicht-endliche Relevanzwerte (skip_component) detektiert; Bild übersprungen.")
                else:
                    if agg_attr_skip_component is None:
                        agg_attr_skip_component = attr_skip_comp.clone()
                    else:
                        agg_attr_skip_component += attr_skip_comp
                if not torch.isfinite(attr_skip_comp_raw).all():
                    logger.warning("Nicht-endliche Relevanzwerte (skip_component_raw) detektiert; Bild übersprungen.")
                else:
                    if agg_attr_skip_component_raw is None:
                        agg_attr_skip_component_raw = attr_skip_comp_raw.clone()
                    else:
                        agg_attr_skip_component_raw += attr_skip_comp_raw
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
    if agg_attr_skip_only is not None:
        agg_attr_skip_only = agg_attr_skip_only / float(processed)
    if agg_attr_skip_component is not None:
        agg_attr_skip_component = agg_attr_skip_component / float(processed)
    if agg_attr_skip_component_raw is not None:
        agg_attr_skip_component_raw = agg_attr_skip_component_raw / float(processed)

    # Sanity-Check: Decoder sollte 300 Objekt-Queries liefern
    try:
        if which_module == "decoder" and index_axis == "token":
            expected_q = int(getattr(getattr(cfg, "MODEL").MaskDINO, "NUM_OBJECT_QUERIES", 300))
            if len(agg_attr) != expected_q:
                logger.warning(
                    f"Decoder-Tokenanzahl = {len(agg_attr)} ungleich erwartet {expected_q}. "
                    f"Gewähltes Layer: {chosen_name}. Prüfe, ob wirklich ein Decoder-Block gehookt wurde."
                )
    except Exception:
        pass

    try:
        logger.info(f"Aggregationsvektor Länge: {len(agg_attr)} (index_axis={index_axis})")
    except Exception:
        pass

    # Export nach CSV
    data_dict = {
        "prev_feature_idx": list(range(len(agg_attr))),
        "relevance": agg_attr.numpy().tolist(),
        "relevance_plus_skip": agg_attr_plus_skip.numpy().tolist(),
        "layer_index": layer_index,
        "layer_name": chosen_name,
        "feature_index": feature_index,
        "epsilon": lrp_epsilon,
        "module_role": layer_role,
        "target_norm": target_norm,
        "index_kind": index_kind,
        "index_axis": index_axis,
        "method": method,
    }
    if agg_attr_skip_only is not None:
        data_dict["relevance_skip_only"] = agg_attr_skip_only.numpy().tolist()
    if agg_attr_skip_component is not None:
        data_dict["relevance_skip_component"] = agg_attr_skip_component.numpy().tolist()
    if agg_attr_skip_component_raw is not None:
        data_dict["relevance_skip_component_raw"] = agg_attr_skip_component_raw.numpy().tolist()
    df = pd.DataFrame(data_dict).sort_values("relevance", ascending=False)

    dirpath = os.path.dirname(output_csv) or "."
    os.makedirs(dirpath, exist_ok=True)
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
