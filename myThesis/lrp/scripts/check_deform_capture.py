from __future__ import annotations

import argparse
import os
from typing import Tuple, List

import torch
import numpy as np
from PIL import Image

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.data import transforms as T

from myThesis.lrp.calc.config_build import build_cfg_for_inference, DEFAULT_WEIGHTS
from myThesis.lrp.calc.layer_finder import (
    list_encoder_like_layers,
    list_decoder_like_layers,
)
from myThesis.lrp.calc.io_utils import collect_images
from myThesis.lrp.calc.hooks_maskdino import register_cut_hooks_by_module
from myThesis.lrp.lrp.value_path import AttnCache


def _load_image(path: str, cfg) -> torch.Tensor:
    """Load image as CHW float32 RGB tensor normalized per cfg."""
    img = Image.open(path).convert("RGB")
    img = np.asarray(img)  # H,W,3 RGB uint8

    # Resize like DefaultPredictor
    resize_aug = T.ResizeShortestEdge(
        short_edge_length=getattr(cfg.INPUT, "MIN_SIZE_TEST", 800),
        max_size=getattr(cfg.INPUT, "MAX_SIZE_TEST", 1333),
    )
    img = resize_aug.get_transform(img).apply_image(img)

    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))  # C,H,W

    # normalize
    mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(3, 1, 1)
    std = torch.tensor(cfg.MODEL.PIXEL_STD).view(3, 1, 1)
    img = (img - mean) / std
    return img


def _choose_layer(model, which_module: str, layer_index: int):
    if which_module == "decoder":
        layers = list_decoder_like_layers(model)
        role = "decoder"
    else:
        layers = list_encoder_like_layers(model)
        role = "encoder"
    if not layers:
        raise RuntimeError(f"Keine {role}-ähnlichen Layer im Modell gefunden.")
    if layer_index <= 0 or layer_index > len(layers):
        raise IndexError(f"layer_index {layer_index} außerhalb [1,{len(layers)}]")
    return role, layers[layer_index - 1]


def run(images_dir: str, which_module: str = "encoder", layer_index: int = 1, device: str = "cpu") -> None:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(images_dir)
    imgs = collect_images(images_dir)
    if not imgs:
        raise RuntimeError(f"Keine Bilder in {images_dir} gefunden")

    cfg = build_cfg_for_inference(device=device)
    if not os.path.exists(cfg.MODEL.WEIGHTS):
        raise FileNotFoundError(
            f"Gewichte nicht gefunden: {cfg.MODEL.WEIGHTS}. Passe DEFAULT_WEIGHTS in config_build.py an."
        )

    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval().to(device)

    role, (name, module) = _choose_layer(model, which_module, layer_index)

    attn_cache = AttnCache()
    cps, handles = register_cut_hooks_by_module(module, attn_cache=attn_cache)

    img_path = imgs[0]
    img_tensor = _load_image(img_path, cfg)
    inputs = [{
        "image": img_tensor.to(device),
        "height": img_tensor.shape[1],
        "width": img_tensor.shape[2],
    }]

    with torch.no_grad():
        _ = model(inputs)

    # Cleanup hooks
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    print(f"Gewähltes {role}-Layer: {name}")
    has_deform = (
        getattr(attn_cache, "deform_sampling_locations", None) is not None and
        getattr(attn_cache, "deform_attention_weights", None) is not None and
        getattr(attn_cache, "deform_spatial_shapes", None) is not None and
        getattr(attn_cache, "deform_level_start_index", None) is not None
    )
    has_mha = getattr(attn_cache, "attn_weights", None) is not None

    if has_deform:
        dsl = attn_cache.deform_sampling_locations
        daw = attn_cache.deform_attention_weights
        ssh = attn_cache.deform_spatial_shapes
        lsi = attn_cache.deform_level_start_index
        print("MSDeformAttn-Capture: OK")
        print(f"  sampling_locations: {tuple(dsl.shape)}")
        print(f"  attention_weights:  {tuple(daw.shape)}")
        print(f"  spatial_shapes:     {tuple(ssh.shape)}")
        print(f"  level_start_index:  {tuple(lsi.shape)}")
    else:
        print("MSDeformAttn-Capture: NICHT vorhanden")
        if has_mha:
            print("  Fallback: nn.MultiheadAttention attn_weights vorhanden → Value-Pfad über MHA möglich.")
        else:
            print("  Weder Deform- noch MHA-Gewichte vorhanden → Encoder-Aussagen wären heuristisch.")
        print("Hinweis: Stelle sicher, dass:")
        print("  - Das gewählte Layer MSDeformAttn enthält (Encoder in MaskDINO)")
        print("  - attach_msdeformattn_capture den richtigen Klassennamen auflöst")
        print("  - Hooks vor dem Forward registriert werden (in diesem Skript bereits der Fall)")


def main():
    ap = argparse.ArgumentParser(description="Prüfe MSDeformAttn-Capture im gewählten Layer")
    ap.add_argument("--images", required=True, help="Ordner mit Bildern")
    ap.add_argument("--module", choices=["encoder", "decoder"], default="encoder")
    ap.add_argument("--layer", type=int, default=1, help="1-basierter Layerindex im gewählten Modul")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    run(args.images, which_module=args.module, layer_index=args.layer, device=args.device)


if __name__ == "__main__":
    main()
