from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*torch.meshgrid.*")
from myThesis.lrp.do.compat import *  # Pillow/NumPy monkey patches
import gc
import logging
import os
import random
from typing import List, Optional
import numpy as np
import pandas as pd
import PIL.Image
import torch
from torch import Tensor
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger as setup_d2_logger
from myThesis.lrp.do.cli import parse_args
from myThesis.lrp.do.config import (
    DETERMINISTIC,
    LN_RULE,
    MEASUREMENT_POINT,
    SEED,
    TARGET_TOKEN_IDX,
    USE_SUBLAYER,
)
from myThesis.lrp.do.config_build import DEFAULT_WEIGHTS, build_cfg_for_inference
from myThesis.lrp.do.io_utils import collect_images
from myThesis.lrp.do.lrp_controller import LRPController
from myThesis.lrp.do.lrp_analysis import LRPAnalysisContext
from myThesis.lrp.do.model_graph_wrapper import (
    LayerType,
    ModelGraph,
    list_decoder_like_layers,
    list_encoder_like_layers,
)
from myThesis.lrp.do.tensor_ops import aggregate_channel_relevance, build_target_relevance
from myThesis.model_config import get_model_config, get_classes, get_dataset_name

logger = logging.getLogger("lrp.calc")

_MODEL_CACHE: dict = {}
_FORWARD_PASS_CACHE: dict = {}



def _setup_determinism(seed: int = SEED, strict: bool = DETERMINISTIC) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            logger.warning("Deterministische Algorithmen konnten nicht aktiviert werden.")


def _register_model_metadata(model: str = "car") -> None:
    try:
        config = get_model_config(model)
        dataset_name = config["dataset_name"]
        classes = config["classes"]
        MetadataCatalog.get(f"{dataset_name}_minimal").set(thing_classes=classes)
    except Exception:
        pass  # Metadaten sind optional


def _normalize_tensor(R: Tensor, mode: str) -> Tensor:
    if mode == "sum1":
        return R / (R.sum() + 1e-12)
    elif mode == "sumAbs1":
        return R / (R.abs().sum() + 1e-12)
    return R


def _aggregate_relevance(vec: Tensor, axis: str) -> Tensor:
    if axis == "channel":
        return aggregate_channel_relevance(vec)
    
    # Token/Query-Aggregation
    # Für Decoder: vec hat Shape (Q, B, C) = (300, 1, 256)
    # Für Encoder: vec hat Shape (T, B, C) = (tokens, 1, 256)
    # Wir wollen einen Vektor der Länge Q bzw. T zurückgeben
    with torch.no_grad():
        if vec.dim() == 3:
            # Erkenne das Format anhand der Batch-Dimension
            # Format (Q/T, B, C) -> B ist typischerweise 1 und an Position 1
            if vec.shape[1] == 1 or vec.shape[1] < vec.shape[0]:
                # Format ist (Q, B, C) - summiere über Batch und Channels
                return vec.sum(dim=(1, 2))  # Ergebnis: (Q,) = (300,)
            else:
                # Format könnte (B, T, C) sein - summiere über Batch und Channels
                return vec.sum(dim=(0, 2))  # Ergebnis: (T,)
        if vec.dim() == 2:
            # (T, C) oder (Q, C) -> über Channels summieren
            return vec.sum(dim=1)
        if vec.dim() == 1:
            return vec
        # Fallback: Token-Achse an Position 0 annehmen (Q, ...)
        try:
            Qdim = vec.shape[0]
            return vec.reshape(Qdim, -1).sum(dim=1)
        except Exception:
            return vec.reshape(-1)

# Wir wollen das Modell machmal aus dem Cach laden/holen

def _get_or_load_model(weights_path: str, device: str = "cpu", use_cache: bool = True, model_type: str = "car"):
    global _MODEL_CACHE
    
    cache_key = (weights_path, device, model_type)
    
    if use_cache and cache_key in _MODEL_CACHE:
        logger.debug("Verwende gecachtes Modell")
        return _MODEL_CACHE[cache_key]
    
    # Modell neu laden
    cfg = build_cfg_for_inference(device=device, weights_path=weights_path, model=model_type)
    
    # Logger nur einmal initialisieren und auf WARNING setzen um "Loading from..." zu unterdrücken
    if not logging.getLogger("detectron2").handlers:
        setup_d2_logger()
    logging.getLogger("detectron2").setLevel(logging.WARNING)
    logging.getLogger("fvcore").setLevel(logging.WARNING)
    
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()
    model.to(device)
    
    if use_cache:
        _MODEL_CACHE[cache_key] = (model, cfg)
        # logger.info(f"Modell gecacht für: {weights_path}")
    
    return model, cfg

# Speicher muss frei gegeben werden

def clear_model_cache():
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Modell-Cache geleert")


def clear_forward_pass_cache():
    global _FORWARD_PASS_CACHE
    _FORWARD_PASS_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Forward Pass Cache geleert")

# aufteilung von Bildern in Branches

def _prepare_batched_inputs_from_images(
    img_files: List[str],
    cfg,
    batch_size: int = 4,
    device: str = "cpu"
) -> List[List[dict]]:
    resize_aug = T.ResizeShortestEdge(
        short_edge_length=getattr(cfg.INPUT, "MIN_SIZE_TEST", 800),
        max_size=getattr(cfg.INPUT, "MAX_SIZE_TEST", 1333),
    )
    
    num_images = len(img_files)
    num_batches = (num_images + batch_size - 1) // batch_size
    all_batches = []
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_images)
        batch_paths = img_files[batch_start:batch_end]
        
        batched_inputs = []
        for img_path in batch_paths:
            try:
                pil_im = PIL.Image.open(img_path).convert("RGB")
                original_rgb = np.array(pil_im)
                original_h, original_w = original_rgb.shape[:2]
                
                if cfg.INPUT.FORMAT == "RGB":
                    model_input = original_rgb
                else:
                    model_input = original_rgb[:, :, ::-1]  # RGB -> BGR
                
                tfm = resize_aug.get_transform(model_input)
                model_input = tfm.apply_image(model_input)
                
                image_tensor = torch.as_tensor(model_input.astype("float32").transpose(2, 0, 1))
                image_tensor = image_tensor.to(device)
                
                batched_inputs.append({
                    "image": image_tensor,
                    "height": original_h,
                    "width": original_w,
                })
            except Exception as e:
                logger.exception(f"Fehler beim Laden von {img_path}: {e}")
                continue
        
        if batched_inputs:
            all_batches.append(batched_inputs)
    
    return all_batches

# Führt die vollständige LRP-Analyse durch.
# Diese Funktion lädt das MaskDINO-Modell, verarbeitet alle Bilder in einem Verzeichnis, berechnet LRP-Attributionen für ein spezifisches Layer/Feature, und exportiert die Ergebnisse als CSV.

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
    weights_path: Optional[str] = None,
    verbose: bool = True,
    num_queries: int = 300,
    use_model_cache: bool = True,
    model_type: str = "car",
    batch_size: int = 1,
) -> pd.DataFrame:
    print("LRP startet")
    # logger.info("Starte LRP/Attribution-Analyse...")
    
    # Gewichte validieren
    chosen_weights = weights_path if weights_path else DEFAULT_WEIGHTS
    if not os.path.exists(chosen_weights):
        raise FileNotFoundError(f"Gewichtsdatei nicht gefunden: {chosen_weights}")
    
    # Methode validieren
    if method != "lrp":
        raise ValueError("Nur method='lrp' wird unterstützt.")
    
    # Modell laden (mit Cache für Wiederverwendung)
    device = "cpu"
    model, cfg = _get_or_load_model(chosen_weights, device=device, use_cache=use_model_cache, model_type=model_type)
    
    # Metadata registrieren
    _register_model_metadata(model_type)
    
    # Determinismus
    _setup_determinism()
    # logger.info(f"Determinismus aktiv: {DETERMINISTIC}, Seed={SEED}")
    
    # Layer finden
    if which_module == "decoder":
        with torch.no_grad():
            enc_layers = list_decoder_like_layers(model)
        layer_role = "Decoder"
    else:
        with torch.no_grad():
            # Use ModelGraph to get all encoder nodes
            graph = ModelGraph(model, include_leaf_modules=True)
            all_enc_nodes = graph.get_by_type(LayerType.ENCODER)
            
            # Filter for high-level blocks (TransformerEncoderLayer) to match user expectations for "Layer X"
            # We look for modules that are likely the main blocks, not internal sub-modules
            enc_layers = []
            for node in all_enc_nodes:
                cls_name = type(node.module).__name__
                # MaskDINO uses MSDeformAttnTransformerEncoderLayer
                if "TransformerEncoderLayer" in cls_name or "TransformerDecoderLayer" in cls_name:
                    enc_layers.append((node.name, node.module))
            
            # if no blocks found (unlikely), use all
            if not enc_layers:
                enc_layers = [(n.name, n.module) for n in all_enc_nodes]
                
        layer_role = "Encoder"
    
    if not enc_layers:
        key = "decoder" if which_module == "decoder" else "encoder"
        names = [n for n, _ in model.named_modules() if key in n.lower()]
        raise RuntimeError(
            f"Konnte keine {layer_role}-Layer finden. Gefundene '{key}'-Module: " + ", ".join(names[:20])
        )
    
    # Layer-Index validieren (1-basiert)
    if layer_index <= 0 or layer_index > len(enc_layers):
        msg = (
            f"layer_index {layer_index} ungültig. Es gibt {len(enc_layers)} {layer_role}-Kandidaten.\n"
            + "Gefundene Layer:\n"
            + "\n".join([f"  {i+1}: {n} ({type(m).__name__})" for i, (n, m) in enumerate(enc_layers[:15])])
        )
        raise IndexError(msg)
    
    chosen_name, chosen_layer = enc_layers[layer_index - 1]
    # logger.info(f"Gewähltes {layer_role}-Layer [{layer_index}]: {chosen_name} ({type(chosen_layer).__name__})")
    
    # Index-Achse bestimmen
    if index_kind not in ("auto", "channel", "token"):
        raise ValueError("index_kind muss 'auto', 'channel' oder 'token' sein")
    index_axis = (
        ("token" if which_module == "decoder" else "channel")
        if index_kind == "auto"
        else index_kind
    )
    # logger.info(f"Index-Achse: {index_axis} (index_kind={index_kind})")
    
    # Bilder sammeln
    img_files = collect_images(images_dir)
    if not img_files:
        raise FileNotFoundError(f"Keine Bilder in {images_dir} gefunden")
    
    # logger.info(f"Verarbeite {len(img_files)} Bilder aus: {images_dir}")
    
    # LRP Controller vorbereiten (verbose=False um "Replaced..." und "ModelGraph Summary" zu unterdrücken)
    controller = LRPController(model, device=device, eps=lrp_epsilon, verbose=False)
    controller.prepare(swap_linear=True)
    
    # Decoder und Encoder arbeiten jetzt gleich - Layer-zu-Layer LRP
    # für eine einzelne Query/Feature, die Relevanz auf vorherige Queries/Features verteilt
    compute_all_queries = False  # Alte Logik deaktiviert
    queries_to_compute = 1
    
    # logger.info(f"{layer_role}-Modus: Layer-zu-Layer LRP für Feature/Query {feature_index}")
    
    # Aggregation über Bilder - jetzt mit Query-Dimension für Decoder
    # agg_attr_per_query[q] enthält die aggregierte Relevanz für Query q
    agg_attr_per_query: List[Optional[Tensor]] = [None] * queries_to_compute
    processed = 0
    cache_key = (images_dir, which_module, layer_index, tuple(sorted(img_files)))
    forward_pass_cached = False
    if cache_key in _FORWARD_PASS_CACHE:
        print(f"✓ Nutze gecachte Forward Pass Aktivierungen für {layer_role} Layer {layer_index}")
        forward_pass_cached = True
        cached_data = _FORWARD_PASS_CACHE[cache_key]
        # cached_data wird später verwendet um Backward Pass schneller zu machen
    
    num_images = len(img_files)
    num_batches = (num_images + batch_size - 1) // batch_size
    
    # Lade alle Bilder in Batches (nutzt die neue Hilfsfunktion)
    all_batches = _prepare_batched_inputs_from_images(img_files, cfg, batch_size=batch_size, device=device)
    
    for batch_idx, batched_inputs in enumerate(all_batches):
        # Neue Batch-Nachricht (reduziert Output)
        feature_or_query = f"Feature {feature_index}" if not compute_all_queries else f"Queries 0-{queries_to_compute-1}"
        print(f"LRP auf {layer_role} in Layer {layer_index} mit {feature_or_query} - Batch {batch_idx+1}/{len(all_batches)} ({len(batched_inputs)} Bilder)")
        
        # Verarbeite den gesamten Batch auf einmal
        try:
            # Einheitliche Layer-zu-Layer LRP für beide Module (Encoder und Decoder)
            # chosen_name ist der Layer, von dem aus wir die Relevanz zurückpropagieren wollen
            # feature_index ist der Query-Index (Decoder) oder Kanal-Index (Encoder)
            result = controller.run(
                batched_inputs,
                target_class=None,
                target_query=feature_index if which_module == "decoder" else 0,  # Query-Index für Decoder
                normalize="none",
                which_module=which_module,
                target_feature=feature_index,
                target_layer_name=chosen_name,
            )
                
            if result.R_input is None:
                continue
            
            # Normalisieren
            R_norm = _normalize_tensor(result.R_input, target_norm)
            
            # Aggregieren
            attr = _aggregate_relevance(R_norm, index_axis)
            
            if not torch.isfinite(attr).all():
                continue
            
            if agg_attr_per_query[0] is None:
                agg_attr_per_query[0] = attr.clone()
            else:
                agg_attr_per_query[0] += attr
            
            processed += len(batched_inputs)
                
        except Exception as e:
            logger.exception(f"Fehler bei Batch {batch_idx+1}: {e}")
            continue
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Aufräumen
    controller.cleanup()
    
    if processed == 0 or all(a is None for a in agg_attr_per_query):
        raise RuntimeError("Keine Attributionen konnten berechnet werden.")
    
    # Mittelwert über Bilder
    for q in range(queries_to_compute):
        if agg_attr_per_query[q] is not None:
            agg_attr_per_query[q] = agg_attr_per_query[q] / float(processed)
    
    # Bestimme die Vektorlänge (von der ersten nicht-None Query)
    vec_len = 0
    for q in range(queries_to_compute):
        if agg_attr_per_query[q] is not None:
            vec_len = len(agg_attr_per_query[q])
            break

    # DataFrame erstellen - einheitliches Format für Encoder und Decoder
    # prev_feature_idx zeigt die Relevanz der vorherigen Features/Queries
    agg_attr = agg_attr_per_query[0]
    relevance_np = agg_attr.detach().cpu().numpy()
    
    # Normalisierte Relevanzverteilung berechnen
    abs_sum = np.abs(relevance_np).sum()
    if abs_sum > 1e-12:
        normalized_relevance = relevance_np / abs_sum
    else:
        normalized_relevance = relevance_np
    
    data_dict = {
        "prev_feature_idx": list(range(len(agg_attr))),
        "relevance": relevance_np.tolist(),
        "normalized_relevance": normalized_relevance.tolist(),
        "layer_index": layer_index,
        "layer_name": chosen_name,
        "feature_index": feature_index,
        "epsilon": lrp_epsilon,
        "module_role": layer_role,
        "target_norm": target_norm,
        "index_kind": index_kind,
        "index_axis": index_axis,
        "method": method,
        "num_images": processed,
    }
    
    df = pd.DataFrame(data_dict)
    
    # Nach normalisierter Relevanz absteigend sortieren
    df = df.sort_values(by="normalized_relevance", ascending=False).reset_index(drop=True)
    
    # CSV speichern
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    # logger.info(f"Ergebnisse gespeichert: {output_csv}")
    
    return df


# Wir müssen einen einstiegspunkt bereitstellen, damit wir je nach Modell und co. sachen einstellen können.

def main(
    images_dir: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/image/car/1images",
    layer_index: int = 3,
    feature_index: int = 214,
    target_norm: str = "sum1",
    lrp_epsilon: float = 1e-6,
    output_csv: str = "/Users/nicklehmacher/Alles/MasterArbeit/myThesis/output/lrp_result.csv",
    which_module: str = "encoder",
    method: str = "lrp",
    weights_path: Optional[str] = None,
    verbose: bool = True,
    num_queries: int = 300,
    use_model_cache: bool = True,
    model_type: str = "car",
    batch_size: int = 1,
) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    return run_analysis(
        images_dir=images_dir,
        layer_index=layer_index,
        feature_index=feature_index,
        output_csv=output_csv,
        target_norm=target_norm,
        lrp_epsilon=lrp_epsilon,
        which_module=which_module,
        method=method,
        weights_path=weights_path,
        verbose=verbose,
        num_queries=num_queries,
        use_model_cache=use_model_cache,
        model_type=model_type,
        batch_size=batch_size,
    )

if __name__ == "__main__":
    _args = parse_args()
    main(
        images_dir=_args.images_dir,
        layer_index=_args.layer_index,
        feature_index=_args.feature_index,
        target_norm=_args.target_norm,
        lrp_epsilon=_args.lrp_epsilon,
        output_csv=_args.output_csv,
        which_module=_args.which_module,
        method=_args.method,
        weights_path=getattr(_args, "weights_path", None),
    )
