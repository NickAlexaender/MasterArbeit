from typing import List, Dict, Any, Tuple, Optional
import torch
import numpy as np
import cv2
import os
import csv
import json
import re
import warnings
warnings.filterwarnings("ignore", message=".*torch\\.meshgrid.*indexing.*", category=UserWarning)


# Konstanten

LABEL_NONE = 0
LABEL_GRAU = 1
LABEL_ORANGE = 2
LABEL_BLAU = 3
LABEL_GRAU_ORANGE = 4
LABEL_GRAU_BLAU = 5
LABEL_ORANGE_BLAU = 6
LABEL_ALL = 7  # Alle drei Farben

OVERLAP_THRESHOLD = 0.5  # 50% Mindestüberlappung für Label-Zuweisung

# Laden der Farbkonzept-Maske und wandle sie in eine Binärmaske um

def load_color_concept_mask(mask_path: str) -> Optional[np.ndarray]:
    if not os.path.exists(mask_path):
        return None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    
    binary_mask = (mask > 0).astype(np.uint8)
    return binary_mask

# Berechnen aller Labels für alle Patches basierend auf Farbkonzept-Masken

def compute_patch_labels(
    grau_mask: Optional[np.ndarray],
    orange_mask: Optional[np.ndarray],
    spatial_shapes: List[Tuple[int, int]],
    input_size: Tuple[int, int],
    mask_size: Tuple[int, int] = (256, 256),
    blau_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    total_tokens = sum(h * w for h, w in spatial_shapes)
    labels = np.zeros(total_tokens, dtype=np.int32)
    
    if grau_mask is None and orange_mask is None and blau_mask is None:
        return labels
    
    input_h, input_w = input_size
    mask_h, mask_w = mask_size
    
    token_idx = 0
    for level_idx, (feat_h, feat_w) in enumerate(spatial_shapes):
        scale_h = mask_h / input_h
        scale_w = mask_w / input_w
        
        patch_h_input = input_h / feat_h
        patch_w_input = input_w / feat_w
        
        patch_h_mask = patch_h_input * scale_h
        patch_w_mask = patch_w_input * scale_w
        
        for i in range(feat_h):
            for j in range(feat_w):
                y_start = int(i * patch_h_mask)
                y_end = int(min((i + 1) * patch_h_mask, mask_h))
                x_start = int(j * patch_w_mask)
                x_end = int(min((j + 1) * patch_w_mask, mask_w))
                
                patch_area = max((y_end - y_start) * (x_end - x_start), 1)
                
                is_grau = False
                is_orange = False
                is_blau = False
                if grau_mask is not None:
                    grau_overlap = np.sum(grau_mask[y_start:y_end, x_start:x_end])
                    if grau_overlap / patch_area >= OVERLAP_THRESHOLD:
                        is_grau = True
                if orange_mask is not None:
                    orange_overlap = np.sum(orange_mask[y_start:y_end, x_start:x_end])
                    if orange_overlap / patch_area >= OVERLAP_THRESHOLD:
                        is_orange = True
                if blau_mask is not None:
                    blau_overlap = np.sum(blau_mask[y_start:y_end, x_start:x_end])
                    if blau_overlap / patch_area >= OVERLAP_THRESHOLD:
                        is_blau = True
                if is_grau and is_orange and is_blau:
                    labels[token_idx] = LABEL_ALL
                elif is_grau and is_orange:
                    labels[token_idx] = LABEL_GRAU_ORANGE
                elif is_grau and is_blau:
                    labels[token_idx] = LABEL_GRAU_BLAU
                elif is_orange and is_blau:
                    labels[token_idx] = LABEL_ORANGE_BLAU
                elif is_grau:
                    labels[token_idx] = LABEL_GRAU
                elif is_orange:
                    labels[token_idx] = LABEL_ORANGE
                elif is_blau:
                    labels[token_idx] = LABEL_BLAU
                else:
                    labels[token_idx] = LABEL_NONE
                
                token_idx += 1
    
    return labels

# Veraltet: Verwendet globale Labels für Queries.

def compute_patch_labels_for_queries(
    grau_mask: Optional[np.ndarray],
    orange_mask: Optional[np.ndarray],
    num_queries: int,
    mask_size: Tuple[int, int] = (256, 256),
    blau_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    labels = np.zeros(num_queries, dtype=np.int32)
    
    mask_area = mask_size[0] * mask_size[1]
    
    grau_coverage = 0
    orange_coverage = 0
    blau_coverage = 0
    
    if grau_mask is not None:
        grau_coverage = np.sum(grau_mask) / mask_area
    
    if orange_mask is not None:
        orange_coverage = np.sum(orange_mask) / mask_area
    
    if blau_mask is not None:
        blau_coverage = np.sum(blau_mask) / mask_area
    
    is_grau = grau_coverage >= 0.10
    is_orange = orange_coverage >= 0.10
    is_blau = blau_coverage >= 0.10
    
    if is_grau and is_orange and is_blau:
        labels[:] = LABEL_ALL
    elif is_grau and is_orange:
        labels[:] = LABEL_GRAU_ORANGE
    elif is_grau and is_blau:
        labels[:] = LABEL_GRAU_BLAU
    elif is_orange and is_blau:
        labels[:] = LABEL_ORANGE_BLAU
    elif is_grau:
        labels[:] = LABEL_GRAU
    elif is_orange:
        labels[:] = LABEL_ORANGE
    elif is_blau:
        labels[:] = LABEL_BLAU
    
    return labels

# Berechnet Labels für Decoder-Queries basierend auf vorhergesagten Masken

def compute_query_labels_from_masks(
    pred_masks: np.ndarray,
    grau_mask: Optional[np.ndarray],
    orange_mask: Optional[np.ndarray],
    iou_threshold: float = 0.1,
    blau_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    num_queries = pred_masks.shape[0]
    labels = np.zeros(num_queries, dtype=np.int32)
    
    # Resize Konzept-Masken auf Größe der Prediction-Masken
    h, w = pred_masks.shape[1], pred_masks.shape[2]
    
    if grau_mask is not None:
        grau_resized = cv2.resize(grau_mask.astype(np.float32), (w, h)) > 0.5
    else:
        grau_resized = None
        
    if orange_mask is not None:
        orange_resized = cv2.resize(orange_mask.astype(np.float32), (w, h)) > 0.5
    else:
        orange_resized = None
    
    if blau_mask is not None:
        blau_resized = cv2.resize(blau_mask.astype(np.float32), (w, h)) > 0.5
    else:
        blau_resized = None
    
    for q_idx in range(num_queries):
        query_mask = pred_masks[q_idx] > 0.5  # Binarisieren
        query_area = np.sum(query_mask)
        
        if query_area == 0:
            labels[q_idx] = LABEL_NONE
            continue
        
        grau_overlap = 0.0
        orange_overlap = 0.0
        blau_overlap = 0.0
        
        if grau_resized is not None:
            intersection = np.sum(query_mask & grau_resized)
            grau_overlap = intersection / query_area if query_area > 0 else 0
        if orange_resized is not None:
            intersection = np.sum(query_mask & orange_resized)
            orange_overlap = intersection / query_area if query_area > 0 else 0
        if blau_resized is not None:
            intersection = np.sum(query_mask & blau_resized)
            blau_overlap = intersection / query_area if query_area > 0 else 0
        is_grau = grau_overlap >= iou_threshold
        is_orange = orange_overlap >= iou_threshold
        is_blau = blau_overlap >= iou_threshold
        
        if is_grau and is_orange and is_blau:
            labels[q_idx] = LABEL_ALL
        elif is_grau and is_orange:
            labels[q_idx] = LABEL_GRAU_ORANGE
        elif is_grau and is_blau:
            labels[q_idx] = LABEL_GRAU_BLAU
        elif is_orange and is_blau:
            labels[q_idx] = LABEL_ORANGE_BLAU
        elif is_grau:
            labels[q_idx] = LABEL_GRAU
        elif is_orange:
            labels[q_idx] = LABEL_ORANGE
        elif is_blau:
            labels[q_idx] = LABEL_BLAU
        else:
            labels[q_idx] = LABEL_NONE
    
    return labels


# Encoder spezifischer Stuff

def register_encoder_hooks(model: torch.nn.Module) -> Tuple[Dict[str, torch.Tensor], List[Any]]:
    feature_buffers: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []
    
    target_prefix = "sem_seg_head.pixel_decoder.transformer.encoder.layers"
    
    def make_hook(key: str):
        def hook_fn(_m, _inp, out):
            try:
                feature_buffers[key] = out.detach().cpu()
            except Exception:
                if isinstance(out, (tuple, list)) and len(out) > 0:
                    feature_buffers[key] = out[0].detach().cpu()
                else:
                    feature_buffers[key] = torch.as_tensor(out).detach().cpu()
        return hook_fn
    
    matched = 0
    for name, module in model.named_modules():
        if name.startswith(target_prefix + "."):
            suffix = name[len(target_prefix) + 1:]
            if suffix.isdigit():
                h = module.register_forward_hook(make_hook(name))
                handles.append(h)
                matched += 1
    
    print(f"🎯 Encoder-Hooks registriert: {matched} Layer gefunden")
    return feature_buffers, handles


def register_encoder_shapes_hook(model: torch.nn.Module) -> Optional[Any]:

    target_mod = None
    for name, module in model.named_modules():
        if name.endswith("sem_seg_head.pixel_decoder.transformer") or name.endswith("pixel_decoder.transformer"):
            target_mod = module
            break
    
    if target_mod is None:
        return None
    
    def hook_fn(_m, _inp, out):
        spatial_shapes = None
        level_start_index = None
        
        try:
            if isinstance(out, (tuple, list)) and len(out) >= 3:
                spatial_shapes = out[1]
                level_start_index = out[2]
        except Exception:
            pass
        
        if spatial_shapes is None or level_start_index is None:
            try:
                if isinstance(_inp, (tuple, list)) and len(_inp) >= 1:
                    srcs = _inp[0]
                    shapes = []
                    for t in srcs:
                        if isinstance(t, torch.Tensor) and t.dim() == 4:
                            shapes.append((int(t.shape[2]), int(t.shape[3])))
                    if shapes:
                        spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=t.device)
                        starts = [0]
                        acc = 0
                        for (hh, ww) in shapes:
                            acc += int(hh) * int(ww)
                            starts.append(acc)
                        level_start_index = torch.as_tensor(starts[:-1], dtype=torch.long, device=t.device)
            except Exception:
                pass
        
        if spatial_shapes is not None and level_start_index is not None:
            try:
                data = {
                    "spatial_shapes": spatial_shapes.detach().cpu().tolist(),
                    "level_start_index": level_start_index.detach().cpu().tolist(),
                }
                setattr(model, "_last_encoder_shapes", data)
            except Exception:
                pass
    
    h = target_mod.register_forward_hook(hook_fn)
    return h


# Decoder spezifischer Stuff

def register_decoder_hooks(model: torch.nn.Module) -> Tuple[Dict[str, List[torch.Tensor]], List[Any]]:
    hidden_states_per_layer: Dict[str, List[torch.Tensor]] = {}
    handles: List[Any] = []
    
    def create_hook(layer_name: str):
        def hook_fn(module, input, output):
            if layer_name not in hidden_states_per_layer:
                hidden_states_per_layer[layer_name] = []
            hidden_states = output[0] if isinstance(output, tuple) else output
            hidden_states_per_layer[layer_name].append(hidden_states.detach().clone())
        return hook_fn
    
    decoder_count = 0
    for name, module in model.named_modules():
        if 'predictor.decoder.layers.' in name and name.count('.') == 4:
            if name.endswith('.layers.0') or name.endswith('.layers.1') or name.endswith('.layers.2'):
                handle = module.register_forward_hook(create_hook(name))
                handles.append(handle)
                decoder_count += 1
                print(f"Hook registriert: {name}")
    
    print(f"✅ {decoder_count} Decoder-Layer gefunden und registriert.")
    return hidden_states_per_layer, handles


# Image Preprocessing

def preprocess_image(image_path: str, input_format: str = "RGB", 
                    min_size: int = 800, max_size: int = 1333) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    
    if input_format.upper() == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    scale = min(min_size / min(h, w), max_size / max(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    
    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).to(dtype=torch.float32)
    return tensor, (nh, nw), (h, w)


# Feature-Tensor-Normalisierung

def to_bdn(arr: np.ndarray) -> Tuple[int, int, int, np.ndarray]:
    x = arr
    if x.ndim == 4:
        B, C, H, W = x.shape
        if C != 256 and x.shape[-1] == 256:
            x = np.transpose(x, (0, 3, 1, 2))
            B, C, H, W = x.shape
        D = C
        N = H * W
        out = x.reshape(B, D, N)
        return B, D, N, out
    elif x.ndim == 3:
        s = list(x.shape)
        if 256 in s:
            d_axis = s.index(256)
        else:
            d_axis = 2
        axes = [0, 1, 2]
        axes.remove(d_axis)
        b_axis = axes[0] if x.shape[axes[0]] <= x.shape[axes[1]] else axes[1]
        n_axis = axes[1] if b_axis == axes[0] else axes[0]
        x_bnd = np.transpose(x, (b_axis, n_axis, d_axis))
        B, N, D = x_bnd.shape
        out = np.transpose(x_bnd, (0, 2, 1))
        return B, D, N, out
    elif x.ndim == 2:
        if x.shape[1] == 256:
            N, D = x.shape
            out = x.T[None, ...]
            return 1, D, N, out
        else:
            D, N = x.shape
            out = x[None, ...]
            return 1, D, N, out
    else:
        flat = x.reshape(1, -1)
        N = flat.shape[1]
        out = flat[None, ...]
        return 1, 1, N, out


# Exportieren der CSVs

def ensure_header(csv_path: str, num_features: int) -> int:

    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["Name", "Label"] + [f"Gewicht {i+1}" for i in range(num_features)]
            writer.writerow(header)
        return num_features
    
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
    
    if header is None:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["Name", "Label"] + [f"Gewicht {i+1}" for i in range(num_features)]
            writer.writerow(header)
        return num_features
    
    return max(0, len(header) - 2)  # -2 für Name und Label


def append_rows_to_csv(csv_path: str, rows: List[Tuple[str, int, List[float]]], num_features: int) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for name, label, features in rows:
            if len(features) < num_features:
                features = features + [0.0] * (num_features - len(features))
            elif len(features) > num_features:
                features = features[:num_features]
            writer.writerow([name, label] + features)


# Kernfunktionen für Encoder/Decoder Linear Probing

# Extrahiert Encoder-Features mit Patch-Labels für Linear Probing

def extract_encoder_features_with_labels(
    model: torch.nn.Module,
    image_path: str,
    image_id: str,
    grau_mask_path: Optional[str],
    orange_mask_path: Optional[str],
    output_dir: str,
    layer: Optional[str] = None,
    blau_mask_path: Optional[str] = None,
) -> bool:
    grau_mask = load_color_concept_mask(grau_mask_path) if grau_mask_path else None
    orange_mask = load_color_concept_mask(orange_mask_path) if orange_mask_path else None
    blau_mask = load_color_concept_mask(blau_mask_path) if blau_mask_path else None  
    feature_buffers, handles = register_encoder_hooks(model)
    shapes_handle = register_encoder_shapes_hook(model)
    if shapes_handle:
        handles.append(shapes_handle)
    
    try:
        image_tensor, input_size, orig_size = preprocess_image(image_path)
        device = next(model.parameters()).device
        
        inputs = [{
            "image": image_tensor.to(device),
            "height": input_size[0],
            "width": input_size[1],
        }]
        
        model.eval()
        with torch.no_grad():
            _ = model(inputs)
        
        
        shapes_info = getattr(model, "_last_encoder_shapes", None)
        if shapes_info is None:
            print(f"⚠️ Keine Shapes für {image_id}")
            return False
        
        spatial_shapes = shapes_info["spatial_shapes"]
        labels = compute_patch_labels(
            grau_mask, orange_mask,
            spatial_shapes, input_size,
            mask_size=(256, 256) if grau_mask is not None else (256, 256),
            blau_mask=blau_mask,
        )
        

        layer_regex = re.compile(r"\.encoder\.layers\.(\d+)")
        

        target_layer_idx = None
        if layer is not None:
            layer_match = re.search(r'(\d+)', layer)
            if layer_match:
                target_layer_idx = int(layer_match.group(1))
        
        for layer_name, tensor in feature_buffers.items():
            m = layer_regex.search(layer_name)
            if not m:
                continue
            
            layer_idx = int(m.group(1))
            
            if target_layer_idx is not None and layer_idx != target_layer_idx:
                continue
            
            arr = tensor.numpy().astype(np.float16)
            B, D, N, out = to_bdn(arr)
            

            layer_dir = os.path.join(output_dir, f"layer{layer_idx}")
            os.makedirs(layer_dir, exist_ok=True)
            csv_path = os.path.join(layer_dir, "patches.csv")
            
            num_features = ensure_header(csv_path, D)
            
            rows = []
            for patch_idx in range(N):
                name = f"{image_id}, Patch{patch_idx + 1}"
                label = int(labels[patch_idx]) if patch_idx < len(labels) else LABEL_NONE
                features = out[0, :, patch_idx].astype(np.float32).tolist()
                rows.append((name, label, features))
            
            append_rows_to_csv(csv_path, rows, num_features)
            print(f"✅ Layer {layer_idx}: {N} Patches exportiert für {image_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei {image_id}: {e}")
        return False
    
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        feature_buffers.clear()

# Extrahiert Decoder-Query-Features mit Labels für Linear Probing

def extract_decoder_features_with_labels(
    model: torch.nn.Module,
    image_path: str,
    image_id: str,
    grau_mask_path: Optional[str],
    orange_mask_path: Optional[str],
    output_dir: str,
    num_queries: int = 300,
    iou_threshold: float = 0.1,
    blau_mask_path: Optional[str] = None,
) -> bool:
    grau_mask = load_color_concept_mask(grau_mask_path) if grau_mask_path else None
    orange_mask = load_color_concept_mask(orange_mask_path) if orange_mask_path else None
    blau_mask = load_color_concept_mask(blau_mask_path) if blau_mask_path else None
    
    hidden_states_per_layer, handles = register_decoder_hooks(model)
    
    raw_decoder_output = {}
    def raw_mask_hook(module, inp, out):
        if isinstance(out, tuple) and len(out) >= 1 and isinstance(out[0], dict):
            if 'pred_masks' in out[0]:
                raw_decoder_output['pred_masks'] = out[0]['pred_masks'].detach().clone()
    

    raw_mask_handle = None
    for name, mod in model.named_modules():
        if name == 'sem_seg_head.predictor':
            raw_mask_handle = mod.register_forward_hook(raw_mask_hook)
            handles.append(raw_mask_handle)
            break
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img_rgb.shape[:2]
        

        img_resized = cv2.resize(img_rgb, (800, 800))
        
        image_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        batched_input = {
            "image": image_tensor,
            "height": orig_h,
            "width": orig_w,
            "file_name": os.path.basename(image_path)
        }
        

        model.eval()
        with torch.no_grad():
            outputs = model([batched_input])
        
        pred_masks = None
        if 'pred_masks' in raw_decoder_output:
            # Shape: [1, 300, H, W] -> [300, H, W]
            pred_masks = raw_decoder_output['pred_masks'][0].cpu().numpy()
            print(f"   → Rohe Masken extrahiert: {pred_masks.shape}")
        
        if pred_masks is not None and pred_masks.shape[0] > 0:
            labels = compute_query_labels_from_masks(
                pred_masks, grau_mask, orange_mask, iou_threshold, blau_mask=blau_mask
            )
            unique, counts = np.unique(labels, return_counts=True)
            label_stats = dict(zip(unique, counts))
            print(f"   → Label-Verteilung: bg={label_stats.get(0, 0)}, grau={label_stats.get(1, 0)}, orange={label_stats.get(2, 0)}, blau={label_stats.get(3, 0)}, g+o={label_stats.get(4, 0)}, g+b={label_stats.get(5, 0)}, o+b={label_stats.get(6, 0)}, alle={label_stats.get(7, 0)}")
        else:
            labels = compute_patch_labels_for_queries(
                grau_mask, orange_mask, num_queries,
                mask_size=(256, 256),
                blau_mask=blau_mask,
            )
            print(f"   → Keine Masken verfügbar, verwende globale Labels")
        
        for layer_name, hidden_states_list in hidden_states_per_layer.items():
            if not hidden_states_list:
                continue
            
            hidden_states = hidden_states_list[-1]
            
            if len(hidden_states.shape) == 3:
                actual_num_queries, batch_size, hidden_dim = hidden_states.shape
                if batch_size == 1:
                    queries = hidden_states[:, 0, :].cpu().numpy()
                else:
                    continue
            elif len(hidden_states.shape) == 2:
                queries = hidden_states.cpu().numpy()
            else:
                continue
            
            layer_number = "unknown"
            if "layers.0" in layer_name:
                layer_number = "layer0"
            elif "layers.1" in layer_name:
                layer_number = "layer1"
            elif "layers.2" in layer_name:
                layer_number = "layer2"
            else:
                match = re.search(r'layers[._](\d+)', layer_name)
                if match:
                    layer_number = f"layer{match.group(1)}"
            
            layer_dir = os.path.join(output_dir, layer_number)
            os.makedirs(layer_dir, exist_ok=True)
            csv_path = os.path.join(layer_dir, "queries.csv")
            
            num_actual_queries, hidden_dim = queries.shape
            num_features = ensure_header(csv_path, hidden_dim)
            
            label_counts = {i: 0 for i in range(8)}
            
            rows = []
            for query_idx in range(num_actual_queries):
                name = f"{image_id}, Query{query_idx + 1}"
                label = int(labels[query_idx]) if query_idx < len(labels) else LABEL_NONE
                label_counts[label] = label_counts.get(label, 0) + 1
                features = queries[query_idx, :].astype(np.float32).tolist()
                rows.append((name, label, features))
            
            append_rows_to_csv(csv_path, rows, num_features)
            
            label_info = f"bg:{label_counts[0]} grau:{label_counts[1]} orange:{label_counts[2]} blau:{label_counts[3]} g+o:{label_counts[4]} g+b:{label_counts[5]} o+b:{label_counts[6]} alle:{label_counts[7]}"
            print(f"✅ {layer_number}: {num_actual_queries} Queries ({label_info})")
        
        return True
        
    except Exception as e:
        print(f"❌ Fehler bei {image_id}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        for layer_states in hidden_states_per_layer.values():
            layer_states.clear()
