from typing import List, Dict, Any, Tuple, Optional
import torch
import re
import numpy as np
import cv2
import os
import csv
import json
import warnings
warnings.filterwarnings("ignore", message=".*torch\\.meshgrid.*indexing.*", category=UserWarning)


# wir wollen die Gewichte pro Bild pro Feature speichern.
# 1. Hook für die Gewichtsextraktion einrichten
# 2. Gewichtsextraktion implementieren
# 3. Räumliche Rekonstruierbarkeit abspeichern
# 4. Ergebnisse speichern

# 1. Hook für die Gewichtsextraktion einrichten

def _register_transformer_encoder_hooks(model: torch.nn.Module) -> Tuple[Dict[str, torch.Tensor], List[Any]]:
    feature_buffers: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []

    target_prefix = "sem_seg_head.pixel_decoder.transformer.encoder.layers"

    def make_hook(key: str):
        def hook_fn(_m, _inp, out):
            # Einheitliche Ablage: immer CPU, ohne Gradienten
            try:
                feature_buffers[key] = out.detach().cpu()
            except Exception:
                # Fallback: falls out ein Tuple/Liste ist, nimm das erste Element
                if isinstance(out, (tuple, list)) and len(out) > 0:
                    feature_buffers[key] = out[0].detach().cpu()
                else:
                    feature_buffers[key] = torch.as_tensor(out).detach().cpu()
        return hook_fn

    matched = 0
    for name, module in model.named_modules():
        # Exakt Layer-Module treffen, nicht deren Untermodule
        # Beispielname: "...encoder.layers.0"
        if name.startswith(target_prefix + "."):
            suffix = name[len(target_prefix) + 1:]
            if suffix.isdigit():
                h = module.register_forward_hook(make_hook(name))
                handles.append(h)
                matched += 1

    # Komfort: am Modell ablegen, damit beim nächsten Schritt abrufbar/auflösbar
    setattr(model, "_encoder_feature_buffers", feature_buffers)
    setattr(model, "_encoder_hook_handles", handles)

    print(f"🎯 Encoder-Hooks registriert: {matched} Layer gefunden")
    if matched == 0:
        print("⚠️ Keine passenden Encoder-Layer gefunden. Prüfe die Modulpfade im Modell.")

    return feature_buffers, handles


def _infer_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")

# 2. Gewichtsextraktion implementieren

def _preprocess_image(image_path: str, input_format: str = "RGB", min_size: int = 800, max_size: int = 1333):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
    if input_format.upper() == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = min(min_size / min(h, w), max_size / max(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(img_resized).permute(2, 0, 1).to(dtype=torch.float16)  # [C,H,W], 0-255 float16
    return tensor, (nh, nw), (h, w), img_resized

# Hook für spatial_shapes und level_start_index am Encoder registrieren

def _register_encoder_shapes_hook(model: torch.nn.Module) -> Optional[Any]:
    target_mod_name = None
    target_mod = None
    for name, module in model.named_modules():
        if name.endswith("sem_seg_head.pixel_decoder.transformer") or name.endswith("pixel_decoder.transformer"):
            target_mod_name = name
            target_mod = module
            break

    if target_mod is None:
        print("⚠️ Encoder-Transformer-Modul nicht gefunden – Shapes-Hook wird nicht registriert.")
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
    print(f"🎯 Shapes-Hook registriert an: {target_mod_name}")
    return h

# Umsetzung der Gewichtsextraktion 

def accept_weights_model_images(
    weights_path: str,
    model: torch.nn.Module,
    image_list: List[str],
    base_out_layers: Optional[str] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    print("📦 Gewichte-Pfad:", weights_path)
    print("🧠 Modell-Typ:", type(model).__name__)
    print("🖼️  #Bilder:", len(image_list))
    
    # 1. Hooks für alle Transformer-Encoder-Layer registrieren (analog zu network_dissection_encoder.py)
    feature_buffers, handles = _register_transformer_encoder_hooks(model)
    print(f"🔌 Aktive Hook-Handles: {len(handles)}")
    print("ℹ️  Beim nächsten Forward-Pass werden die Encoder-Features in 'model._encoder_feature_buffers' gesammelt.")

    # 1.1 Hook für spatial_shapes / level_start_index am Encoder registrieren
    shapes_handle = _register_encoder_shapes_hook(model)
    if shapes_handle is not None:
        handles.append(shapes_handle)

    # 2. Vorbereitung: Ausgabeordner und CSV-Bereinigung (Duplikate vermeiden)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Globaler Ordner für Layer-CSV-Dateien (eine CSV je Layer)
    if base_out_layers is None:
        base_out_layers = os.path.join(project_root, "output", "encoder")
    os.makedirs(base_out_layers, exist_ok=True)

    def _clear_existing_csvs(base_dir: str) -> None:
        removed = 0
        for root, _dirs, files in os.walk(base_dir):
            for fn in files:
                if fn.lower() == "feature.csv":
                    try:
                        os.remove(os.path.join(root, fn))
                        removed += 1
                    except Exception:
                        pass
        if removed:
            print(f"🧹 Alte CSVs entfernt: {removed} Datei(en)")

    _clear_existing_csvs(base_out_layers)

    # 3) Gewichtsextraktion: Für jedes Bild Forward ausführen und Layer-Ausgaben direkt streamend exportieren
    device = _infer_device(model)
    model_was_training = model.training
    model.eval()
    input_format = getattr(model, "input_format", "RGB")
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
        
    # Header anlegen, wenn nicht vorhanden    
        
    def _ensure_header(csv_path: str, N: int) -> int:
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["Name"] + [f"Gewicht {i+1}" for i in range(N)]
                writer.writerow(header)
            return N
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if header is None:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = ["Name"] + [f"Gewicht {i+1}" for i in range(N)]
                writer.writerow(header)
            return N
        return max(0, len(header) - 1)

    def _append_rows(csv_path: str, names_and_values: List[Tuple[str, List[float]]], N_header: int) -> None:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for name, vals in names_and_values:
                if len(vals) < N_header:
                    vals = vals + [""] * (N_header - len(vals))
                elif len(vals) > N_header:
                    vals = vals[:N_header]
                writer.writerow([name] + vals)

    layer_index_regex = re.compile(r"\.encoder\.layers\.(\d+)")

    try:
        with torch.no_grad():
            for img_path in image_list:
                try:
                    image_tensor, (nh, nw), (h0, w0), _ = _preprocess_image(img_path, input_format=input_format)
                except Exception as e:
                    print(f"❌ Fehler beim Laden/Preprocessing: {img_path} -> {e}")
                    continue

                inputs = [{
                    "image": image_tensor.to(device, dtype=torch.float32),
                    "height": nh,
                    "width": nw,
                }]

                feature_buffers.clear()

                try:
                    _ = model(inputs)
                except Exception as e:
                    print(f"❌ Forward-Fehler bei {img_path}: {e}")
                    continue
                try:
                    shapes_info = getattr(model, "_last_encoder_shapes", None)
                    spatial_shapes = None
                    level_start_index = None
                    if isinstance(shapes_info, dict):
                        spatial_shapes = shapes_info.get("spatial_shapes")
                        level_start_index = shapes_info.get("level_start_index")
                    if spatial_shapes is None or level_start_index is None:
                        print("⚠️ Konnte spatial_shapes/level_start_index nicht erfassen – JSON wird übersprungen.")
                    else:
                        n_tokens = int(sum(int(h) * int(w) for (h, w) in spatial_shapes))
                        image_id = os.path.splitext(os.path.basename(img_path))[0]
                        base_out_image = os.path.join(base_out_layers, image_id)
                        os.makedirs(base_out_image, exist_ok=True)
                        json_path = os.path.join(base_out_image, "shapes.json")
                        payload = {
                            "image_id": image_id,
                            "image_path": img_path,
                            "orig_size": [int(h0), int(w0)],
                            "input_size": [int(nh), int(nw)],
                            "spatial_shapes": [[int(h), int(w)] for (h, w) in spatial_shapes],
                            "level_start_index": [int(x) for x in level_start_index],
                            "N_tokens": n_tokens,
                        }
                        with open(json_path, "w", encoding="utf-8") as jf:
                            json.dump(payload, jf, ensure_ascii=False, indent=2)
                        print(f"💾 Shapes-JSON gespeichert: {json_path}")
                except Exception as e:
                    print(f"⚠️ Shapes-JSON konnte nicht geschrieben werden: {e}")
                image_id = os.path.splitext(os.path.basename(img_path))[0]

                for layer_name, tensor in feature_buffers.items():
                    try:
                        arr = tensor.numpy()
                    except Exception:
                        arr = np.asarray(tensor.cpu())
                    arr = arr.astype(np.float16, copy=False)

                    m = layer_index_regex.search(layer_name)
                    if not m:
                        continue
                    lidx = int(m.group(1))
                    B, D, N, out = to_bdn(arr)
                    if D != 256:
                        print(f"⚠️ Layer {lidx}: Feature-Dim={D}≠256 – exportiere dennoch.")
                    if B != 1:
                        print(f"ℹ️ Layer {lidx}: Batchgröße {B} – exportiere nur b=0.")

                    layer_dir = os.path.join(base_out_layers, f"layer{lidx}")
                    os.makedirs(layer_dir, exist_ok=True)
                    csv_path = os.path.join(layer_dir, "feature.csv")

                    N_header = _ensure_header(csv_path, N)

                    rows: List[Tuple[str, List[float]]] = []
                    for fidx in range(D):
                        name = f"{image_id}, Feature{fidx+1}"
                        values = out[0, fidx, :].astype(np.float32).tolist()
                        rows.append((name, values))

                    _append_rows(csv_path, rows, N_header)
                    del arr, out, rows
                feature_buffers.clear()
                import gc as _gc
                _gc.collect()

                print(f"✅ Features extrahiert: {img_path} | Layer: {len(handles)-1}")

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        if model_was_training:
            model.train()

    print("📊 Extraktion abgeschlossen (streaming).")
    return {}
