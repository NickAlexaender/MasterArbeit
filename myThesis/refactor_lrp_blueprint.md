# MaskDINO LRP Refactor Blueprint (`refactor_lrp_blueprint.md`)

**Ziel:** Dies ist ein _konkreter_ Plan inkl. Code‑Skeletons, um deinen aktuellen monolithischen Analyse‑Code in klar getrennte Module zu zerlegen. Der Fokus liegt darauf, **LRP-Backpropagation sauber zu kapseln** (eigene Datei/Modul), damit komplexe Architekturen wie MaskDINO nachvollziehbar und testbar bleiben — ohne „Bullshit‑Heatmaps“.

---

## 1) Zielstruktur (Dateibaum)

```text
myThesis/
└─ attribution/
   ├─ __init__.py
   ├─ cli.py                         # Nur CLI-Argumente & main()
   ├─ runner.py                      # run_analysis() orchestriert alles
   ├─ config.py                      # Konstanten/Defaults/Paths
   ├─ compat.py                      # Pillow/NumPy-Kompatibilitätsfixe
   ├─ data/
   │  ├─ __init__.py
   │  └─ images.py                   # collect_images(), Preprocessing Auginstanz
   ├─ export/
   │  ├─ __init__.py
   │  └─ csv.py                      # write_csv(), top-k logging
   ├─ model/
   │  ├─ __init__.py
   │  ├─ cfg.py                      # build_cfg_for_inference()
   │  ├─ probe.py                    # list_encoder_like_layers(), list_decoder_like_layers()
   │  └─ register_minimal.py         # Dataset-Stub/Metadata
   ├─ utils/
   │  ├─ __init__.py
   │  ├─ tensor.py                   # _to_BTC(), guards, finite checks
   │  ├─ hooks.py                    # _LayerTap
   │  └─ logging.py                  # setup_logging()
   └─ lrp/
      ├─ __init__.py
      ├─ interface.py                # LRPConfig, Rule-Auswahl, Public API-Typen
      ├─ tracer.py                   # LRPTracer (x/y-Caching, Hook mgmt)
      ├─ logic.py                    # ★ reine LRP-Regeln pro Modultyp (epsilon, alpha-beta, ReLU/GELU/Norm …)
      └─ propagate.py                # propagate_lrp() Dispatcher & Erhaltungstests
```

> **Leitidee:** Alles, was _LRP-Backprop-Logik_ ist, liegt ausschließlich unter `lrp/` — insbesondere `lrp/logic.py` (Regeln) und `lrp/propagate.py` (Ablauf & Dispatch). Der Rest (Model laden, Layer finden, Bilder iterieren, CSV exportieren) ist LRP‑agnostisch.

---

## 2) Mapping: Was wandert wohin?

- **Kompatibilitätsfixe (Pillow/NumPy)** → `attribution/compat.py`
- **Config/Defaults (Gewichte/Pfade/Thresholds)** → `attribution/config.py`
- **Detectron2/MaskDINO Konfiguration** → `attribution/model/cfg.py`
- **Layer‑Suche (Encoder/Decoder Heuristik)** → `attribution/model/probe.py`
- **Dataset‑Registrierung (Minimal)** → `attribution/model/register_minimal.py`
- **Bildsammlung & Resize‑Transform** → `attribution/data/images.py`
- **Tensor‑Tools (`_to_BTC`, finite‑Guards)** → `attribution/utils/tensor.py`
- **Temporärer Hook (`_LayerTap`)** → `attribution/utils/hooks.py`
- **CSV‑Export & Top‑K‑Logging** → `attribution/export/csv.py`
- **LRP Startziel (`build_target_relevance`)** → `attribution/utils/tensor.py` (oder `attribution/runner.py`)
- **LRP Aggregation (`aggregate_channel_relevance`)** → `attribution/utils/tensor.py`
- **Orchestrierung (`run_analysis`)** → `attribution/runner.py`
- **CLI/`main`** → `attribution/cli.py`
- **LRP Kern (Config/Tracer/Regeln/Propagator)** → `attribution/lrp/*.py`

---

## 3) Public API beibehalten (Drop-in)

Damit dein bestehender Code ohne große Änderungen weiterläuft, halten wir die bisherigen Imports stabil:

```python
# alt (monolith):
from myThesis.lrp.engine import LRPConfig, LRPTracer, propagate_lrp

# neu (weiterhin möglich dank __init__.py Re-Export):
from myThesis.attribution.lrp import LRPConfig, LRPTracer, propagate_lrp
```

`myThesis/attribution/lrp/__init__.py` re-exportiert die Symbole aus `interface.py`, `tracer.py`, `propagate.py`.

---

## 4) Kerndatei nur für LRP‑Logik: `lrp/logic.py`

> **Zweck:** Pro‑Modultyp die _LRP‑Regel_ implementieren (epsilon‑Regel, optional alpha‑beta). Ohne Model‑Know‑how, ohne I/O, ohne CLI — **nur Relevanzfluss**.

Minimal lauffähiges Skeleton (CPU‑tauglich, speicherschonend; Conv2d nutzt z+‑Variante):

```python
# myThesis/attribution/lrp/logic.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
Tensor = torch.Tensor

@dataclass(frozen=True)
class LinearParams:
    W: Tensor  # (out, in)
    b: Optional[Tensor]

@dataclass(frozen=True)
class Conv2dParams:
    W: Tensor  # (out_c, in_c, kh, kw)
    b: Optional[Tensor]
    stride: Tuple[int, int]
    padding: Tuple[int, int]
    dilation: Tuple[int, int]
    groups: int

def _safe_eps(x: Tensor, eps: float) -> Tensor:
    return eps * (x.sign() + (x == 0).to(x.dtype))

def lrp_linear_epsilon(x: Tensor, y: Tensor, R: Tensor, p: LinearParams, eps: float) -> Tensor:
    """
    epsilon/z+-Regel für nn.Linear.
    x: (B, in), y: (B, out), R: (B, out) oder (out,)
    Rückgabe: (B, in) Relevanzen.
    """
    if R.dim() == 1:
        R = R.unsqueeze(0).expand_as(y)
    Wp = torch.clamp(p.W, min=0)                 # z+
    z = x @ Wp.t() + 1e-12                       # (B, out)
    s = R / (z + _safe_eps(z, eps))              # (B, out)
    c = s @ Wp                                   # (B, in)
    Rin = x * c                                  # (B, in)
    return Rin

def lrp_conv2d_epsilon(x: Tensor, y: Tensor, R: Tensor, p: Conv2dParams, eps: float) -> Tensor:
    """
    epsilon/z+-Regel für nn.Conv2d (kanal-lokal).
    x: (B, Cin,H,W), y,R: (B, Cout,H',W')
    Rückgabe: (B, Cin,H,W)
    """
    Wp = torch.clamp(p.W, min=0)                 # z+
    z = F.conv2d(x, Wp, bias=None, stride=p.stride, padding=p.padding,
                 dilation=p.dilation, groups=p.groups) + 1e-12
    s = R / (z + _safe_eps(z, eps))              # (B, Cout,H',W')
    c = F.conv_transpose2d(s, Wp, bias=None, stride=p.stride, padding=p.padding,
                           dilation=p.dilation, groups=p.groups)
    Rin = x * c                                  # (B, Cin,H,W)
    return Rin

def lrp_relu_passthrough(x: Tensor, y: Tensor, R: Tensor) -> Tensor:
    """ReLU verteilt Relevanz 1:1 zu den aktiven Eingängen (hier: Identität)."""
    return R

def lrp_gelu_passthrough(x: Tensor, y: Tensor, R: Tensor) -> Tensor:
    """GELU: praktikabler Default — passt Relevanz ohne Gewichtung durch."""
    return R

def lrp_layernorm_proportional(x: Tensor, y: Tensor, R: Tensor, gamma: Optional[Tensor]) -> Tensor:
    """
    Näherung: verteile R proportional zur absoluten Aktivierung (oder gleichmäßig).
    Robust und konservativ; für exakte LRP braucht man LN-Jacobian-sparse Tricks.
    """
    if gamma is not None:
        w = torch.abs(x * gamma) + 1e-12
    else:
        w = torch.abs(x) + 1e-12
    wsum = w.sum(dim=tuple(range(1, x.dim())), keepdim=True)
    return R * (w / wsum).expand_as(x)
```

> **Hinweis:** `nn.MultiheadAttention` und `MSDeformAttn` (MaskDINO) sind **spezial**. Standard‑Fallback: _Stopp am Modulrand_ (Relevanz bleibt dort) **oder** konservative Verteilung auf Value‑Pfad. Für produktive Genauigkeit brauchst du angepasste Regeln (siehe Abschnitt 7).

---

## 5) Dispatcher & Erhaltungstest: `lrp/propagate.py`

```python
# myThesis/attribution/lrp/propagate.py
from __future__ import annotations
from typing import Dict, Optional
import torch
import torch.nn as nn
from .logic import (
    lrp_linear_epsilon, lrp_conv2d_epsilon,
    lrp_relu_passthrough, lrp_gelu_passthrough, lrp_layernorm_proportional,
    LinearParams, Conv2dParams
)
Tensor = torch.Tensor

class LRPConfig:
    def __init__(self, rule_linear="epsilon", rule_conv="epsilon", alpha=1.0, beta=0.0, epsilon=1e-6):
        self.rule_linear = rule_linear
        self.rule_conv = rule_conv
        self.alpha = alpha
        self.beta = beta
        self.epsilon = float(epsilon)

class LRPStore:
    """Leichtgewichtiger (x,y)-Cache je Modul."""
    def __init__(self): self._xy: Dict[nn.Module, Dict[str, Tensor]] = {}
    def set(self, m: nn.Module, *, x: Tensor, y: Tensor): self._xy.setdefault(m, {}).update({"x": x, "y": y})
    def get(self, m: nn.Module) -> Optional[Dict[str, Tensor]]: return self._xy.get(m)
    def clear(self): self._xy.clear()

class LRPTracer:
    """Registriert Forward-Hooks und cached x/y pro Modul (nur was wir unterstützen)."""
    def __init__(self, cfg: LRPConfig):
        self.cfg = cfg
        self.store = LRPStore()
        self._hooks = []

    def add_module(self, m: nn.Module):
        def _hook(mod, inp, out):
            # Finde erstes Tensor-Eingang/-Ausgang
            x = next((t for t in inp if isinstance(t, torch.Tensor)), None)
            y = out if isinstance(out, torch.Tensor) else next((t for t in (out if isinstance(out, (tuple, list)) else [out]) if isinstance(t, torch.Tensor)), None)
            if x is not None and y is not None:
                self.store.set(mod, x=x.detach(), y=y.detach())
        self._hooks.append(m.register_forward_hook(_hook))

    def remove(self):
        for h in self._hooks:
            try: h.remove()
            except Exception: pass
        self._hooks.clear()

def _propagate_module(m: nn.Module, x: Tensor, y: Tensor, R: Tensor, cfg: LRPConfig) -> Tensor:
    """Modul-spezifische Rückverteilung R_out -> R_in."""
    if isinstance(m, nn.Linear):
        p = LinearParams(W=m.weight, b=m.bias)
        return lrp_linear_epsilon(x, y, R, p, cfg.epsilon)
    if isinstance(m, nn.Conv2d):
        p = Conv2dParams(W=m.weight, b=m.bias, stride=m.stride, padding=m.padding,
                         dilation=m.dilation, groups=m.groups)
        return lrp_conv2d_epsilon(x, y, R, p, cfg.epsilon)
    if isinstance(m, nn.ReLU):
        return lrp_relu_passthrough(x, y, R)
    if hasattr(nn, "GELU") and isinstance(m, nn.GELU):
        return lrp_gelu_passthrough(x, y, R)
    if isinstance(m, nn.LayerNorm):
        gamma = getattr(m, "weight", None)
        return lrp_layernorm_proportional(x, y, R, gamma)
    # Fallback: stoppe am Modul — konservativ (keine Verteilung nach innen)
    return torch.zeros_like(x)

def propagate_lrp(model: nn.Module, tracer: LRPTracer, start_module: nn.Module, R_out: Tensor, cfg: LRPConfig):
    """
    Läuft den Subgraph (rekursiv) innerhalb von start_module rückwärts ab und
    liefert eine Map {modul -> R_in}. Relevanz-Erhaltung wird überwacht.
    """
    rels: Dict[nn.Module, Tensor] = {}
    stack = [(start_module, tracer.store.get(start_module), R_out)]
    while stack:
        mod, xy, Rout = stack.pop()
        if xy is None: continue
        x, y = xy["x"], xy["y"]
        Rin = _propagate_module(mod, x, y, Rout, cfg)
        rels[mod] = Rin
        # Kinder-Module rückwärts (tiefer zuerst pushen)
        for child in reversed(list(mod.children())):
            xy_child = tracer.store.get(child)
            if xy_child is not None:
                # naive Aufteilung: gleichmäßig oder proportional zur Aktivierung
                frac = 1.0 / max(len(list(mod.children())), 1)
                stack.append((child, xy_child, Rin * frac))
    # einfacher Erhaltungstest (Summe)
    r_in_sum = sum(v.sum() for v in rels.values())
    r_out_sum = R_out.sum()
    if torch.isnan(r_in_sum) or torch.isinf(r_in_sum):
        raise RuntimeError("Nicht-endliche Relevanzwerte entstanden.")
    if torch.abs(r_in_sum - r_out_sum) > (1e-3 * torch.abs(r_out_sum) + 1e-6):
        print(f"[LRP] Warnung: Relevanz nicht exakt erhalten (in={float(r_in_sum):.6g}, out={float(r_out_sum):.6g})")
    return rels
```

---

## 6) Tracer (x/y-Capture): `lrp/tracer.py` (optional getrennt)

Wenn du `LRPTracer` separat halten willst (mehr Kontrolle, Tests): verschiebe die Klasse aus `propagate.py` in `tracer.py` und importiere sie dort.

---

## 7) Spezielle MaskDINO/DETR-Teile (Attention/DeformableAttn)

- **`nn.MultiheadAttention`** (klassisch): Für präzise LRP brauchst du Zugriff auf `attn_output_weights` und den Value‑Pfad. Empfohlen:
  1) **Wrapper‑Modul** schreiben, das im `forward()` die Attention‑Gewichte cached,
  2) in `logic.py` eine Regel `lrp_mha_value_route(...)` implementieren, die Relevanz **über die Value‑Aggregation** mit den Gewichten verteilt.
- **`MSDeformAttn`** (Deformable DETR/MaskDINO): Custom CUDA‑Op. Solange keine explizite Regel vorhanden ist, **stoppe am Modulrand** (konservativer Fallback) oder verteile gleichmäßig, **kennzeichne** diesen Pfad im CSV (z.B. `method: lrp_deform_fallback`).

> **Praxis:** Beginne mit sicheren Modulen (Linear/Conv/Norm/ReLU/GELU). Hebe Attention später modular nach, sodass du die Genauigkeit gezielt testen kannst (Erhaltungstest + Sensitivitätsanalyse).

---

## 8) Orchestrierung: `runner.py` (LRP-agnostisch)

```python
# myThesis/attribution/runner.py
from __future__ import annotations
import os, gc, torch, logging
import pandas as pd
from .config import DEFAULT_WEIGHTS
from .model.cfg import build_cfg_for_inference
from .model.probe import list_encoder_like_layers, list_decoder_like_layers
from .model.register_minimal import ensure_minimal_dataset
from .data.images import collect_images, make_resizer
from .utils.tensor import to_BTC, aggregate_channel_relevance, build_target_relevance
from .utils.hooks import LayerTap
from .lrp.propagate import LRPConfig, LRPTracer, propagate_lrp

def run_analysis(images_dir: str, layer_index: int, feature_index: int, output_csv: str,
                 target_norm: str="sum1", lrp_epsilon: float=1e-6,
                 which_module: str="encoder", method: str="lrp"):
    log = logging.getLogger("lrp")
    if not os.path.exists(DEFAULT_WEIGHTS):
        raise FileNotFoundError(f"Gewichtsdatei nicht gefunden: {DEFAULT_WEIGHTS}")
    device = "cpu"
    cfg = build_cfg_for_inference(device=device)
    ensure_minimal_dataset()

    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    model = build_model(cfg); DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval().to(device)

    enc_layers = (list_decoder_like_layers if which_module == "decoder" else list_encoder_like_layers)(model)
    if not (1 <= layer_index <= len(enc_layers)):
        raise IndexError(f"layer_index {layer_index} nicht in [1,{len(enc_layers)}]")
    chosen_name, chosen_layer = enc_layers[layer_index-1]
    index_axis = "token" if which_module == "decoder" else "channel"

    img_files = collect_images(images_dir)
    resizer = make_resizer(short=320, max_size=512, fmt="RGB")

    lrp_cfg = LRPConfig(epsilon=lrp_epsilon)
    tracer = LRPTracer(lrp_cfg)
    agg_attr = None; processed = 0

    for p in img_files:
        try:
            batched_inputs, original = resizer(p, device=device)
            # Pass 1: y-only
            tracer.add_module(chosen_layer); tap = LayerTap(tracer, chosen_layer)
            with torch.inference_mode(): _ = model(batched_inputs)
            cache = tracer.store.get(chosen_layer); tracer.remove(); tap.remove()
            if not cache or "y" not in cache: continue
            y_start = cache["y"]

            # Pass 2: full trace nur auf Subtree
            tracer = LRPTracer(lrp_cfg)
            for m in chosen_layer.modules(): tracer.add_module(m)
            tap = LayerTap(tracer, chosen_layer)
            with torch.inference_mode(): _ = model(batched_inputs)

            R_out = build_target_relevance(y_start, feature_index, "mean", target_norm, index_axis=index_axis)
            rels = propagate_lrp(model, tracer, chosen_layer, R_out, lrp_cfg)
            R_in = rels.get(chosen_layer); tracer.remove(); tap.remove()

            if R_in is None: continue
            attr = aggregate_channel_relevance(R_in)
            agg_attr = attr if agg_attr is None else (agg_attr + attr)
            processed += 1
            del y_start, R_out, rels, R_in, attr; gc.collect()
        except Exception as e:
            log.exception(f"Fehler bei {p}: {e}")
            tracer.store.clear()

    if not processed: raise RuntimeError("Keine Attributionen berechnet.")
    agg_attr = agg_attr / float(processed)

    import pandas as pd
    df = pd.DataFrame({
        "prev_feature_idx": list(range(len(agg_attr))),
        "relevance": agg_attr.numpy().tolist(),
        "layer_index": layer_index,
        "layer_name": chosen_name,
        "feature_index": feature_index,
        "epsilon": lrp_epsilon,
        "module_role": "Decoder" if which_module=="decoder" else "Encoder",
        "target_norm": target_norm,
        "method": method,
    }).sort_values("relevance", ascending=False)

    from .export.csv import write_csv
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_csv(df, output_csv, log_topk=10)
```

---

## 9) Utilities (Auszug)

**`utils/tensor.py`**
```python
from __future__ import annotations
import torch
from torch import Tensor

def to_BTC(t: Tensor) -> Tensor:
    if t.dim() == 4:
        B, C, H, W = t.shape
        return t.permute(0,2,3,1).reshape(B, H*W, C)
    if t.dim() == 3:
        dims = list(t.shape)
        b = min(range(3), key=lambda i: dims[i])
        t_ax = max(range(3), key=lambda i: dims[i])
        c = ({0,1,2}-{b,t_ax}).pop()
        return t if (b,t_ax,c)==(0,1,2) else t.permute(b,t_ax,c)
    raise ValueError(f"Unexpected shape: {tuple(t.shape)}")

def aggregate_channel_relevance(R: Tensor) -> Tensor:
    if R.dim()==4: return R.sum(dim=(0,2,3)).detach().cpu()
    if R.dim()==3: return R.sum(dim=(0,1)).detach().cpu()
    if R.dim()==2: return R.sum(dim=0).detach().cpu()
    raise ValueError(f"Unexpected R shape: {tuple(R.shape)}")

def build_target_relevance(layer_out: Tensor, feature_index: int, token_reduce: str,
                           target_norm: str="sum1", index_axis: str="channel") -> Tensor:
    y = to_BTC(layer_out); B,T,C = y.shape
    base = torch.zeros_like(y)
    if index_axis=="channel":
        if not (0 <= feature_index < C): raise IndexError("feature_index out of range")
        feat = y[..., feature_index]
        w = torch.ones_like(feat) if token_reduce=="mean" else torch.relu(feat)
        s = w.sum().clamp_min(1e-12)
        if target_norm=="sum1": w = w/s
        elif target_norm=="sumT": w = w/s * float(T)
        base[..., feature_index] = w
    elif index_axis=="token":
        if not (0 <= feature_index < T): raise IndexError("feature_index out of range")
        w = torch.ones_like(y[:, feature_index, :])
        s = w.sum().clamp_min(1e-12)
        if target_norm=="sum1": w = w/s
        elif target_norm=="sumT": w = w/s * float(y.shape[-1])
        base[:, feature_index, :] = w
    else:
        raise ValueError("index_axis must be 'channel' or 'token'")
    if layer_out.dim()==4:
        B2,C2,H,W = layer_out.shape
        assert (B2==B) and (C2==C) and (H*W==T)
        base = base.view(B,H,W,C).permute(0,3,1,2).contiguous()
    return base
```

**`export/csv.py`**
```python
from __future__ annotations
import pandas as pd

def write_csv(df: pd.DataFrame, path: str, log_topk: int=10):
    df.to_csv(path, index=False)
    topk = df.head(log_topk)
    for _, row in topk.iterrows():
        print(f"idx={int(row.prev_feature_idx):4d}  rel={row.relevance:.6f}")
```

**`utils/hooks.py`**
```python
from __future__ annotations
import torch, torch.nn as nn
from typing import Optional

class LayerTap:
    def __init__(self, tracer, module: nn.Module, retain_grad: bool=False):
        self.tracer = tracer; self.module = module; self.retain_grad = retain_grad
        self._h = module.register_forward_hook(self._on_forward)
    def _on_forward(self, mod, inp, out):
        def _first_tensor(o):
            if isinstance(o, (list,tuple)):
                for t in o:
                    r = _first_tensor(t)
                    if r is not None: return r
                return None
            return o if isinstance(o, torch.Tensor) else None
        x = _first_tensor(inp); y = _first_tensor(out)
        if self.retain_grad and isinstance(x, torch.Tensor) and x.requires_grad: x.retain_grad()
        if x is not None and y is not None: self.tracer.store.set(mod, x=x, y=y)
    def remove(self):
        try: self._h.remove()
        except Exception: pass
```

---

## 10) CLI: `cli.py`

```python
# myThesis/attribution/cli.py
from __future__ annotations
import argparse
from .runner import run_analysis

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LRP/Attribution für MaskDINO")
    p.add_argument("--images-dir", type=str, required=True)
    p.add_argument("--layer-index", type=int, default=3)
    p.add_argument("--feature-index", type=int, default=214)
    p.add_argument("--target-norm", type=str, default="sum1", choices=["sum1","sumT","none"])
    p.add_argument("--lrp-epsilon", type=float, default=1e-6)
    p.add_argument("--output-csv", type=str, required=True)
    p.add_argument("--which-module", type=str, default="encoder", choices=["encoder","decoder"])
    p.add_argument("--method", type=str, default="lrp", choices=["lrp","gradinput"])
    return p.parse_args()

def main():
    a = parse_args()
    run_analysis(images_dir=a.images_dir, layer_index=a.layer_index, feature_index=a.feature_index,
                 output_csv=a.output_csv, target_norm=a.target_norm, lrp_epsilon=a.lrp_epsilon,
                 which_module=a.which_module, method=a.method)

if __name__ == "__main__":
    main()
```

---

## 11) Tests & Anti‑Bullshit‑Guards

1) **Relevanz-Erhaltung:** In `propagate_lrp()` ist ein Summen‑Check eingebaut (tolerant). Logge Abweichungen.
2) **Finiteness:** Abort bei NaN/Inf.
3) **Ablation‑Check:** Erhöhe/erniedrige `feature_index` und prüfe monotone Veränderungen der Top‑Kanäle.
4) **Sanity:** Vergleiche mit Grad*Input als Baseline; starke Divergenz → untersuchen.

---

## 12) Migrationsschritte (konkret & kurz)

1. Lege den Dateibaum wie oben an (leere `__init__.py` inkludieren).
2. Verschiebe Funktionen gemäß Mapping in die neuen Module.
3. Ersetze die monolithische `run_analysis()` durch `attribution/runner.py`.
4. Importiere `LRPConfig/LRPTracer/propagate_lrp` fortan aus `attribution/lrp`.
5. Führe über `python -m myThesis.attribution.cli --images-dir ... --output-csv ...` aus.
6. (Optional) Ergänze später `lrp/logic.py` um präzise Regeln für Attention/DeformableAttn.

---

## 13) Was du _jetzt_ hast

- Eine **klare Trennung** von LRP‑Backprop‑Logik (rein in `lrp/logic.py`) und Orchestrierung.
- Ein **Dispatcher** mit konservativen Defaults, die funktionieren (Linear/Conv/Norm/ReLU/GELU).
- Ein **Drop‑in‑API** kompatibel zu deinen bisherigen Imports.
- Einen **Pfad**, Attention‑Regeln gezielt nachzurüsten, ohne den Rest zu berühren.
