"""
Hooks, um stabile Cut-Punkte (x_prev, y_curr) rund um einen gewählten SubLayer in
MaskDINO zu erfassen. Fällt robust auf ein generisches Modul-Hooking zurück, wenn
kein strukturierter Zugriff auf Transformer-(En/De)coder gefunden wird.
"""
from __future__ import annotations

from typing import Tuple, List, Optional
import warnings
import torch
from torch import nn, Tensor
import torch.nn.functional as F
try:
    # Optional: nur für LRP-Value-Pfad-Caching benötigt
    from myThesis.lrp.lrp.value_path import AttnCache  # type: ignore
except Exception:
    AttnCache = None  # type: ignore


class CutPoints:
    def __init__(self):
        self.x_prev: Optional[Tensor] = None  # Tensor vor Sublayer (L-1 → L Input)
        self.y_curr: Optional[Tensor] = None  # Tensor am Messpunkt (pre_res oder post_res)
    def clear(self):
        self.x_prev = None
        self.y_curr = None


def _robust_get_layers(model: nn.Module, which_module: str) -> Optional[nn.ModuleList]:
    """Versuche, eine Layerliste (ModuleList) für Encoder/Decoder zu finden."""
    names = [n for n, _ in model.named_modules()]
    lower = {n.lower(): n for n in names}
    # Direkter Zugriff, falls vorhanden
    if which_module == "encoder":
        for key in ("transformer.encoder.layers", "encoder.layers"):
            ln = lower.get(key)
            if ln is not None:
                mod = dict(model.named_modules())[ln]
                if isinstance(mod, nn.ModuleList):
                    return mod
    else:
        for key in ("transformer.decoder.layers", "decoder.layers"):
            ln = lower.get(key)
            if ln is not None:
                mod = dict(model.named_modules())[ln]
                if isinstance(mod, nn.ModuleList):
                    return mod
    # Heuristik: nimm das erste ModuleList unterhalb von *encoder/*decoder*
    for n, m in model.named_modules():
        lname = n.lower()
        if which_module in ("encoder", "decoder") and which_module in lname and isinstance(m, nn.ModuleList):
            return m
    return None


def register_cut_hooks(
    model: nn.Module,
    which_module: str,
    layer_index: int,
    use_sublayer: str,
    measurement_point: str,
) -> Tuple[CutPoints, List[torch.utils.hooks.RemovableHandle]]:
    """Registriere Hooks um (x_prev, y_curr) für den gewünschten Layer/SubLayer zu erfassen.

    Falls die spezifische Struktur nicht erkannt wird, wird direkt auf dem Ziel-Layer
    selbst gehookt (Generalfall). layer_index ist 1-basiert.
    """
    cps = CutPoints()
    handles: List[torch.utils.hooks.RemovableHandle] = []

    layers = _robust_get_layers(model, which_module)
    target_module: Optional[nn.Module] = None
    if layers is not None and 1 <= layer_index <= len(layers):
        Lmod = layers[layer_index - 1]
        # SubLayer auswählen (best effort)
        sub: Optional[nn.Module] = None
        if use_sublayer == "self_attn" and hasattr(Lmod, "self_attn"):
            sub = getattr(Lmod, "self_attn")
        elif use_sublayer == "cross_attn" and hasattr(Lmod, "multihead_attn"):
            sub = getattr(Lmod, "multihead_attn")
        elif use_sublayer == "ffn" and hasattr(Lmod, "linear2"):
            sub = getattr(Lmod, "linear2")
        target_module = sub or Lmod
    else:
        # Generischer Fallback: nimm das ganze Modell (später nützlicher Caller übergibt konkretes Modul)
        target_module = model

    def pre_hook(_m, inp):
        x = inp[0] if isinstance(inp, (tuple, list)) else inp
        if isinstance(x, Tensor):
            cps.x_prev = x.detach().requires_grad_(False)
        return None

    def post_hook(_m, _inp, out):
        y = out
        if isinstance(y, Tensor):
            cps.y_curr = y.detach()
        return None

    if target_module is None:
        raise RuntimeError("Kein Zielmodul für Hooks gefunden")

    handles.append(target_module.register_forward_pre_hook(pre_hook))
    handles.append(target_module.register_forward_hook(post_hook))
    return cps, handles


def register_cut_hooks_by_module(
    module: nn.Module,
    attn_cache: Optional[object] = None,
) -> Tuple[CutPoints, List[torch.utils.hooks.RemovableHandle]]:
    """Einfacher Helper, um direkt an einem gegebenen Modul x/y zu erfassen.

    Hinweise für Attention-Capture:
    - Für nn.MultiheadAttention liefert PyTorch die Gewichte nur zuverlässig,
      wenn im Aufruf need_weights=True gesetzt wird. Andernfalls kann attn_weights
      fehlen und der Value-Pfad-Fix fällt in Fallbacks.

    Optional: attn_cache – falls übergeben, werden auf darunter liegenden
    nn.MultiheadAttention-Modulen zusätzlich die Attention-Gewichte zwischengespeichert.
    """
    cps = CutPoints()
    handles: List[torch.utils.hooks.RemovableHandle] = []

    def pre_hook(_m, inp):
        x = inp[0] if isinstance(inp, (tuple, list)) else inp
        if isinstance(x, Tensor):
            cps.x_prev = x.detach().requires_grad_(False)
        return None

    def post_hook(_m, _inp, out):
        y = out
        if isinstance(y, Tensor):
            cps.y_curr = y.detach()
        # Warnen, falls Attention-Capture gewünscht, aber keine Gewichte vorliegen
        if attn_cache is not None and getattr(attn_cache, "attn_weights", None) is None:
            if not getattr(attn_cache, "_warned_no_attn", False):
                warnings.warn(
                    "register_cut_hooks_by_module: attn_cache aktiv, aber keine attn_weights erfasst. "
                    "Stelle sicher, dass need_weights=True beim Aufruf von nn.MultiheadAttention gesetzt ist.",
                    stacklevel=2,
                )
                try:
                    setattr(attn_cache, "_warned_no_attn", True)
                except Exception:
                    pass
        return None

    handles.append(module.register_forward_pre_hook(pre_hook))
    handles.append(module.register_forward_hook(post_hook))

    # Optional: Attention-/MHA-Zwischenwerte aus nn.MultiheadAttention erfassen
    if attn_cache is not None:
        # Kleiner Helper, um forward eines MHA-Modules temporär so zu wrappen,
        # dass need_weights=True erzwungen wird. Wir hängen einen "Handle" an,
        # dessen remove() den Patch zurücksetzt.
        class _ForwardPatchHandle:
            def __init__(self, mod, orig):
                self._m = mod
                self._orig = orig
                self._active = True
            def remove(self):
                if self._active and hasattr(self._m, "forward"):
                    try:
                        self._m.forward = self._orig
                    except Exception:
                        pass
                self._active = False

        def _mha_forward_hook(_m: nn.MultiheadAttention, _inp, out):
            try:
                # PyTorch MHA liefert (attn_output, attn_weights)
                if isinstance(out, (tuple, list)) and len(out) >= 2:
                    aw = out[1]
                else:
                    # Unbekanntes Format – abbrechen
                    return None
                if isinstance(aw, Tensor):
                    # Normalisiere Shapes: (B,H,T,S)
                    B = None
                    H = _m.num_heads
                    if aw.dim() == 2:
                        # (T,S) – batch implizit 1, keine Head-Dimension verfügbar
                        aw4 = aw.unsqueeze(0).unsqueeze(0)  # (1,1,T,S)
                    elif aw.dim() == 3:
                        # Entweder (B*H,T,S) oder (B,T,S)
                        if aw.shape[0] == H:
                            # (H,T,S) -> (1,H,T,S)
                            aw4 = aw.unsqueeze(0)
                        else:
                            # (B,T,S) -> (B,1,T,S)
                            aw4 = aw.unsqueeze(1)
                    elif aw.dim() == 4:
                        # (B,H,T,S)
                        aw4 = aw
                    else:
                        return None
                    attn_cache.attn_weights = aw4.detach()

                # Versuche zusätzlich V-Projektion und Projektionsgewichte zu erfassen
                # Eingaben: query, key, value
                if isinstance(_inp, (tuple, list)) and len(_inp) >= 3:
                    q, k, v = _inp[:3]
                else:
                    return None

                batch_first = getattr(_m, 'batch_first', False)
                embed_dim = _m.embed_dim
                num_heads = _m.num_heads
                head_dim = embed_dim // num_heads

                def to_btc(t: Tensor) -> Tensor:
                    # (T,B,C) -> (B,T,C) wenn batch_first False
                    return t.transpose(0, 1) if (t.dim() == 3 and not batch_first) else t

                v_btc: Tensor = to_btc(v)
                # in_proj_weight: (3E, E), in der Reihenfolge (Q,K,V)
                W_in = getattr(_m, 'in_proj_weight', None)
                b_in = getattr(_m, 'in_proj_bias', None)
                if W_in is not None and isinstance(W_in, Tensor):
                    W_in = W_in.detach()
                    b_in = b_in.detach() if isinstance(b_in, Tensor) else None
                    E = embed_dim
                    W_v = W_in[2*E:3*E, :]  # (E, E)
                    b_v = b_in[2*E:3*E] if b_in is not None else None
                    # v_proj: (B,T,E)
                    v_proj = F.linear(v_btc, W_v, b_v)
                    # (B,T,H,Dh) -> (B,H,T,Dh) -> (B,H,S,Dh)
                    Bv, Tv, Ev = v_proj.shape
                    v_proj = v_proj.view(Bv, Tv, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
                    attn_cache.Vproj = v_proj.detach()  # (B,H,T,Dh) == (B,H,S,Dh) für self-attn
                    # Out-Projektion: (E_out, E_in) = (E, E) – reshape zu (H,Dh,C)
                    W_o = _m.out_proj.weight.detach()  # (E, E)
                    # (E, E) -> (E_in=C, E_out=C); wir brauchen (H,Dh,C_out)
                    # Wir interpretieren Spalten (C_out) und teilen Zeilen (C_in) auf Köpfe/Dh
                    W_o_reshaped = W_o.t().contiguous().view(embed_dim, num_heads, head_dim).permute(1, 2, 0).contiguous()
                    attn_cache.W_O = W_o_reshaped  # (H,Dh,C)
                    # Value-Projektionsgewichte pro Kopf/Dh
                    W_v_reshaped = W_v.view(num_heads, head_dim, embed_dim)  # (H,Dh,C)
                    attn_cache.W_V = W_v_reshaped.contiguous()
            except Exception:
                pass
            return None

        for m in module.modules():
            if isinstance(m, nn.MultiheadAttention):
                # 1) Hook für Output (um attn_weights zu lesen)
                handles.append(m.register_forward_hook(_mha_forward_hook))
                # 2) Patch: need_weights=True erzwingen
                try:
                    orig_forward = m.forward
                    def _wrapped_forward(*args, **kwargs):
                        kwargs = dict(kwargs)
                        kwargs.setdefault("need_weights", True)
                        return orig_forward(*args, **kwargs)
                    m.forward = _wrapped_forward
                    handles.append(_ForwardPatchHandle(m, orig_forward))
                except Exception:
                    pass

        # Zusätzlich: MaskDINO/Deformable-DETR MSDeformAttn ähnlicher Module
        # Wir können intern keine attention_weights/Samplingpunkte abgreifen,
        # aber zumindest die Projektionsgewichte W_V/W_O aus value_proj/output_proj.
        def _maybe_capture_msdeform(m: nn.Module):
            try:
                # Heuristik: Modul mit .value_proj & .output_proj (nn.Linear) sowie Attributen n_heads/d_model
                vp = getattr(m, 'value_proj', None)
                op = getattr(m, 'output_proj', None)
                H = getattr(m, 'n_heads', None)
                E = getattr(m, 'd_model', None)
                if isinstance(vp, nn.Linear) and isinstance(op, nn.Linear) and isinstance(H, int) and isinstance(E, int) and E % H == 0:
                    Dh = E // H
                    W_v = vp.weight.detach()  # (E, E)
                    W_o = op.weight.detach()  # (E, E)
                    # Forme zu (H,Dh,C) mit C=E
                    attn_cache.W_V = W_v.view(E, H, Dh).permute(1, 2, 0).contiguous()
                    attn_cache.W_O = W_o.t().contiguous().view(E, H, Dh).permute(1, 2, 0).contiguous()
            except Exception:
                pass
        for m in module.modules():
            _maybe_capture_msdeform(m)

        # FFN-Gewichte (linear1/linear2 oder fc1/fc2) innerhalb des gewählten Layers erfassen
        try:
            W1_found = False
            W2_found = False
            for name, m in module.named_modules():
                # Bevorzugte Namenskonventionen: linear1/linear2, fallback fc1/fc2
                lin1 = getattr(m, 'linear1', None)
                lin2 = getattr(m, 'linear2', None)
                if isinstance(lin1, nn.Linear) and isinstance(lin2, nn.Linear):
                    attn_cache.W_FFN1 = lin1.weight.detach()
                    attn_cache.W_FFN2 = lin2.weight.detach()
                    W1_found = W2_found = True
                    break
                # Alternative Bezeichnungen
                fc1 = getattr(m, 'fc1', None)
                fc2 = getattr(m, 'fc2', None)
                if isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear):
                    attn_cache.W_FFN1 = fc1.weight.detach()
                    attn_cache.W_FFN2 = fc2.weight.detach()
                    W1_found = W2_found = True
                    break
            # Wenn nicht gefunden, keine Exception – Engine fällt robust zurück.
        except Exception:
            pass

    return cps, handles


__all__ = [
    "CutPoints",
    "register_cut_hooks",
    "register_cut_hooks_by_module",
]
