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
    from myThesis.lrp.calc.msdeformattn_capture import attach_msdeformattn_capture
except Exception:
    # Fallback: Wenn das Deformable-Attention-Capture nicht verfügbar ist,
    # definieren wir eine No-Op-Funktion, damit der Rest der Hooks weiterhin funktioniert.
    def attach_msdeformattn_capture(module: nn.Module, attn_cache):  # type: ignore
        return []
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

    # Hilfsfunktionen: erstes Tensorobjekt finden und robust in (B,T,C) bringen
    def _first_tensor_in(o):
        if isinstance(o, Tensor):
            return o
        if isinstance(o, (tuple, list)):
            for it in o:
                ft = _first_tensor_in(it)
                if isinstance(ft, Tensor):
                    return ft
        if isinstance(o, dict):
            for v in o.values():
                ft = _first_tensor_in(v)
                if isinstance(ft, Tensor):
                    return ft
        return None

    def _to_btc_like(t: Tensor) -> Tensor:
        # 4D (B,C,H,W) -> (B,H*W,C)
        if t.dim() == 4 and t.shape[1] < max(t.shape[2], t.shape[3]):
            B, C, H, W = t.shape
            return t.permute(0, 2, 3, 1).reshape(B, H * W, C)
        # 3D: Wenn offensichtlich (T,B,C), transponieren nach (B,T,C)
        if t.dim() == 3 and t.shape[0] > t.shape[1]:
            return t.transpose(0, 1).contiguous()
        return t

    def pre_hook(_m, inp):
        # Robust: erstes Tensor-Argument (rekursiv) suchen; leere Tupel erlauben
        x = _first_tensor_in(inp)
        if isinstance(x, Tensor):
            x_btc = _to_btc_like(x)
            cps.x_prev = x_btc.detach().requires_grad_(False)
        return None

    # Flag, ob unterhalb des Moduls ein nn.MultiheadAttention gefunden wurde
    saw_mha: bool = False

    def post_hook(_m, _inp, out):
        def _first_tensor(o):
            if isinstance(o, Tensor):
                return o
            if isinstance(o, (tuple, list)):
                for it in o:
                    ft = _first_tensor(it)
                    if isinstance(ft, Tensor):
                        return ft
            if isinstance(o, dict):
                for v in o.values():
                    ft = _first_tensor(v)
                    if isinstance(ft, Tensor):
                        return ft
            return None
        y = _first_tensor(out)
        if isinstance(y, Tensor):
            y_btc = _to_btc_like(y)
            cps.y_curr = y_btc.detach()
        # Warnen, falls explizit MHA vorhanden, aber keine attn_weights erfasst wurden
        # und zugleich keine Deformable-Attention-Daten vorliegen.
        if attn_cache is not None:
            has_aw = getattr(attn_cache, "attn_weights", None) is not None
            has_deform = (
                getattr(attn_cache, "deform_sampling_locations", None) is not None or
                getattr(attn_cache, "deform_attention_weights", None) is not None
            )
            if saw_mha and (not has_aw) and (not has_deform):
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

    # Fallback: Zusätzlich an geeigneten Kind-Modulen hooken, falls der Top-Level-Hook
    # (z. B. wegen reinem kwargs-Call) nicht feuert. Wir setzen cps-Werte nur,
    # wenn sie noch nicht befüllt sind (nicht überschreiben).
    def child_pre_hook(_m, inp):
        if cps.x_prev is not None:
            return None
        x = _first_tensor_in(inp)
        if isinstance(x, Tensor):
            x_btc = _to_btc_like(x)
            cps.x_prev = x_btc.detach().requires_grad_(False)
        return None

    def child_post_hook(_m, _inp, out):
        # Nur als Fallback: y_curr nicht von Kindmodulen mit 1D/2D-Outputs setzen,
        # da dies die Token-Dimension zerstören kann. Warte bevorzugt auf den Top-Level-Hook.
        if cps.y_curr is not None:
            return None
        def _first_tensor(o):
            if isinstance(o, Tensor):
                return o
            if isinstance(o, (tuple, list)):
                for it in o:
                    ft = _first_tensor(it)
                    if isinstance(ft, Tensor):
                        return ft
            if isinstance(o, dict):
                for v in o.values():
                    ft = _first_tensor(v)
                    if isinstance(ft, Tensor):
                        return ft
            return None
        y = _first_tensor(out)
        # Nur setzen, wenn es sich plausibel um (B,T,C) handelt (>=3D)
        if isinstance(y, Tensor) and y.dim() >= 3:
            y_btc = _to_btc_like(y)
            cps.y_curr = y_btc.detach()
        return None

    # Nur gezielt an Blättern oder bekannten Untermodule hooken, um Overhead gering zu halten
    known_names = ("self_attn", "cross_attn", "multihead_attn", "ffn", "msdeformattn")
    for name, m in module.named_modules():
        if m is module:
            continue
        has_children = any(True for _ in m.children())
        is_known = any(kw in name.lower() for kw in known_names)
        if (not has_children) or is_known:
            try:
                handles.append(m.register_forward_pre_hook(child_pre_hook))
                handles.append(m.register_forward_hook(child_post_hook))
            except Exception:
                pass

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
                    # Kopfzahl-Metadaten für spätere Konsistenzprüfungen
                    try:
                        setattr(attn_cache, "_mha_num_heads", H)
                    except Exception:
                        pass
                    attn_weights_captured = aw4.detach()

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
                # Bevorzugt PyTorch >=2: getrennte v_proj_weight/bias nutzen
                W_v = getattr(_m, 'v_proj_weight', None)
                b_v = getattr(_m, 'v_proj_bias', None)
                if isinstance(W_v, Tensor):
                    W_v = W_v.detach()
                    b_v = b_v.detach() if isinstance(b_v, Tensor) else None
                else:
                    # Fallback: in_proj_* Segmentierung
                    W_in = getattr(_m, 'in_proj_weight', None)
                    b_in = getattr(_m, 'in_proj_bias', None)
                    if W_in is None or not isinstance(W_in, Tensor):
                        # Ohne Gewichte können wir V-Projektion nicht berechnen
                        W_in = None
                    else:
                        W_in = W_in.detach()
                        b_in = b_in.detach() if isinstance(b_in, Tensor) else None
                        E = embed_dim
                        W_v = W_in[2*E:3*E, :]  # (E,E)
                        b_v = b_in[2*E:3*E] if b_in is not None else None

                if isinstance(W_v, Tensor):
                    # v_proj: (B,T,E)
                    v_proj = F.linear(v_btc, W_v, b_v)
                    # (B,T,H,Dh) -> (B,H,T,Dh) -> (B,H,S,Dh)
                    Bv, Tv, Ev = v_proj.shape
                    v_proj = v_proj.view(Bv, Tv, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
                    Vproj_captured = v_proj.detach()  # (B,H,T,Dh) == (B,H,S,Dh) für self-attn
                    # Out-Projektion: (E_out, E_in) = (E, E) – reshape zu (H,Dh,C)
                    W_o = _m.out_proj.weight.detach()  # (E, E)
                    W_o_reshaped = W_o.t().contiguous().view(embed_dim, num_heads, head_dim).permute(1, 2, 0).contiguous()
                    W_O_captured = W_o_reshaped  # (H,Dh,C)
                    # Value-Projektionsgewichte pro Kopf/Dh
                    W_v_reshaped = W_v.view(num_heads, head_dim, embed_dim)  # (H,Dh,C)
                    W_V_captured = W_v_reshaped.contiguous()
                else:
                    Vproj_captured = None
                    W_O_captured = None
                    W_V_captured = None

                # Self- vs Cross-Attention Heuristik bestimmen
                try:
                    kind = "self" if (q is k and k is v) else "cross"
                except Exception:
                    kind = None
                prefer = getattr(attn_cache, "prefer_kind", None)
                last_kind = getattr(attn_cache, "_last_kind", None)
                def _should_update(k):
                    if k is None:
                        return True
                    if prefer is None:
                        return (last_kind is None) or (last_kind == k)
                    else:
                        return (prefer == k) or (last_kind is None)

                if _should_update(kind):
                    # Konsistent speichern, ggf. Kopfzahl der Gewichte an Vproj anpassen
                    if isinstance(attn_weights_captured, Tensor):
                        aw4 = attn_weights_captured
                        if isinstance(Vproj_captured, Tensor) and aw4.dim() == 4 and Vproj_captured.dim() == 4:
                            if aw4.shape[1] != Vproj_captured.shape[1]:
                                aw4 = aw4.expand(aw4.shape[0], Vproj_captured.shape[1], aw4.shape[2], aw4.shape[3]).contiguous()
                        attn_cache.attn_weights = aw4
                    if isinstance(Vproj_captured, Tensor):
                        attn_cache.Vproj = Vproj_captured
                    if isinstance(W_O_captured, Tensor):
                        attn_cache.W_O = W_O_captured
                    if isinstance(W_V_captured, Tensor):
                        attn_cache.W_V = W_V_captured
                    try:
                        setattr(attn_cache, "_last_kind", kind)
                    except Exception:
                        pass
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
                        # Erzwinge Gewichte unabhängig von Caller-Defaults
                        kwargs["need_weights"] = True
                        # Und erzwinge: keine Head-Mittelung
                        kwargs["average_attn_weights"] = False
                        return orig_forward(*args, **kwargs)
                    m.forward = _wrapped_forward
                    handles.append(_ForwardPatchHandle(m, orig_forward))
                except Exception:
                    pass
                saw_mha = True

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

        # Deformable Attention (MSDeformAttn) internals (sampling_locations, attention_weights)
        try:
            deform_handles = attach_msdeformattn_capture(module, attn_cache)
            for h in deform_handles:
                handles.append(h)
        except Exception:
            pass

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

        # LayerNorm-Capture: Eingang x, Gamma (weight) und optional Beta (bias)
        try:
            for m in module.modules():
                if isinstance(m, nn.LayerNorm):
                    def _ln_pre(m_, inp):
                        # Gleiches robustes Suchen wie oben, um IndexError zu vermeiden
                        def _first_tensor_in(o):
                            if isinstance(o, Tensor):
                                return o
                            if isinstance(o, (tuple, list)):
                                for it in o:
                                    ft = _first_tensor_in(it)
                                    if isinstance(ft, Tensor):
                                        return ft
                            if isinstance(o, dict):
                                for v in o.values():
                                    ft = _first_tensor_in(v)
                                    if isinstance(ft, Tensor):
                                        return ft
                            return None
                        x = _first_tensor_in(inp)
                        if isinstance(x, Tensor):
                            try:
                                # Rohwert (ohne detach) für optionale grad-basierte LN-Regel
                                try:
                                    attn_cache.ln_x_in_raw = x
                                except Exception:
                                    pass
                                attn_cache.ln_x_in = x.detach()
                                attn_cache.ln_gamma = m_.weight.detach() if isinstance(m_.weight, Tensor) else None
                                attn_cache.ln_beta = m_.bias.detach() if isinstance(m_.bias, Tensor) else None
                            except Exception:
                                pass
                        return None
                    handles.append(m.register_forward_pre_hook(_ln_pre))
        except Exception:
            pass

    return cps, handles


__all__ = [
    "CutPoints",
    "register_cut_hooks_by_module",
]
