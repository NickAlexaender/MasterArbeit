
from __future__ import annotations
import warnings
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor


_MSDEFORMATTN_CLS_CANDIDATES = [
    "models.ops.modules.ms_deform_attn.MSDeformAttn",
    "projects.DETR2.d2.layers.deformable_attention.MSDeformAttn",
    "maskdino.modeling.pixel_decoder.ops.modules.ms_deform_attn.MSDeformAttn",
    "maskdino.layers.ms_deform_attn.MSDeformAttn",
    "detectron2.layers.deformable_attention.MSDeformAttn",
    "ultralytics.nn.modules.transformer.MSDeformAttn",
]

# Hier versuchen wir die MSDeformAttn-Klasse zu laden

def _resolve_msdeformattn_class() -> Optional[type]:
    for fq in _MSDEFORMATTN_CLS_CANDIDATES:
        try:
            mod_path, cls_name = fq.rsplit('.', 1)
            mod = __import__(mod_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name, None)
            if isinstance(cls, type):
                return cls
        except Exception:
            continue
    return None


# Wir managen hier den Forward-Patch, um diesesn sauber wieder zu entfernen

class _ForwardPatchHandle:
    
    def __init__(self, mod: nn.Module, orig_forward):
        self._m = mod
        self._orig = orig_forward
        self._active = True
    
    def remove(self):
        if self._active and hasattr(self._m, 'forward'):
            try:
                self._m.forward = self._orig
            except Exception:
                pass
        self._active = False


# Iterieren des Modulbaums, um Eltern-Module zu finden

def _iter_parents(root: nn.Module, child: nn.Module):
    name_to_module = {n: m for n, m in root.named_modules()}
    module_to_parent = {}
    for name, module in root.named_modules():
        for sub_name, sub_module in module.named_children():
            module_to_parent[sub_module] = module
    current = child
    while current in module_to_parent:
        parent = module_to_parent[current]
        pname = next((n for n, m in name_to_module.items() if m is parent), '<unnamed>')
        yield pname, parent
        current = parent

# Auslesen der Projektionsgewichte

def _maybe_capture_parent_projections(parent_block: nn.Module, attn_cache: object) -> None:
    try:
        vp = getattr(parent_block, 'value_proj', None)
        op = getattr(parent_block, 'output_proj', None)
        n_heads = getattr(parent_block, 'n_heads', None) or getattr(parent_block, 'num_heads', None)
        d_model = getattr(parent_block, 'd_model', None) or getattr(parent_block, 'embed_dim', None)
        
        if (isinstance(vp, nn.Linear) and isinstance(op, nn.Linear) and 
            isinstance(n_heads, int) and isinstance(d_model, int) and 
            d_model % n_heads == 0):
            
            Dh = d_model // n_heads
            E = d_model
            W_v = vp.weight.detach()
            W_o = op.weight.detach()
            
            setattr(attn_cache, 'W_V_deform', 
                    W_v.view(E, n_heads, Dh).permute(1, 2, 0).contiguous())
            setattr(attn_cache, 'W_O_deform', 
                    W_o.t().contiguous().view(E, n_heads, Dh).permute(1, 2, 0).contiguous())
    except Exception:
        pass


# Erfassungen von Aktivierungen in MSDeformAttn-Modulen

def attach_msdeformattn_capture(
    root: nn.Module,
    attn_cache: Optional[object],
) -> List[_ForwardPatchHandle]:
    handles: List[_ForwardPatchHandle] = []
    MSDeformAttn = _resolve_msdeformattn_class()
    
    if MSDeformAttn is None:
        # Diagnose: Keine MSDeformAttn-Klasse gefunden
        try:
            already = getattr(attn_cache, "_warned_no_msdeform", False) if attn_cache is not None else False
        except Exception:
            already = False
        if not already:
            warnings.warn(
                "attach_msdeformattn_capture: Keine MSDeformAttn-Klasse gefunden. "
                "Füge ggf. den vollqualifizierten Klassennamen in _MSDEFORMATTN_CLS_CANDIDATES hinzu, "
                "damit Sampling-Lokationen und Gewichte erfasst werden können.",
                stacklevel=2,
            )
            try:
                if attn_cache is not None:
                    setattr(attn_cache, "_warned_no_msdeform", True)
            except Exception:
                pass
        return handles
    
    if attn_cache is None:
        warnings.warn('attach_msdeformattn_capture called without attn_cache', stacklevel=2)
        
    # Jetzt wrappen wir die forward-Methode

    def _wrap_forward(mod: nn.Module):
        orig_forward = mod.forward
        
        def _patched_forward(
            query: Tensor,
            reference_points: Tensor,
            input_flatten: Tensor,
            input_spatial_shapes: Tensor,
            input_level_start_index: Tensor,
            input_padding_mask: Optional[Tensor] = None,
            *args,
            **kwargs
        ):
            # Original Forward aufrufen
            out = orig_forward(
                query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask, *args, **kwargs
            )
            
            # Aktivierungen erfassen
            try:
                if attn_cache is not None:
                    _capture_activations(
                        mod, query, reference_points, input_flatten,
                        input_spatial_shapes, input_level_start_index,
                        input_padding_mask, attn_cache
                    )
            except Exception:
                pass
            
            return out
        
        mod.forward = _patched_forward
        handles.append(_ForwardPatchHandle(mod, orig_forward))
    # Alle MSDeformAttn-Instanzen finden und wrappen
    found = 0
    for m in root.modules():
        if isinstance(m, MSDeformAttn):
            _wrap_forward(m)
            # Eltern-Block für zusätzliche Projektionen merken
            for parent_name, parent in _iter_parents(root, m):
                try:
                    setattr(m, '_lrp_parent_block', parent)
                except Exception:
                    pass
                break
            found += 1
    if found == 0:
        # Keine Instanzen gefunden
        try:
            already2 = getattr(attn_cache, "_warned_no_msdeform_in_subtree", False) if attn_cache is not None else False
        except Exception:
            already2 = False
        if not already2:
            warnings.warn(
                "attach_msdeformattn_capture: Im gewählten Modul wurden keine MSDeformAttn-Instanzen gefunden. "
                "Encoder-Relevanzen werden ggf. heuristisch (ohne Sampling-Lokationen) zurückgeführt.",
                stacklevel=2,
            )
            try:
                if attn_cache is not None:
                    setattr(attn_cache, "_warned_no_msdeform_in_subtree", True)
            except Exception:
                pass
    
    return handles

# Erfassung der Aktivierungen in einem MSDeformAttn-Modul

def _capture_activations(
    mod: nn.Module,
    query: Tensor,
    reference_points: Tensor,
    input_flatten: Tensor,
    input_spatial_shapes: Tensor,
    input_level_start_index: Tensor,
    input_padding_mask: Optional[Tensor],
    attn_cache: object,
) -> None:
    N, Len_q, _ = query.shape
    N2, S, _ = input_flatten.shape
    if N2 != N:
        raise RuntimeError("MSDeformAttn: Batchgröße inkonsistent") 
    # Value-Projektion und Maskierung
    value = mod.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))
    Hh = mod.n_heads
    Dh = mod.d_model // mod.n_heads
    value_reshaped = value.view(N, S, Hh, Dh)  # (N, S, H, Dh)
    # Sampling-Offsets und Attention-Gewichte berechnen
    sampling_offsets = mod.sampling_offsets(query).view(
        N, Len_q, Hh, mod.n_levels, mod.n_points, 2
    )
    attention_weights = mod.attention_weights(query).view(
        N, Len_q, Hh, mod.n_levels * mod.n_points
    )
    attention_weights = F.softmax(attention_weights, -1).view(
        N, Len_q, Hh, mod.n_levels, mod.n_points
    )
    # Sampling-Lokationen berechnen
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
        )
        sampling_locations = (
            reference_points[:, :, None, :, None, :] + 
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            reference_points[:, :, None, :, None, :2] + 
            sampling_offsets / mod.n_points * 
            reference_points[:, :, None, :, None, 2:] * 0.5
        )
    else:
        sampling_locations = None
    # In den Cache schreiben (detach für Stabilität)
    if sampling_locations is not None:
        setattr(attn_cache, 'deform_sampling_locations', sampling_locations.detach())
        setattr(attn_cache, 'deform_attention_weights', attention_weights.detach())
        setattr(attn_cache, 'deform_spatial_shapes', input_spatial_shapes.detach())
        setattr(attn_cache, 'deform_level_start_index', input_level_start_index.detach())
    # Metadaten
    im2col = getattr(mod, 'im2col_step', None)
    if isinstance(im2col, int):
        setattr(attn_cache, 'deform_im2col_step', im2col)
    # Projektionsgewichte (H, Dh, C)
    try:
        W_v = mod.value_proj.weight.detach()  # (C, C)
        W_o = mod.output_proj.weight.detach()  # (C, C)
        C = W_v.shape[0]
        WVd = W_v.view(C, Hh, Dh).permute(1, 2, 0).contiguous()  # (H, Dh, C)
        WOd = W_o.t().contiguous().view(C, Hh, Dh).permute(1, 2, 0).contiguous()
        setattr(attn_cache, 'W_V_deform', WVd)
        setattr(attn_cache, 'W_O_deform', WOd)
    except Exception:
        pass
    # Parent-Projektionen optional
    parent = getattr(mod, '_lrp_parent_block', None)
    if parent is not None:
        _maybe_capture_parent_projections(parent, attn_cache)

__all__ = [
    "attach_msdeformattn_capture",
    "_resolve_msdeformattn_class",
    "_ForwardPatchHandle",
    "_MSDEFORMATTN_CLS_CANDIDATES",
]
