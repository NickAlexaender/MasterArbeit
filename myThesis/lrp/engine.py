"""
Rule-based Layer-wise Relevance Propagation (LRP) engine for PyTorch modules.

Scope implemented:
- Linear: epsilon, z+, alpha-beta rules
- Conv2d: epsilon/z+ (implemented via unfold/fold to avoid explicit im2col copies when possible)
- Activations: ReLU/GELU as identity pass-through (redistribute relevance unchanged)
- Normalizations: LayerNorm/FrozenBN as pass-through (optionally account for affine scale)
- Residual: y = x + f(x) split proportionally by contribution magnitude
- MultiheadAttention: value-path propagation (R_out -> attn weights -> V -> proj V)

Notes:
- We avoid autograd gradients. LRP uses forward contributions z_ij with stabilizers.
- Conservation: sum of relevances should be (approximately) preserved per module.
- This engine traces a single forward pass to cache needed tensors, then applies
  rule-based backward relevance propagation.

Limitations:
- Deformable attention is not implemented (left as future work). MultiheadAttention V-path only.
- Batch-first assumed for MHA (PyTorch MHA supports batch_first=True); if not, we convert.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LRPConfig:
    rule_linear: str = "epsilon"  # one of: epsilon, zplus, alphabeta
    rule_conv: str = "epsilon"     # one of: epsilon, zplus
    alpha: float = 1.0             # for alpha-beta
    beta: float = 0.0              # for alpha-beta (alpha - beta = 1 recommended)
    epsilon: float = 1e-6          # stabilizer for epsilon-rule
    conserve_tol: float = 1e-4     # tolerance for conservation checks
    use_affine_norm: bool = False  # if True, apply affine scaling for norms
    verbose_checks: bool = False


class TraceStore:
    """Stores forward-pass caches for modules of interest."""

    def __init__(self):
        # map: module -> dict of cached tensors
        self.caches: Dict[nn.Module, Dict[str, Any]] = {}

    def set(self, module: nn.Module, **tensors: Any):
        self.caches.setdefault(module, {}).update(tensors)

    def get(self, module: nn.Module) -> Dict[str, Any]:
        return self.caches.get(module, {})

    def clear(self):
        self.caches.clear()

    def clear(self):
        """Clear all cached tensors to free memory between images."""
        self.caches.clear()


def _sum_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x) if x.numel() > 0 else torch.tensor(0.0, device=x.device, dtype=x.dtype)


def check_conservation(R_in: torch.Tensor, R_out: torch.Tensor, cfg: LRPConfig, where: str):
    s_in = float(_sum_safe(R_in).detach().cpu())
    s_out = float(_sum_safe(R_out).detach().cpu())
    if abs(s_in - s_out) > cfg.conserve_tol and cfg.verbose_checks:
        print(f"[LRP] Conservation warn at {where}: in={s_in:.6e}, out={s_out:.6e}, diff={s_in - s_out:.2e}")


class LRPTracer:
    """Register minimal forward hooks to cache inputs/weights needed for LRP."""

    def __init__(self, cfg: LRPConfig):
        self.cfg = cfg
        self.store = TraceStore()
        self._handles: List[Any] = []

    def _register_linear(self, m: nn.Linear):
        def fwd_hook(module, inp, out):
            x = inp[0]
            self.store.set(module, x=x.detach(), W=module.weight.detach(), b=(module.bias.detach() if module.bias is not None else None), y=out.detach())
        self._handles.append(m.register_forward_hook(fwd_hook))

    def _register_conv2d(self, m: nn.Conv2d):
        def fwd_hook(module, inp, out):
            x = inp[0]
            self.store.set(module, x=x.detach(), W=module.weight.detach(), b=(module.bias.detach() if module.bias is not None else None), stride=module.stride, padding=module.padding, dilation=module.dilation, groups=module.groups, y=out.detach())
        self._handles.append(m.register_forward_hook(fwd_hook))

    def _register_activation(self, m: nn.Module):
        def fwd_hook(module, inp, out):
            self.store.set(module, x=inp[0].detach(), y=out.detach())
        self._handles.append(m.register_forward_hook(fwd_hook))

    def _register_activation_y_only(self, m: nn.Module):
        def fwd_hook(module, inp, out):
            self.store.set(module, y=out.detach())
        self._handles.append(m.register_forward_hook(fwd_hook))

    def _register_norm(self, m: nn.Module):
        def fwd_hook(module, inp, out):
            cache = {"x": inp[0].detach(), "y": out.detach()}
            if hasattr(module, 'weight') and module.weight is not None:
                cache["w"] = module.weight.detach()
            if hasattr(module, 'bias') and module.bias is not None:
                cache["b"] = module.bias.detach()
            self.store.set(module, **cache)
        self._handles.append(m.register_forward_hook(fwd_hook))

    def _register_norm_y_only(self, m: nn.Module):
        def fwd_hook(module, inp, out):
            self.store.set(module, y=out.detach())
        self._handles.append(m.register_forward_hook(fwd_hook))

    def _register_residual_add(self, add_module: nn.Module):
        # There is no explicit add module; caller can provide a wrapper module if needed.
        def fwd_hook(module, inp, out):
            # inp is a tuple of tensors to be added
            xs = [t.detach() for t in inp]
            self.store.set(module, xs=xs, y=out.detach())
        self._handles.append(add_module.register_forward_hook(fwd_hook))

    def _register_mha(self, m: nn.MultiheadAttention):
        # Works for batch_first=True or False; we record what we see.
        def fwd_hook(module, inp, out):
            # PyTorch MHA forward signature can return (attn_output, attn_output_weights)
            if isinstance(out, tuple):
                y, A = out
            else:
                y = out
                A = None
            # inputs: query, key, value
            q, k, v = inp[:3]
            cache = {
                "q": q.detach(), "k": k.detach(), "v": v.detach(), "y": y.detach(),
                "embed_dim": module.embed_dim, "num_heads": module.num_heads,
                "in_proj_weight": module.in_proj_weight.detach() if module.in_proj_weight is not None else None,
                "in_proj_bias": module.in_proj_bias.detach() if module.in_proj_bias is not None else None,
                "proj_weight": module.out_proj.weight.detach(),
                "proj_bias": module.out_proj.bias.detach() if module.out_proj.bias is not None else None,
                "batch_first": getattr(module, 'batch_first', False),
                "attn_weights": (A.detach() if isinstance(A, torch.Tensor) else None),
            }
            self.store.set(module, **cache)
        self._handles.append(m.register_forward_hook(fwd_hook))

    def _register_mha_y_only(self, m: nn.MultiheadAttention):
        def fwd_hook(module, inp, out):
            y = out[0] if isinstance(out, tuple) else out
            self.store.set(module, y=y.detach())
        self._handles.append(m.register_forward_hook(fwd_hook))

    def add_module(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            self._register_linear(m)
        elif isinstance(m, nn.Conv2d):
            self._register_conv2d(m)
        elif isinstance(m, (nn.ReLU, nn.GELU)):
            self._register_activation(m)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            self._register_norm(m)
        elif isinstance(m, nn.MultiheadAttention):
            self._register_mha(m)
        # Residual add: to be provided by caller if a distinct module exists

    def add_from_model(self, model: nn.Module):
        for m in model.modules():
            self.add_module(m)

    def add_module_y_only(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            # y-only: avoid storing x/weights, only output
            def fwd_hook(module, inp, out):
                self.store.set(module, y=out.detach())
            self._handles.append(m.register_forward_hook(fwd_hook))
        elif isinstance(m, nn.Conv2d):
            def fwd_hook(module, inp, out):
                self.store.set(module, y=out.detach())
            self._handles.append(m.register_forward_hook(fwd_hook))
        elif isinstance(m, (nn.ReLU, nn.GELU)):
            self._register_activation_y_only(m)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            self._register_norm_y_only(m)
        elif isinstance(m, nn.MultiheadAttention):
            self._register_mha_y_only(m)

    def add_from_module_y_only(self, module: nn.Module):
        for m in module.modules():
            self.add_module_y_only(m)

    def remove(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()


class LRPPropagator:
    def __init__(self, tracer: LRPTracer, cfg: LRPConfig):
        self.tracer = tracer
        self.cfg = cfg

    # -------- Core rules --------
    def linear(self, module: nn.Linear, R_out: torch.Tensor) -> torch.Tensor:
        cache = self.tracer.store.get(module)
        x: torch.Tensor = cache["x"]
        W: torch.Tensor = cache["W"]
        b: Optional[torch.Tensor] = cache.get("b")

        # x: (B, I), W: (O, I), y: (B, O), R_out: (B, O)
        # contributions z_ij = x_i * W_ji
        W_T = W.t()  # (I, O)
        rule = self.cfg.rule_linear.lower()

        if rule == "zplus":
            W_pos = torch.clamp(W_T, min=0.0)
            z = x @ W_pos  # (B, O)
            s = z + self.cfg.epsilon * torch.sign(z)  # stabilize
            s = s + 1e-12
            msg = (R_out / s)  # (B, O)
            R_in = (msg @ W_pos.t()) * x  # ((B,O)@(O,I) -> (B,I)) * x
        elif rule == "alphabeta":
            alpha, beta = self.cfg.alpha, self.cfg.beta
            W_pos = torch.clamp(W_T, min=0.0)
            W_neg = torch.clamp(W_T, max=0.0)
            z_pos = x @ W_pos  # (B,O) = s_pos
            z_neg = x @ W_neg  # (B,O) = s_neg (<=0)
            s_pos = z_pos + self.cfg.epsilon * torch.sign(z_pos)
            s_neg = z_neg - self.cfg.epsilon * torch.sign(z_neg)
            s_pos = s_pos + 1e-12
            s_neg = s_neg - 1e-12
            # messages per output neuron
            msg_pos = (alpha * (R_out / s_pos))
            msg_neg = (beta * (R_out / s_neg))
            R_in = ((msg_pos @ W_pos.t()) + (msg_neg @ W_neg.t())) * x
        else:  # epsilon
            z = x @ W_T  # (B, O)
            s = z + self.cfg.epsilon * torch.sign(z)
            s = s + 1e-12
            msg = R_out / s
            R_in = (msg @ W) * x  # note: (B,O)@(O,I)->(B,I)

        check_conservation(R_in, R_out, self.cfg, where=f"Linear[{id(module)}]")
        return R_in

    def conv2d(self, module: nn.Conv2d, R_out: torch.Tensor) -> torch.Tensor:
        cache = self.tracer.store.get(module)
        x: torch.Tensor = cache["x"]  # (B, C_in, H, W)
        W: torch.Tensor = cache["W"]  # (C_out, C_in/groups, kH, kW)
        stride = cache["stride"]
        padding = cache["padding"]
        dilation = cache["dilation"]
        groups = cache["groups"]

        rule = self.cfg.rule_conv.lower()
        # forward z = conv(x, W)
        if rule == "zplus":
            W_pos = torch.clamp(W, min=0.0)
            z = F.conv2d(x.clamp(min=0.0), W_pos, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
            s = z + self.cfg.epsilon * torch.sign(z)
            s = s + 1e-12
            msg = R_out / s  # (B, C_out, H', W')
            # backprop relevance to input via transposed conv with positive weights
            R_in = F.conv_transpose2d(msg, W_pos, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups) * x.clamp(min=0.0)
        else:  # epsilon
            z = F.conv2d(x, W, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
            s = z + self.cfg.epsilon * torch.sign(z)
            s = s + 1e-12
            msg = R_out / s
            R_in = F.conv_transpose2d(msg, W, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups) * x

        check_conservation(R_in, R_out, self.cfg, where=f"Conv2d[{id(module)}]")
        return R_in

    def activation(self, module: nn.Module, R_out: torch.Tensor) -> torch.Tensor:
        cache = self.tracer.store.get(module)
        x: torch.Tensor = cache["x"]
        # pass-through; optionally gate by derivative (for ReLU same as masking negatives)
        if isinstance(module, nn.ReLU):
            mask = (x > 0).to(dtype=R_out.dtype)
            R_in = R_out * mask
        else:
            # GELU or others: approximate identity pass-through
            R_in = R_out
        check_conservation(R_in, R_out, self.cfg, where=f"Act[{type(module).__name__}:{id(module)}]")
        return R_in

    def norm(self, module: nn.Module, R_out: torch.Tensor) -> torch.Tensor:
        cache = self.tracer.store.get(module)
        x: torch.Tensor = cache["x"]
        if self.cfg.use_affine_norm and hasattr(module, 'weight') and module.weight is not None:
            w = cache.get("w")
            # scale relevance inversely to weight magnitude to conserve
            scale = 1.0 / (torch.abs(w) + 1e-12)
            # reshape scale to input shape
            while scale.dim() < x.dim():
                scale = scale.view(*([1] * (x.dim() - scale.dim())), *scale.shape)
            R_in = R_out * scale
        else:
            R_in = R_out  # neutral pass-through
        check_conservation(R_in, R_out, self.cfg, where=f"Norm[{type(module).__name__}:{id(module)}]")
        return R_in

    def residual_add(self, module: nn.Module, R_out: torch.Tensor) -> List[torch.Tensor]:
        cache = self.tracer.store.get(module)
        xs: List[torch.Tensor] = cache["xs"]
        # Split relevance proportional to absolute contribution magnitudes at forward
        mags = [x.abs().sum().clamp_min(1e-12) for x in xs]
        total = torch.stack([m.detach() for m in mags]).sum()
        parts = [float(m / total) for m in mags]
        R_ins = [R_out * p for p in parts]
        # conservation by construction
        return R_ins

    def multihead_attention(self, module: nn.MultiheadAttention, R_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache = self.tracer.store.get(module)
        q: torch.Tensor = cache["q"]
        k: torch.Tensor = cache["k"]
        v: torch.Tensor = cache["v"]
        y: torch.Tensor = cache["y"]
        A: Optional[torch.Tensor] = cache.get("attn_weights")
        in_proj_weight = cache.get("in_proj_weight")
        in_proj_bias = cache.get("in_proj_bias")
        proj_weight: torch.Tensor = cache["proj_weight"]
        proj_bias: Optional[torch.Tensor] = cache.get("proj_bias")
        batch_first: bool = cache.get("batch_first", False)

        # shape normalize to (B, T, C)
        def to_btc(t: torch.Tensor) -> torch.Tensor:
            if not batch_first:
                return t.transpose(0, 1)  # (T, B, C) -> (B, T, C)
            return t

        q, k, v, y = map(to_btc, (q, k, v, y))
        B, T, C = y.shape
        H = module.num_heads
        d = C // H

        # Out projection: y = concat(heads) @ W_o + b
        # Propagate R_out back through W_o with epsilon rule over features
        R_y = R_out
        W_o = proj_weight.t()  # (C, C)
        z = torch.einsum('btc,co->bto', y, W_o)
        s = z + self.cfg.epsilon * torch.sign(z)
        s = s + 1e-12
        msg = R_y / s
        R_heads_concat = torch.einsum('bto,oc->btc', msg, proj_weight) * y  # back to y space conservatively

        # Split into heads: (B,T,H,d)
        R_heads = R_heads_concat.view(B, T, H, d)

        # If attention weights A are available (B, num_heads, T_q, T_k) in PyTorch
        # we can propagate along A to V positions. Otherwise we approximate uniform split.
        if A is None:
            # Assume self-attention with T_q=T_k=T; spread R equally over keys
            A = torch.full((B, H, T, T), 1.0 / T, device=R_out.device, dtype=R_out.dtype)
        else:
            if not batch_first:
                # PyTorch returns (B*H, T_q, T_k) sometimes; attempt to reshape
                if A.dim() == 3 and A.shape[0] == B * H:
                    A = A.view(B, H, A.shape[1], A.shape[2])

        # Relevance from output tokens to value tokens per head
        # R_head_out: (B, T_q, H, d) -> distribute over keys using A
        R_head_out = R_heads  # (B,T,H,d)
        # Token-level distribution, ignore feature mixing inside head for A step: sum over d
        R_tok = R_head_out.sum(dim=-1)  # (B, T, H)
        # Propagate over attention map: for each head, distribute R_tok over keys with A
        # R_key_tok[b,h,t_k] = sum_tq R_tok[b,tq,h] * A[b,h,tq,t_k]
        R_key_tok = torch.einsum('bth,bhtk->bhk', R_tok, A)  # (B,H,T_k)

        # Now map key-token relevance back to value features v: share equally across d or weight by v magnitude
        v_bthd = v.view(B, T, H, d)
        v_mag = v_bthd.abs() + 1e-12
        v_mag_sum = v_mag.sum(dim=-1, keepdim=True)  # (B,T,H,1)
        share = v_mag / v_mag_sum  # (B,T,H,d)
        R_v = share * R_key_tok.permute(0, 2, 1).unsqueeze(-1)  # (B,T,H,d)
        R_v = R_v.reshape(B, T, C)

        # Back through value projection (part of in_proj_weight): v_proj = X @ W_v
        if in_proj_weight is not None:
            # in_proj_weight: (3*C, C); slice value part
            W_qkv = in_proj_weight
            W_v = W_qkv[2*C:3*C, :]  # (C, C)
            W_v_T = W_v.t()
            z_v = torch.einsum('btc,co->bto', v, W_v)
            s_v = z_v + self.cfg.epsilon * torch.sign(z_v)
            s_v = s_v + 1e-12
            msg_v = R_v / s_v
            R_x = torch.einsum('bto,oc->btc', msg_v, W_v) * v  # relevance to value input features
        else:
            # If weights unavailable, return R_v as relevance on v
            R_x = R_v

        check_conservation(R_x, R_out, self.cfg, where=f"MHA[{id(module)}]")
        return R_x, torch.zeros_like(q), torch.zeros_like(k)


def propagate_lrp(model: nn.Module, tracer: LRPTracer, target_module: nn.Module, target_tensor: torch.Tensor, cfg: LRPConfig) -> Dict[nn.Module, torch.Tensor]:
    """Starting from a target relevance at target_module output shape, propagate back to cached modules.

    Returns a dict mapping module -> relevance at its input shape. Only modules with caches are returned.
    """
    prop = LRPPropagator(tracer, cfg)
    relevances: Dict[nn.Module, torch.Tensor] = {target_module: target_tensor}

    # Traverse modules in reverse registration order for a simple chain; for complex DAGs
    # users should explicitly call per-layer using this engine's primitives.
    for m in list(tracer.store.caches.keys())[::-1]:
        if m not in relevances:
            continue
        R_out = relevances[m]
        R_in = None
        if isinstance(m, nn.Linear):
            R_in = prop.linear(m, R_out)
        elif isinstance(m, nn.Conv2d):
            R_in = prop.conv2d(m, R_out)
        elif isinstance(m, (nn.ReLU, nn.GELU)):
            R_in = prop.activation(m, R_out)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            R_in = prop.norm(m, R_out)
        elif isinstance(m, nn.MultiheadAttention):
            R_x, R_q, R_k = prop.multihead_attention(m, R_out)
            # By default return value-input relevance
            R_in = R_x
        else:
            # If module is unsupported, pass-through if shapes match
            cache = tracer.store.get(m)
            x = cache.get("x")
            if x is not None and x.shape == R_out.shape:
                R_in = R_out
        if R_in is not None:
            relevances[m] = R_in
    return relevances


# -------- Minimal tests (can be used by caller) --------
def _test_mlp_conservation(device="cpu"):
    torch.manual_seed(0)
    mlp = nn.Sequential(
        nn.Linear(4, 3, bias=False),
        nn.ReLU(),
        nn.Linear(3, 2, bias=False),
    ).to(device)
    x = torch.randn(5, 4, device=device)
    cfg = LRPConfig(rule_linear="epsilon", epsilon=1e-6, verbose_checks=True)
    tracer = LRPTracer(cfg)
    tracer.add_from_model(mlp)
    with torch.no_grad():
        y = mlp(x)
    # target relevance as output activations (positive)
    R_out = F.relu(y)
    # Start from last Linear
    last_lin = [m for m in mlp.modules() if isinstance(m, nn.Linear)][-1]
    rels = propagate_lrp(mlp, tracer, last_lin, R_out, cfg)
    # Check sums
    Rin = rels.get(last_lin)
    assert Rin is not None
    s_in = float(Rin.sum().cpu())
    s_out = float(R_out.sum().cpu())
    return abs(s_in - s_out)


if __name__ == "__main__":
    diff = _test_mlp_conservation("cpu")
    print("MLP conservation diff:", diff)
