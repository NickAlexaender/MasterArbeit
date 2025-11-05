"""
Hook helpers used by the analysis code.
"""
from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
from torch import Tensor

from myThesis.lrp.calc.tensor_utils import _first_tensor

# LRP Engine import
try:
    from myThesis.lrp.engine import LRPTracer  # type: ignore
except Exception:
    # Fallback for direct script execution context
    import os, sys
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _parent = os.path.dirname(os.path.dirname(_this_dir))
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from myThesis.lrp.engine import LRPTracer  # type: ignore


class LayerCapture:
    """Legacy placeholder kept for compatibility (no longer used with LRP).

    The LRP engine uses its own tracer; this class is retained to minimize diffs
    if external code still imports it. It does nothing.
    """
    def __init__(self, module: nn.Module):
        self.module = module
        self.inputs_flat: List[Tensor] = []
        self.grad_inputs_flat: List[Optional[Tensor]] = []
        self.output_tensor: Optional[Tensor] = None

    def remove(self):
        pass


class _LayerTap:
    """Registers a temporary forward hook on a chosen layer to cache x/y in the LRP tracer
    when the tracer didn't register it (unsupported module types)."""
    def __init__(self, tracer: LRPTracer, module: nn.Module, retain_grad: bool = False):
        self.tracer = tracer
        self.module = module
        self.retain_grad = retain_grad
        self._h = module.register_forward_hook(self._on_forward)

    def _on_forward(self, mod: nn.Module, inp, out):
        try:
            x = _first_tensor(inp)
            y = _first_tensor(out)
            # Falls Gradpfad gewünscht ist: Grad auf x behalten
            if self.retain_grad and isinstance(x, torch.Tensor) and x.requires_grad:
                x.retain_grad()
            self.tracer.store.set(mod, x=x, y=y)
        except Exception:
            pass

    def remove(self):
        try:
            self._h.remove()
        except Exception:
            pass
