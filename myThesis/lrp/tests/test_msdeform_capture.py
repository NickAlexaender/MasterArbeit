from __future__ import annotations

import sys
import types
import torch
from torch import nn, Tensor

from myThesis.lrp.lrp.value_path import AttnCache
from myThesis.lrp.calc.msdeformattn_capture import attach_msdeformattn_capture

# Create a fake module path and MSDeformAttn class so we don't depend on compiled ops
mod_name = 'maskdino.layers.ms_deform_attn'
if mod_name not in sys.modules:
    fake_mod = types.ModuleType(mod_name)
    class MSDeformAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.im2col_step = 1
        def forward(self, value: Tensor, spatial_shapes: Tensor, level_start_index: Tensor,
                    sampling_locations: Tensor, attention_weights: Tensor, *args, **kwargs):
            # Return value unchanged; capture happens in wrapper
            return value
    fake_mod.MSDeformAttn = MSDeformAttn
    sys.modules[mod_name] = fake_mod

class ParentBlock(nn.Module):
    def __init__(self, embed_dim: int = 8, num_heads: int = 2):
        super().__init__()
        self.d_model = embed_dim
        self.n_heads = num_heads
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        # child deformable attn
        MSDeformAttn = sys.modules['maskdino.layers.ms_deform_attn'].MSDeformAttn
        self.msda = MSDeformAttn()
    def forward(self, value: Tensor, spatial_shapes: Tensor, level_start_index: Tensor,
                sampling_locations: Tensor, attention_weights: Tensor):
        return self.msda(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)


def run_smoke_test():
    B, Tq, H, L, P = 2, 3, 2, 2, 4
    E = 8
    # build parent
    parent = ParentBlock(embed_dim=E, num_heads=H)
    attn_cache = AttnCache()
    handles = attach_msdeformattn_capture(parent, attn_cache)

    # Fake inputs
    value = torch.randn(B, 5, E)
    spatial_shapes = torch.tensor([[2, 2], [1, 2]], dtype=torch.long)
    level_start_index = torch.tensor([0, 4], dtype=torch.long)
    sampling_locations = torch.rand(B, Tq, H, L, P, 2)
    attention_weights = torch.rand(B, Tq, H, L, P)

    # forward
    _ = parent(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)

    # assertions
    assert isinstance(attn_cache.deform_sampling_locations, torch.Tensor)
    assert isinstance(attn_cache.deform_attention_weights, torch.Tensor)
    assert isinstance(attn_cache.deform_spatial_shapes, torch.Tensor)
    assert isinstance(attn_cache.deform_level_start_index, torch.Tensor)
    # optional parent projections
    assert attn_cache.W_V_deform is None or isinstance(attn_cache.W_V_deform, torch.Tensor)
    assert attn_cache.W_O_deform is None or isinstance(attn_cache.W_O_deform, torch.Tensor)

    # cleanup
    for h in handles:
        h.remove()

    print('MSDeformAttn capture smoke test: OK')


if __name__ == '__main__':
    run_smoke_test()
