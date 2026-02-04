from __future__ import annotations
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_, constant_
from .lrp_param_base import LRPActivations, LRPModuleMixin


# Wir erstellen einen Linear-Layer, der aktive LRP-Modus Eingaben und Gewichte speichert.

class LRP_Linear(nn.Linear, LRPModuleMixin):

    
    def forward(self, input: Tensor) -> Tensor:
        output = F.linear(input, self.weight, self.bias)
        
        if self._is_lrp:
            self.activations.input = input.detach()
            self.activations.output = output.detach()
            self.activations.weights = self.weight.detach()
            if self.bias is not None:
                self.activations.bias = self.bias.detach()
        
        return output
    
    @classmethod
    def from_linear(cls, linear: nn.Linear) -> 'LRP_Linear':
        lrp_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        lrp_linear.weight = linear.weight
        if linear.bias is not None:
            lrp_linear.bias = linear.bias
        return lrp_linear


# Alle Parameter und Statistiken von LayerNorm sollen hier gespeichert werden.

class LRP_LayerNorm(nn.LayerNorm, LRPModuleMixin):
    
    def forward(self, input: Tensor) -> Tensor:
        if self._is_lrp:
            # Manuelle Berechnung für Zugriff auf mean/var
            dims = tuple(range(-len(self.normalized_shape), 0))
            mean = input.mean(dim=dims, keepdim=True)
            var = input.var(dim=dims, unbiased=False, keepdim=True)
            
            x_norm = (input - mean) / torch.sqrt(var + self.eps)
            
            if self.elementwise_affine:
                output = x_norm * self.weight + self.bias
            else:
                output = x_norm
            
            # Aktivierungen speichern
            self.activations.input = input.detach()
            self.activations.output = output.detach()
            self.activations.mean = mean.detach()
            self.activations.var = var.detach()
            if self.elementwise_affine:
                self.activations.gamma = self.weight.detach()
                self.activations.beta = self.bias.detach() if self.bias is not None else None
            
            return output
        else:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )
    
    @classmethod
    def from_layernorm(cls, ln: nn.LayerNorm) -> 'LRP_LayerNorm':
        lrp_ln = cls(
            normalized_shape=ln.normalized_shape,
            eps=ln.eps,
            elementwise_affine=ln.elementwise_affine,
            device=ln.weight.device if ln.weight is not None else None,
            dtype=ln.weight.dtype if ln.weight is not None else None,
        )
        if ln.elementwise_affine:
            lrp_ln.weight = ln.weight
            if ln.bias is not None:
                lrp_ln.bias = ln.bias
        return lrp_ln


# Für MultiheadAttention speichern wir wie folgt alle relevanten Zwischenwerte.

class LRP_MultiheadAttention(nn.MultiheadAttention, LRPModuleMixin):
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        
        if self._is_lrp:
            # Erzwinge vollständige Gewichte
            need_weights = True
            average_attn_weights = False
        
        # Original forward
        attn_output, attn_weights_out = super().forward(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )
        
        if self._is_lrp:
            self._capture_activations(query, key, value, attn_output, attn_weights_out)
        
        return attn_output, attn_weights_out
    
    def _capture_activations(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_output: Tensor,
        attn_weights: Optional[Tensor],
    ):
        batch_first = getattr(self, 'batch_first', False)
        embed_dim = self.embed_dim
        num_heads = self.num_heads
        head_dim = embed_dim // num_heads
        
        # Zu (B, T, C) konvertieren
        def to_btc(t: Tensor) -> Tensor:
            if t.dim() == 3 and not batch_first:
                return t.transpose(0, 1).contiguous()
            return t
        
        q_btc = to_btc(query)
        k_btc = to_btc(key)
        v_btc = to_btc(value)
        
        B = q_btc.shape[0]
        T = q_btc.shape[1]
        S = k_btc.shape[1]
        
        # Q, K, V Projektionen berechnen
        W_Q, W_K, W_V = self._get_qkv_weights()
        
        Q_proj = F.linear(q_btc, W_Q)  # (B, T, E)
        K_proj = F.linear(k_btc, W_K)  # (B, S, E)
        V_proj = F.linear(v_btc, W_V)  # (B, S, E)
        
        # Reshape zu (B, H, T/S, Dh)
        Q_heads = Q_proj.view(B, T, num_heads, head_dim).permute(0, 2, 1, 3)
        K_heads = K_proj.view(B, S, num_heads, head_dim).permute(0, 2, 1, 3)
        V_heads = V_proj.view(B, S, num_heads, head_dim).permute(0, 2, 1, 3)
        
        # Attention Scores berechnen (für LRP Softmax-Pfad)
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) * scale
        
        # Aktivierungen speichern
        self.activations.input = (q_btc.detach(), k_btc.detach(), v_btc.detach())
        self.activations.output = to_btc(attn_output).detach()
        self.activations.Q = Q_heads.detach()
        self.activations.K = K_heads.detach()
        self.activations.V = V_heads.detach()
        self.activations.attn_scores = attn_scores.detach()
        self.activations.scale = scale
        
        if attn_weights is not None:
            # Normalisiere Shape zu (B, H, T, S)
            aw = attn_weights
            if aw.dim() == 3:
                # (B, T, S) -> (B, 1, T, S)
                aw = aw.unsqueeze(1)
            self.activations.attn_weights = aw.detach()
        
        # Projektionsgewichte speichern (H, Dh, C)
        self.activations.W_Q = W_Q.view(num_heads, head_dim, embed_dim).detach()
        self.activations.W_K = W_K.view(num_heads, head_dim, embed_dim).detach()
        self.activations.W_V = W_V.view(num_heads, head_dim, embed_dim).detach()
        
        W_O = self.out_proj.weight  # (E, E)
        self.activations.W_O = W_O.t().view(embed_dim, num_heads, head_dim).permute(1, 2, 0).detach()
    
    def _get_qkv_weights(self) -> Tuple[Tensor, Tensor, Tensor]:
        E = self.embed_dim
        
        if self.in_proj_weight is not None:
            # Combined QKV weights
            W_Q = self.in_proj_weight[:E, :]
            W_K = self.in_proj_weight[E:2*E, :]
            W_V = self.in_proj_weight[2*E:3*E, :]
        else:
            # Separate Q, K, V projections
            W_Q = self.q_proj_weight
            W_K = self.k_proj_weight
            W_V = self.v_proj_weight
        
        return W_Q, W_K, W_V
    
    @classmethod
    def from_mha(cls, mha: nn.MultiheadAttention) -> 'LRP_MultiheadAttention':
        lrp_mha = cls(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            bias=mha.in_proj_bias is not None,
            add_bias_kv=mha.bias_k is not None,
            add_zero_attn=mha.add_zero_attn,
            kdim=mha.kdim,
            vdim=mha.vdim,
            batch_first=getattr(mha, 'batch_first', False),
            device=mha.in_proj_weight.device if mha.in_proj_weight is not None else None,
            dtype=mha.in_proj_weight.dtype if mha.in_proj_weight is not None else None,
        )
        
        # Gewichte kopieren
        if mha.in_proj_weight is not None:
            lrp_mha.in_proj_weight = mha.in_proj_weight
        if mha.in_proj_bias is not None:
            lrp_mha.in_proj_bias = mha.in_proj_bias
        lrp_mha.out_proj.weight = mha.out_proj.weight
        if mha.out_proj.bias is not None:
            lrp_mha.out_proj.bias = mha.out_proj.bias
        if mha.bias_k is not None:
            lrp_mha.bias_k = mha.bias_k
        if mha.bias_v is not None:
            lrp_mha.bias_v = mha.bias_v
        
        return lrp_mha


# Multi-Scale Deformable Attention mit LRP-Aktivierungsspeicherung

class LRP_MSDeformAttn(nn.Module, LRPModuleMixin):
    
    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by n_heads ({n_heads})')
        
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self._d_per_head = d_model // n_heads
        
        self.im2col_step = 128
        
        # Projektions-Layer
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)
    
    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        input_spatial_shapes: Tensor,
        input_level_start_index: Tensor,
        input_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        
        # Value Projection
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)
        value = value.view(N, Len_in, self.n_heads, self._d_per_head)
        
        # Sampling Offsets und Attention Weights
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        
        # Sampling Locations berechnen
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points
                * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f'reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}')
        
        # Deformable Attention Kernel (PyTorch-Fallback)
        output = self._ms_deform_attn_core(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )
        
        # Output Projection
        output = self.output_proj(output)
        
        # LRP-Aktivierungen speichern
        if self._is_lrp:
            self._capture_activations(
                query, reference_points, input_flatten, input_spatial_shapes,
                input_level_start_index, value, sampling_locations, attention_weights, output
            )
        
        return output
    
    # Deformable Attention Kernelfunktion
    
    def _ms_deform_attn_core(
        self,
        value: Tensor,
        spatial_shapes: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        N, S, H, Dh = value.shape
        _, Len_q, _, L, P, _ = sampling_locations.shape
        
        # Values pro Level aufteilen
        value_list = value.split(
            [int(h * w) for h, w in spatial_shapes.tolist()], dim=1
        )
        
        sampling_grids = 2 * sampling_locations - 1  # [0,1] -> [-1, 1] für grid_sample
        
        sampling_value_list = []
        for lvl, (H_l, W_l) in enumerate(spatial_shapes.tolist()):
            H_l, W_l = int(H_l), int(W_l)
            
            # Value für dieses Level: (N, H_l*W_l, H, Dh) -> (N*H, Dh, H_l, W_l)
            value_l = value_list[lvl].permute(0, 2, 3, 1).reshape(N * H, Dh, H_l, W_l)
            
            # Sampling Grid für dieses Level: (N, T, H, P, 2) -> (N*H, T, P, 2)
            sampling_grid_l = sampling_grids[:, :, :, lvl, :, :]
            sampling_grid_l = sampling_grid_l.permute(0, 2, 1, 3, 4).reshape(N * H, Len_q, P, 2)
            
            # Bilineare Interpolation
            sampled = F.grid_sample(
                value_l,
                sampling_grid_l,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )  # (N*H, Dh, T, P)
            
            sampling_value_list.append(sampled)
        
        # Attention über alle Levels und Points
        # attention_weights: (N, T, H, L, P) -> (N, H, T, L*P) -> (N*H, 1, T, L*P)
        attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(N * H, 1, Len_q, L * P)
        
        # sampling_values: (N*H, Dh, T, L*P)
        sampling_values = torch.stack(sampling_value_list, dim=-1).reshape(N * H, Dh, Len_q, L * P)
        
        # Gewichtete Summe: (N*H, Dh, T, L*P) * (N*H, 1, T, L*P) -> (N*H, Dh, T)
        output = (sampling_values * attention_weights).sum(-1)
        
        # Reshape: (N*H, Dh, T) -> (N, H, Dh, T) -> (N, T, H*Dh) = (N, T, C)
        output = output.view(N, H, Dh, Len_q).permute(0, 3, 1, 2).reshape(N, Len_q, H * Dh)
        
        return output
    
    def _capture_activations(
        self,
        query: Tensor,
        reference_points: Tensor,
        input_flatten: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        value: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
        output: Tensor,
    ):
        self.activations.query = query.detach()
        self.activations.reference_points = reference_points.detach()
        self.activations.input_flatten = input_flatten.detach()
        self.activations.spatial_shapes = spatial_shapes.detach()
        self.activations.level_start_index = level_start_index.detach()
        self.activations.value_proj = value.detach()  # (N, S, H, Dh)
        self.activations.sampling_locations = sampling_locations.detach()
        self.activations.deform_attention_weights = attention_weights.detach()
        self.activations.output = output.detach()
        
        #Setze auch activations.input für die generische Propagator-Prüfung
        # Für MSDeformAttn ist input_flatten der Haupteingang
        self.activations.input = input_flatten.detach()
        
        # Projektionsgewichte (H, Dh, C)
        C = self.d_model
        H = self.n_heads
        Dh = self._d_per_head
        
        W_V = self.value_proj.weight.detach()  # (C, C)
        W_O = self.output_proj.weight.detach()  # (C, C)
        
        self.activations.W_V = W_V.view(C, H, Dh).permute(1, 2, 0).contiguous()
        self.activations.W_O = W_O.t().view(C, H, Dh).permute(1, 2, 0).contiguous()
    
    @classmethod
    def from_msdeformattn(cls, msda: nn.Module) -> 'LRP_MSDeformAttn':
        lrp_msda = cls(
            d_model=msda.d_model,
            n_levels=msda.n_levels,
            n_heads=msda.n_heads,
            n_points=msda.n_points,
        )
        
        # Gewichte kopieren
        lrp_msda.sampling_offsets.load_state_dict(msda.sampling_offsets.state_dict())
        lrp_msda.attention_weights.load_state_dict(msda.attention_weights.state_dict())
        lrp_msda.value_proj.load_state_dict(msda.value_proj.state_dict())
        lrp_msda.output_proj.load_state_dict(msda.output_proj.state_dict())
        
        # im2col_step übernehmen
        if hasattr(msda, 'im2col_step'):
            lrp_msda.im2col_step = msda.im2col_step
        
        return lrp_msda

__all__ = [
    "LRP_Linear",
    "LRP_LayerNorm",
    "LRP_MultiheadAttention",
    "LRP_MSDeformAttn",
]
