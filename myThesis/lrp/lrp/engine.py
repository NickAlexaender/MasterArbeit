"""
LRPEngine: Orchestriert den lokalen Relevanzfluss von Layer L nach L-1.

Die Engine ist so gebaut, dass sie keine CLI-Signaturen ändert und mit Hooks
(cps: CutPoints) aus hooks_maskdino.py arbeitet. Für den ersten Integrationsschritt
wird der SubLayer-Pfad als Identität behandelt; die Residual-Aufteilung ist implementiert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from .rules import residual_split, value_path_split
from .config import RESIDUAL_SPLIT, TARGET_TOKEN_IDX

# Für robuste Tensorform-Konvertierung (B,T,C) – zentralisiert
from myThesis.lrp.calc.tensor_utils import _to_BTC, build_target_relevance


@dataclass
class LRPResult:
    R_prev: Tensor       # Relevanz auf x^{L-1} (B,T,C)
    g_spec: str          # Dokumentation der Skalarisierung


class LRPEngine:
    def __init__(self, epsilon: float = 1e-6):
        self.eps = float(epsilon)

    def run_local(
        self,
        cps,
        feature_index: int,
        target_token_idx: int = TARGET_TOKEN_IDX,
        norm: str = "sum1",
        which_module: str = "decoder",
        use_sublayer: str = "self_attn",
        measurement_point: str = "post_res",
        attn_cache: Optional[object] = None,
        conservative_residual: bool = False,
        # Neu: Zielauswahl-Modus und Value-Pfad-Config
        index_axis: str = "channel",   # {"channel","token"}
        token_reduce: str = "mean",     # {"mean","max"} wenn index_axis=="channel"
        value_path_abs: bool = True,     # Steuert |.| in value_path_split
    ) -> LRPResult:
        """Berechne lokale Attribution von L nach L-1.

        Aktuelle Implementierung: Zerlegung über Residual, SubLayer Pfad = Identität.
        Damit ist der Fluss bereits sinnvoll normiert und stabil – die detaillierte
        SubLayer-Zerlegung (Attention/FFN) kann schrittweise ergänzt werden.
        """
        if cps is None or cps.x_prev is None or cps.y_curr is None:
            raise RuntimeError("CutPoints sind nicht befüllt – wurden die Hooks ausgelöst?")

        x_prev = _to_BTC(cps.x_prev.detach())   # (B,T,C)
        y_curr = _to_BTC(cps.y_curr.detach())   # (B,T,C)
        # Heuristische Layout-Rettung: falls Batch-Dimensionen nicht passen, versuche Transponieren
        if x_prev.shape[0] != y_curr.shape[0]:
            if x_prev.dim() == 3 and x_prev.shape[1] == y_curr.shape[0]:
                x_prev = x_prev.transpose(0, 1).contiguous()
            elif y_curr.dim() == 3 and y_curr.shape[1] == x_prev.shape[0]:
                y_curr = y_curr.transpose(0, 1).contiguous()
        B, T, C = y_curr.shape

        # Token-Wahl (für Fallback/FFN):
        # - Wenn index_axis=="token": t = feature_index
        # - sonst: t = target_token_idx (>=0), oder auto (max Energie) wenn <0
        if index_axis == "token":
            t_sel = int(feature_index)
            if not (0 <= t_sel < T):
                raise IndexError(f"token feature_index {t_sel} außerhalb [0,{T-1}] (axis=token)")
        else:
            if target_token_idx >= 0:
                t_sel = int(target_token_idx)
                if not (0 <= t_sel < T):
                    raise IndexError(f"TARGET_TOKEN_IDX {t_sel} außerhalb [0,{T-1}]")
            else:
                # Auto-Token: argmax mittlerer Token-Energie in y_curr
                token_energy = (y_curr.pow(2)).sum(dim=-1)          # (B,T)
                t_sel = int(token_energy.mean(dim=0).argmax().item())

        # Kanal-Check für channel-Modus
        if index_axis == "channel":
            c_sel = int(feature_index)
            if not (0 <= c_sel < C):
                raise IndexError(f"feature_index {c_sel} außerhalb [0,{C-1}] (axis=channel)")

        # 1) Startrelevanz basierend auf index_axis
        #    Wir nutzen immer target_norm="sum1" für stabile Skalierung
        Ro = build_target_relevance(
            layer_output=y_curr,
            feature_index=feature_index,
            token_reduce=token_reduce,
            target_norm="sum1",
            index_axis=index_axis,
        )

        # 2) Residual-Aufteilung y = x + F(x)
        #    Für Attention- und FFN-SubLayer attribuieren wir ausschließlich über den Transform-Pfad,
        #    um Self-/Diagonal-Dominanz durch den Skip zu vermeiden.
        only_transform = use_sublayer in ("self_attn", "cross_attn", "ffn")
        if only_transform:
            if conservative_residual:
                if measurement_point == "post_res":
                    xw = x_prev - x_prev.mean(dim=-1, keepdim=True)
                    yw = y_curr - y_curr.mean(dim=-1, keepdim=True)
                    Fx = (yw - xw)
                    rx, rFx = residual_split(xw, Fx, Ro, mode=RESIDUAL_SPLIT)
                else:
                    Fx = (y_curr - x_prev)
                    rx, rFx = residual_split(x_prev, Fx, Ro, mode=RESIDUAL_SPLIT)
            else:
                rx = torch.zeros_like(Ro)
                rFx = Ro
        else:
            if measurement_point == "post_res":
                xw = x_prev - x_prev.mean(dim=-1, keepdim=True)
                yw = y_curr - y_curr.mean(dim=-1, keepdim=True)
                Fx = (yw - xw)
                rx, rFx = residual_split(xw, Fx, Ro, mode=RESIDUAL_SPLIT)
            else:
                Fx = (y_curr - x_prev)
                rx, rFx = residual_split(x_prev, Fx, Ro, mode=RESIDUAL_SPLIT)

        # 3) Verteile die Transform-Pfad-Relevanz rFx zurück auf x_prev.
        #    Bevorzugt über Attention-Gewichte (Value-Pfad Approximation),
        #    Fallback: robuste token-/kanalweise Gewichte basierend auf Energie/|x|.
        # Gesamt-Relevanzanteil des Transform-Pfads als Skalar pro Batch
        rFx_scalar = rFx.sum(dim=(1, 2), keepdim=True)  # (B,1,1)

        def _channel_weights_absx(xbtc: Tensor) -> Tensor:
            ch_abs_ = xbtc.abs() + 1e-12
            return ch_abs_ / (ch_abs_.sum(dim=-1, keepdim=True) + 1e-12)

        R_fx_to_x: Tensor
        used_value_path = False
        if attn_cache is not None and index_axis == "channel":
            try:
                aw = getattr(attn_cache, "attn_weights", None)
                Vproj = getattr(attn_cache, "Vproj", None)
                W_O = getattr(attn_cache, "W_O", None)
                W_V = getattr(attn_cache, "W_V", None)
                if isinstance(aw, torch.Tensor) and isinstance(Vproj, torch.Tensor) and isinstance(W_O, torch.Tensor):
                    # Erwartete Shapes: aw (B,H,T,S), Vproj (B,H,S,Dh), W_O (H,Dh,C)
                    if aw.dim() == 4:
                        aw4 = aw
                    elif aw.dim() == 3:
                        aw4 = aw.unsqueeze(1)  # (B,1,T,S)
                    else:
                        aw4 = None
                    if aw4 is not None:
                        # Relevanz auf Kanal c über alle Ziel-Tokens (hier nur t* belegt)
                        Ro_j = torch.zeros((B, aw4.shape[2], 1), device=y_curr.device, dtype=y_curr.dtype)
                        Ro_j[:, t_sel, 0] = rFx_scalar.view(B)  # (B,T,1)
                        W_O_j = W_O[:, :, c_sel:c_sel+1]  # (H,Dh,1)
                        Rs = value_path_split(Ro_j, aw4, Vproj, W_O_j, use_abs=value_path_abs)  # (B,S,1)
                        # Cross-Attention: Attribution über SOURCE (S)
                        if use_sublayer == "cross_attn" and isinstance(getattr(attn_cache, "Vproj", None), torch.Tensor):
                            Bv, Hv, Sv, Dh = attn_cache.Vproj.shape
                            x_src = attn_cache.Vproj.permute(0, 2, 1, 3).reshape(Bv, Sv, Hv * Dh).detach()  # (B,S,C)
                            x_abs = x_src.abs() + 1e-12
                            ch_w = x_abs / (x_abs.sum(dim=-1, keepdim=True) + 1e-12)  # (B,S,C)
                            R_fx_to_x = Rs * ch_w  # (B,S,C)
                            # Skip-Anteil auf Quelle nullen, damit Summation gelingt
                            rx = torch.zeros_like(R_fx_to_x)
                            used_value_path = True
                        else:
                            # Kanalverteilung pro Source-Token mit j-spezifischer Kopplung:
                            # M_k(j) = sum_{h,d} |W_V[h,d,k] * W_O[h,d,j]|
                            if isinstance(W_V, torch.Tensor) and isinstance(W_O, torch.Tensor):
                                WO_j = W_O[:, :, c_sel]  # (H,Dh)
                                M = (W_V.abs() * WO_j.abs().unsqueeze(-1)).sum(dim=(0, 1)).clamp_min(1e-12)  # (C_in,)
                                base = (x_prev.abs() * M.view(1, 1, -1))  # (B,S,C_in) bei self-attn S==T
                                ch_w = base / (base.sum(dim=-1, keepdim=True) + 1e-12)
                            elif isinstance(W_V, torch.Tensor):
                                # Fallback: nur W_V bekannt
                                Wv_sum = W_V.abs().sum(dim=(0, 1)).clamp_min(1e-12)  # (C,)
                                base = (x_prev.abs() * Wv_sum.view(1, 1, -1))
                                ch_w = base / (base.sum(dim=-1, keepdim=True) + 1e-12)
                            else:
                                ch_w = _channel_weights_absx(x_prev)
                            R_fx_to_x = Rs * ch_w  # (B,S,C)
                            used_value_path = True
            except Exception:
                used_value_path = False

        if not used_value_path:
            # Fallback: unterscheide FFN- und Nicht-FFN-Fälle
            if use_sublayer == "ffn":
                # FFN ist token-lokal: Relevanz nur auf Token t verteilen
                token_w = torch.zeros((B, T, 1), device=x_prev.device, dtype=x_prev.dtype)
                token_w[:, t_sel, 0] = 1.0
                # Kanalgewichte über W2∘W1 koppeln (j-spezifisch), sonst |x|-Fallback
                ch_w = None
                if attn_cache is not None:
                    try:
                        W1 = getattr(attn_cache, "W_FFN1", None)  # (Dhid, C_in)
                        W2 = getattr(attn_cache, "W_FFN2", None)  # (C_out, Dhid)
                        if index_axis == "channel" and isinstance(W1, torch.Tensor) and isinstance(W2, torch.Tensor) and 0 <= c_sel < W2.shape[0]:
                            w2_c = W2[c_sel].abs()                    # (Dhid,)
                            M = (w2_c.unsqueeze(1) * W1.abs()).sum(dim=0).clamp_min(1e-12)  # (C_in,)
                            base = (x_prev.abs() * M.view(1, 1, -1))
                            ch_w = base / (base.sum(dim=-1, keepdim=True) + 1e-12)
                    except Exception:
                        ch_w = None
                if ch_w is None:
                    ch_w = _channel_weights_absx(x_prev)
                R_fx_to_x = rFx_scalar * token_w * ch_w
            else:
                # Nicht-FFN: Token-Gewichte proportional ||x||^2 (Decoder) oder gleichmäßig (Encoder)
                if which_module == "encoder":
                    token_w = torch.full((B, T, 1), 1.0 / float(T), device=x_prev.device, dtype=x_prev.dtype)
                else:
                    token_energy = (x_prev.pow(2)).sum(dim=-1, keepdim=True)  # (B,T,1)
                    token_w = token_energy / (token_energy.sum(dim=1, keepdim=True) + 1e-12)  # (B,T,1)
                # Kanalgewichte aus W_V/W_O wenn verfügbar, sonst |x|
                ch_w = None
                if attn_cache is not None:
                    try:
                        W_V = getattr(attn_cache, "W_V", None)
                        W_O = getattr(attn_cache, "W_O", None)
                        if index_axis == "channel" and isinstance(W_V, torch.Tensor) and isinstance(W_O, torch.Tensor):
                            if 0 <= c_sel < W_O.shape[2]:
                                WO_j = W_O[:, :, c_sel]              # (H,Dh)
                                M = (W_V.abs() * WO_j.abs().unsqueeze(-1)).sum(dim=(0, 1))  # (C_in,)
                                M = M.clamp_min(1e-12)
                                base = (x_prev.abs() * M.view(1, 1, -1))  # (B,T,C_in)
                                ch_w = base / (base.sum(dim=-1, keepdim=True) + 1e-12)
                        elif isinstance(W_V, torch.Tensor):
                            Wv_sum = W_V.abs().sum(dim=(0, 1))  # (C,)
                            base = (x_prev.abs() * Wv_sum.view(1, 1, -1))
                            ch_w = base / (base.sum(dim=-1, keepdim=True) + 1e-12)
                    except Exception:
                        ch_w = None
                if ch_w is None:
                    ch_w = _channel_weights_absx(x_prev)  # (B,T,C)
                R_fx_to_x = rFx_scalar * token_w * ch_w

        # Gesamtrelevanz ergibt sich aus Skip-Anteil + verteilter Transform-Anteil
        if rx.shape == R_fx_to_x.shape:
            R_prev = rx + R_fx_to_x
        elif torch.allclose(rx.abs().sum(), torch.zeros((), device=rx.device, dtype=rx.dtype)):
            R_prev = R_fx_to_x
        else:
            R_prev = rx + R_fx_to_x

        # 4) Normierung
        if norm == "sum1":
            s = (R_prev.sum() + 1e-12)
            R_prev = R_prev / s
        elif norm == "sumAbs1":
            s = (R_prev.abs().sum() + 1e-12)
            R_prev = R_prev / s
        elif norm == "none":
            pass
        else:
            raise ValueError("target_norm muss 'sum1', 'sumAbs1' oder 'none' sein")

        # g_spec zur Nachvollziehbarkeit
        if index_axis == "channel":
            gspec = f"y[t={t_sel}, c={c_sel}]"
        else:
            gspec = f"y[token={t_sel}, channels=*]"
        return LRPResult(R_prev=R_prev.detach(), g_spec=gspec)


__all__ = ["LRPEngine", "LRPResult"]
