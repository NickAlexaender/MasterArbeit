"""
LRPEngine: Orchestriert den lokalen Relevanzfluss von Layer L nach L-1.

Die Engine ist so gebaut, dass sie keine CLI-Signaturen ändert und mit Hooks
(cps: CutPoints) aus hooks_maskdino.py arbeitet. Für den ersten Integrationsschritt
wird der SubLayer-Pfad als Identität behandelt; die Residual-Aufteilung ist implementiert.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import logging

import torch
from torch import Tensor

from .rules import residual_split, value_path_split, value_path_split_deform, layernorm_backshare
from .config import RESIDUAL_SPLIT, TARGET_TOKEN_IDX, LN_RULE, ATTN_QK_SHARE

# Für robuste Tensorform-Konvertierung (B,T,C) – zentralisiert
from myThesis.lrp.calc.tensor_utils import _to_BTC, build_target_relevance


@dataclass
class LRPResult:
    R_prev: Tensor       # Relevanz auf x^{L-1} (B,T,C), ggf. normalisiert gemäß 'norm'
    g_spec: str          # Dokumentation der Skalarisierung
    # Optional: Rohkomponenten zur Diagnose/rekonstruierten Kombinationen
    R_prev_raw: Optional[Tensor] = None     # unnormalisierte Summe (rx + R_fx)
    R_transform_raw: Optional[Tensor] = None  # unnormalisierte reine Transform-Komponente
    Rx_skip_raw: Optional[Tensor] = None      # unnormalisierte reine Skip-Komponente (Identität auf x)


class LRPEngine:
    def __init__(self, epsilon: float = 1e-6):
        self.eps = float(epsilon)
        self._logger = logging.getLogger("lrp")

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
        value_path_abs: bool | None = None,     # None ⇒ sign-preserving z-rule, True ⇒ |.|, False ⇒ quad
    ) -> LRPResult:
        """Berechne lokale Attribution von L nach L-1.

    Aktuelle Implementierung: Echte LRP-Zerlegung über Residual und SubLayer.
    Attention- und Deformable-Attention werden über den Value-Pfad auf Quelle
    zurückgeführt; FFN wird über die projektiven Gewichte rückverteilt.
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

        def _add_signed_eps(t: Tensor) -> Tensor:
            sign_or_one = torch.where(t >= 0, torch.ones_like(t), -torch.ones_like(t))
            return t + self.eps * sign_or_one

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

        # 1b) LayerNorm-Rückverteilung NICHT auf die Gesamt-Relevanz Ro anwenden,
        #     sondern erst nach der Residualaufteilung ausschließlich auf den Transform-Anteil.

        # 2) Residual-Aufteilung y = x + F(x)
        #    Für Attention- und FFN-SubLayer attribuieren wir bevorzugt über den Transform-Pfad,
        #    um Self-/Diagonal-Dominanz durch den Skip zu vermeiden. Falls conservative_residual=True
        #    und wirklich post_res gehookt wurde, kann anteilig über den Skip gesplittet werden.
        only_transform = use_sublayer in ("self_attn", "cross_attn", "ffn")
        # Falls wir offensichtlich auf einem Sub-Layer hooken (Hinweis durch attn_cache)
        # UND der Messpunkt nicht post_res ist, ist conservative_residual konzeptuell falsch → deaktivieren.
        if conservative_residual and attn_cache is not None and measurement_point != "post_res":
            try:
                if (getattr(attn_cache, "attn_weights", None) is not None) or (getattr(attn_cache, "W_FFN2", None) is not None):
                    conservative_residual = False
            except Exception:
                pass
        # Bereite direkte Aufteilung ohne Whitening vor
        Fx = (y_curr - x_prev)
        rx_full, rFx_full = residual_split(x_prev, Fx, Ro, mode=RESIDUAL_SPLIT, eps=self.eps)
        # Sicherheitsnetz: Falls Residual-Split numerisch stark von Ro abweicht, setze Transform=Ro, Skip=0
        try:
            sum_Ro0 = Ro.double().sum()
            sum_rx0 = rx_full.double().sum()
            sum_rFx0 = rFx_full.double().sum()
            delta0 = float((sum_rx0 + sum_rFx0 - sum_Ro0).item())
            tol0_base = float(abs(sum_Ro0).item())
            tol0 = 1e-3 * max(tol0_base, 1.0)
            if not (abs(delta0) <= tol0):
                # Korrigiere: Verwende nur Transform-Pfad und halte Summe identisch zu Ro
                rx_full = torch.zeros_like(Ro)
                rFx_full = Ro.clone()
                try:
                    self._logger.warning(
                        f"LRP residual_split drift={delta0:.3e} -> forcing rFx=Ro, rx=0 (safety)"
                    )
                except Exception:
                    pass
        except Exception:
            pass

        # LayerNorm-Backshare: nur auf den Transform-Pfad anwenden, damit der Skip-Pfad
        # identitätsgetreu auf denselben Kanal/Token zurückführt.
        if (
            attn_cache is not None
            and getattr(attn_cache, "ln_x_in", None) is not None
            and getattr(attn_cache, "ln_gamma", None) is not None
            and measurement_point == "post_res"
        ):
            try:
                ln_x = getattr(attn_cache, "ln_x_in")
                try:
                    from .config import LN_RULE as _LN_RULE_CFG
                except Exception:
                    _LN_RULE_CFG = LN_RULE
                if _LN_RULE_CFG == "abs-grad-xmu" and getattr(attn_cache, "ln_x_in_raw", None) is not None:
                    ln_x = getattr(attn_cache, "ln_x_in_raw")
                ln_g = getattr(attn_cache, "ln_gamma")
                ln_b = getattr(attn_cache, "ln_beta", None)
                beta = ln_b if isinstance(ln_b, torch.Tensor) else torch.zeros_like(ln_g)
                rFx_full = layernorm_backshare(ln_x, ln_g, beta, rFx_full, rule=LN_RULE, eps=self.eps)
            except Exception:
                pass
        if only_transform and not conservative_residual:
            # Nur Transform-Pfad verwenden: effektiver Skip 0.
            # WICHTIG: Erhalte die Form (Token-/Kanalverteilung) aus rFx_full und
            # skaliere lediglich auf die Gesamtmasse von Ro. Falls die Summe von rFx_full
            # numerisch problematisch ist (NaN/Inf oder nahezu 0), falle robust auf rFx=Ro zurück.
            rx = torch.zeros_like(Ro)
            s_rFx = rFx_full.sum(dim=(1, 2), keepdim=True)
            s_Ro = Ro.sum(dim=(1, 2), keepdim=True)
            # Robustheits-Checks
            bad = (~torch.isfinite(s_rFx)) | (s_rFx.abs() < 1e-12)
            if bad.any():
                # Fallback: ignoriere Form von rFx_full und verwende Ro direkt als Transform-Masse
                rFx = Ro
            else:
                rFx = rFx_full * (s_Ro / s_rFx)
        else:
            # Effektiver Split gemäß Regel
            rx = rx_full
            rFx = rFx_full

        # Erzwinge Batch-weise Relevanzkonservierung vor der Rückverteilung:
        # Skaliere rx und rFx so, dass sum_b,sum_t,sum_c(rx+rFx) exakt sum(Ro) pro Batch entspricht.
        try:
            s_rxFx = (rx + rFx).sum(dim=(1, 2), keepdim=True)  # (B,1,1)
            s_Ro_b = Ro.sum(dim=(1, 2), keepdim=True)          # (B,1,1)
            bad_mask = (~torch.isfinite(s_rxFx)) | (s_rxFx.abs() < 1e-12)
            scale_b = torch.where(bad_mask, torch.ones_like(s_rxFx), s_Ro_b / s_rxFx)
            rx = rx * scale_b
            rFx = rFx * scale_b
        except Exception:
            pass

        # 3) Verteile die Transform-Pfad-Relevanz rFx zurück auf x_prev.
        #    Bevorzugt über Attention-Gewichte (Value-Pfad Approximation),
        #    Fallback: robuste token-/kanalweise Gewichte basierend auf Energie/|x|.
        # Gesamt-Relevanzanteil des Transform-Pfads als Skalar pro Batch
        rFx_scalar = rFx.sum(dim=(1, 2), keepdim=True)  # (B,1,1)
        Ro_scalar = Ro.sum(dim=(1, 2), keepdim=True)    # (B,1,1)

        def _channel_weights_absx(xbtc: Tensor) -> Tensor:
            ch_abs_ = xbtc.abs() + 1e-12
            return ch_abs_ / (ch_abs_.sum(dim=-1, keepdim=True) + 1e-12)

        R_fx_to_x: Tensor
        used_value_path = False
        path_label = "fallback"
        if attn_cache is not None:
            try:
                # 3a) Bevorzugt: Deformable-Attention Pfad, falls vorhanden
                dsl = getattr(attn_cache, "deform_sampling_locations", None)
                daw = getattr(attn_cache, "deform_attention_weights", None)
                dshapes = getattr(attn_cache, "deform_spatial_shapes", None)
                dstart = getattr(attn_cache, "deform_level_start_index", None)
                if isinstance(dsl, torch.Tensor) and isinstance(daw, torch.Tensor) and isinstance(dshapes, torch.Tensor) and isinstance(dstart, torch.Tensor):
                    # Relevanz je Ziel-Token: bevorzuge die per-Token-Verteilung aus rFx,
                    # fallback auf Single-Token t_sel falls T nicht passt.
                    if dsl.shape[1] == y_curr.shape[1]:
                        Ro_j = rFx.sum(dim=2, keepdim=True)  # (B,T,1)
                    else:
                        Ro_j = torch.zeros((B, dsl.shape[1], 1), device=y_curr.device, dtype=y_curr.dtype)
                        t_for_deform = min(t_sel, dsl.shape[1] - 1)
                        Ro_j[:, t_for_deform, 0] = rFx.sum(dim=(1, 2), keepdim=True).view(B)
                    Rs = value_path_split_deform(Ro_j, dsl, daw, dshapes, dstart)  # (B,S,1)
                    # Kanal-/Tokenverteilung:
                    if index_axis == "token":
                        # Token-Achse: Vorzugsweise 1:1-Mapping S->T, sonst Fallback auf t_sel
                        if Rs.shape[1] == T:
                            # Kanalverteilung bevorzugt über erfasste Projektionsgewichte
                            WVd = getattr(attn_cache, "W_V_deform", None)
                            WOd = getattr(attn_cache, "W_O_deform", None)
                            if isinstance(WVd, torch.Tensor) and isinstance(WOd, torch.Tensor) and 0 <= t_sel < T:
                                # Für Token-Ziel: verteile Kanäle global, unabhängig vom Token
                                M = (WVd.abs() * WOd.abs().sum(dim=2, keepdim=True)).sum(dim=(0,1)).view(1,1,-1)  # (1,1,C)
                                z = x_prev * M
                                zsum = _add_signed_eps(z.sum(dim=-1, keepdim=True))
                                ch_w = z / zsum  # (B,T,C)
                            else:
                                ch_w = _channel_weights_absx(x_prev)  # (B,T,C)
                            R_fx_to_x = Rs.expand(-1, -1, C) * ch_w  # (B,T,C)
                        else:
                            token_w = torch.zeros((B, T, 1), device=x_prev.device, dtype=x_prev.dtype)
                            token_w[:, t_sel, 0] = 1.0
                            WVd = getattr(attn_cache, "W_V_deform", None)
                            WOd = getattr(attn_cache, "W_O_deform", None)
                            if isinstance(WVd, torch.Tensor) and isinstance(WOd, torch.Tensor):
                                M = (WVd.abs() * WOd.abs().sum(dim=2, keepdim=True)).sum(dim=(0,1)).view(1,1,-1)  # (1,1,C)
                                z = x_prev[:, t_sel:t_sel+1, :] * M
                                zsum = _add_signed_eps(z.sum(dim=-1, keepdim=True))
                                ch_w = z / zsum  # (B,1,C)
                            else:
                                ch_w = _channel_weights_absx(x_prev[:, t_sel:t_sel+1, :])  # (B,1,C)
                            r_total = Rs.sum(dim=1, keepdim=True)  # (B,1,1)
                            R_fx_to_x = r_total * token_w * ch_w
                    else:
                        # Kanal-Achse: Wenn S==T, nutze Rs tokenweise; sonst Fallback (gleichmäßig über T)
                        if Rs.shape[1] == T:
                            WVd = getattr(attn_cache, "W_V_deform", None)
                            WOd = getattr(attn_cache, "W_O_deform", None)
                            if isinstance(WVd, torch.Tensor) and isinstance(WOd, torch.Tensor) and 0 <= c_sel < WOd.shape[2]:
                                # Effektives Kanalmaß M_c = sum_{h,dh} |WVd[h,dh,c] * WOd[h,dh,c_sel]|
                                M = (WVd.abs() * WOd[:, :, c_sel:c_sel+1].abs()).sum(dim=(0,1)).view(1,1,-1)  # (1,1,C)
                                z = x_prev * M
                                zsum = _add_signed_eps(z.sum(dim=-1, keepdim=True))
                                ch_w = z / zsum  # (B,T,C)
                            else:
                                ch_w = _channel_weights_absx(x_prev)  # (B,T,C)
                            R_fx_to_x = Rs * ch_w  # (B,T,C)
                        else:
                            token_w = torch.full((B, T, 1), 1.0 / float(T), device=x_prev.device, dtype=x_prev.dtype)
                            WVd = getattr(attn_cache, "W_V_deform", None)
                            WOd = getattr(attn_cache, "W_O_deform", None)
                            if isinstance(WVd, torch.Tensor) and isinstance(WOd, torch.Tensor) and 0 <= c_sel < WOd.shape[2]:
                                M = (WVd.abs() * WOd[:, :, c_sel:c_sel+1].abs()).sum(dim=(0,1)).view(1,1,-1)  # (1,1,C)
                                z = x_prev * M
                                zsum = _add_signed_eps(z.sum(dim=-1, keepdim=True))
                                ch_w = z / zsum  # (B,T,C)
                            else:
                                ch_w = _channel_weights_absx(x_prev)  # (B,T,C)
                            r_total = Rs.sum(dim=1, keepdim=True)  # (B,1,1)
                            R_fx_to_x = r_total * token_w * ch_w
                    used_value_path = True
                    path_label = "deform"
                    try:
                        self._logger.debug("LRP: using deformable value-path (MSDeformAttn)")
                    except Exception:
                        pass
                if not used_value_path:
                    # 3b) Standard MHA-Pfad
                    aw = getattr(attn_cache, "attn_weights", None)
                    Vproj = getattr(attn_cache, "Vproj", None)
                    W_O = getattr(attn_cache, "W_O", None)
                    W_V = getattr(attn_cache, "W_V", None)
                    # Normalisiere Shapes: (B,H,T,S) falls möglich
                    if isinstance(aw, torch.Tensor):
                        if aw.dim() == 4:
                            aw4 = aw
                        elif aw.dim() == 3:
                            aw4 = aw.unsqueeze(1)
                        else:
                            aw4 = None
                    else:
                        aw4 = None
                    if index_axis == "channel" and isinstance(aw4, torch.Tensor) and isinstance(Vproj, torch.Tensor) and isinstance(W_O, torch.Tensor):
                        # Erwartete Shapes: aw (B,H,T,S), Vproj (B,H,S,Dh), W_O (H,Dh,C)
                        # Heads ggf. an Vproj anpassen
                        try:
                            if Vproj.dim() == 4 and aw4.dim() == 4 and aw4.shape[1] != Vproj.shape[1]:
                                aw4 = aw4.expand(aw4.shape[0], Vproj.shape[1], aw4.shape[2], aw4.shape[3]).contiguous()
                        except Exception:
                            pass
                        # Query-Domain Attribution (B,T,C): klassische LRP ohne Wechsel in S-Domäne
                        # Hilfsfunktion: Value-Pfad in Query-Domain
                        def _value_path_query_domain(Ro_j: Tensor, attn_w: Tensor, Vproj_: Tensor, W_O_j_: Tensor, W_V_: Tensor) -> Tensor:
                            # z = sum_s a[t,s,h]*Vproj[s,h,d] -> (B,T,H,Dh)
                            z = torch.einsum("bhts,bhsd->bthd", attn_w, Vproj_)
                            # z-Regel zurück von Out-Kanal j auf z-Dims
                            scores = z.unsqueeze(-1) * W_O_j_.unsqueeze(0).unsqueeze(0)  # (B,T,H,Dh,1)
                            denom = scores.sum(dim=(2,3), keepdim=True)
                            denom = denom + self.eps * torch.where(denom >= 0, torch.ones_like(denom), -torch.ones_like(denom))
                            alpha = scores / denom
                            Rz = alpha * Ro_j.view(B, aw4.shape[2], 1, 1, 1)
                            Rz = Rz.squeeze(-1)  # (B,T,H,Dh)
                            # Verteile Rz auf Kanäle C via |W_V|
                            if not isinstance(W_V_, torch.Tensor):
                                # Fallback: |x|-Gewichte
                                zloc = x_prev
                                zsum = _add_signed_eps(zloc.sum(dim=-1, keepdim=True))
                                return (Rz.sum(dim=(2,3), keepdim=False).unsqueeze(-1)) * (zloc / zsum)
                            abs_WV = W_V_.abs()  # (H,Dh,C)
                            denom_c = abs_WV.sum(dim=(1,2), keepdim=True).clamp_min(1e-12)
                            w_norm = abs_WV / denom_c  # (H,Dh,C)
                            Rbtc = torch.einsum("bthd,hdc->btc", Rz, w_norm)
                            return Rbtc

                        # Ro_j per Token bevorzugen, wenn Query-Länge passt; sonst Single-Token-Fallback
                        if aw4.shape[2] == rFx.shape[1]:
                            Ro_j = rFx.sum(dim=2, keepdim=True)  # (B,T,1)
                        else:
                            Ro_j = torch.zeros((B, aw4.shape[2], 1), device=y_curr.device, dtype=y_curr.dtype)
                            Ro_j[:, t_sel, 0] = rFx.sum(dim=(1, 2), keepdim=True).view(B)
                        W_O_j = W_O[:, :, c_sel:c_sel+1]
                        R_query = _value_path_query_domain(Ro_j, aw4, Vproj, W_O_j, W_V)
                        # Optionaler Q/K-Anteil ρ: als Query-Token-Anteil (domänenkonform)
                        try:
                            rho = float(ATTN_QK_SHARE)
                        except Exception:
                            rho = 0.0
                        if rho > 0.0 and 0 <= t_sel < T:
                            x_t = x_prev[:, t_sel, :].abs()
                            x_w = x_t / (x_t.sum(dim=-1, keepdim=True).clamp_min(1e-12))
                            R_qk = rFx_scalar.view(B, 1, 1) * x_w.unsqueeze(1)  # (B,1,C)
                            R_query = (1.0 - rho) * R_query
                            R_query[:, t_sel, :] = R_query[:, t_sel, :] + rho * R_qk.squeeze(1)
                        R_fx_to_x = R_query
                        used_value_path = True
                        path_label = "mha"
                        try:
                            self._logger.debug("LRP: using MHA value-path (query-domain)")
                        except Exception:
                            pass
                    elif index_axis == "token" and isinstance(aw4, torch.Tensor):
                        # Token-Ziel: Verteile Transform-Relevanz über Tokens via Value-Pfad,
                        # statt trivial nur auf t_sel zu setzen. Das nutzt attn_weights, Vproj, W_O.
                        # 1) Heads ggf. an Vproj anpassen
                        try:
                            if isinstance(Vproj, torch.Tensor) and aw4.dim() == 4 and Vproj.dim() == 4 and aw4.shape[1] != Vproj.shape[1]:
                                aw4 = aw4.expand(aw4.shape[0], Vproj.shape[1], aw4.shape[2], aw4.shape[3]).contiguous()
                        except Exception:
                            pass

                        # 2) Baue per-Token Relevanz Ro_j (B,T,1). Falls T nicht passt, fallback auf t_sel.
                        if aw4.shape[2] == rFx.shape[1]:
                            Ro_j = rFx.sum(dim=2, keepdim=True)  # (B,T,1)
                        else:
                            Ro_j = torch.zeros((B, aw4.shape[2], 1), device=y_curr.device, dtype=y_curr.dtype)
                            Ro_j[:, min(t_sel, aw4.shape[2]-1), 0] = rFx.sum(dim=(1, 2), keepdim=True).view(B)

                        # 3) Effektives W_O für Kanal-unschärfere Token-Zerlegung: Summe |W_O| über Kanäle
                        if isinstance(Vproj, torch.Tensor) and isinstance(W_O, torch.Tensor):
                            W_O_eff = W_O.abs().sum(dim=2, keepdim=True)  # (H,Dh,1)
                            try:
                                # Optional: Konfiguration, ob |.|-Gewichte, quadratisch oder sign-preserving
                                from .config import SIGN_PRESERVING as _SGN
                                use_abs = None if bool(_SGN) else True
                            except Exception:
                                use_abs = None
                            Rs = value_path_split(Ro_j, aw4, Vproj, W_O_eff, use_abs=use_abs, eps=self.eps)  # (B,S,1)
                            # 4) Mappe Rs auf (B,T,1): bei Self-Attn ist S==T, sonst fallback auf Token-Gewichte
                            if Rs.shape[1] == T:
                                token_w = Rs  # (B,T,1)
                            else:
                                # Gleichmäßig oder energiegewichtet auf Tokens verteilen (Decoder bevorzugt Energie)
                                if which_module == "encoder":
                                    token_w = torch.full((B, T, 1), 1.0 / float(T), device=x_prev.device, dtype=x_prev.dtype)
                                else:
                                    token_energy = (x_prev.pow(2)).sum(dim=-1, keepdim=True)
                                    token_w = token_energy / (token_energy.sum(dim=1, keepdim=True) + 1e-12)
                            # 5) Kanalverteilung pro Token via |x|
                            ch_w = _channel_weights_absx(x_prev)  # (B,T,C)
                            r_total = Ro_j.sum(dim=1, keepdim=True)  # (B,1,1)
                            # Skaliere Rs auf Gesamtmasse rFx_scalar, bewahre Token-Form
                            s_Rs = Rs.sum(dim=1, keepdim=True).clamp_min(1e-12)  # (B,1,1)
                            scale = rFx_scalar / s_Rs
                            token_mass = (Rs * scale)  # (B,S,1)
                            if token_mass.shape[1] == T:
                                token_w = token_mass  # (B,T,1)
                            R_fx_to_x = token_w * ch_w
                            used_value_path = True
                            path_label = "mha-token"
                        else:
                            # Fallback: wie zuvor – one-hot auf t_sel, Kanäle via |W_O| oder |x|
                            token_w = torch.zeros((B, T, 1), device=x_prev.device, dtype=x_prev.dtype)
                            token_w[:, t_sel, 0] = 1.0
                            if isinstance(W_O, torch.Tensor):
                                w_c = W_O.abs().sum(dim=(0,1))  # (C,)
                                w_c = w_c / w_c.sum().clamp_min(1e-12)
                                ch_w = w_c.view(1, 1, -1).expand(B, 1, -1)
                            else:
                                ch_w = _channel_weights_absx(x_prev[:, t_sel:t_sel+1, :])
                            R_fx_to_x = rFx_scalar * token_w * ch_w
                            used_value_path = True
                            path_label = "mha-fallback"
            except Exception:
                used_value_path = False

        if not used_value_path:
            # Fallback: unterscheide FFN- und Nicht-FFN-Fälle
            if use_sublayer == "ffn":
                # FFN ist token-lokal: Relevanz nur auf Token t verteilen
                token_w = torch.zeros((B, T, 1), device=x_prev.device, dtype=x_prev.dtype)
                token_w[:, t_sel, 0] = 1.0
                # Kanalgewichte über W2∘W1 koppeln (j-spezifisch) mit z-Regel, sonst signed x-Fallback
                ch_w = None
                if attn_cache is not None:
                    try:
                        W1 = getattr(attn_cache, "W_FFN1", None)  # (Dhid, C_in)
                        W2 = getattr(attn_cache, "W_FFN2", None)  # (C_out, Dhid)
                        if index_axis == "channel" and isinstance(W1, torch.Tensor) and isinstance(W2, torch.Tensor) and 0 <= c_sel < W2.shape[0]:
                            w2_c = W2[c_sel]                    # (Dhid,)
                            M = (w2_c.unsqueeze(1) * W1).sum(dim=0)  # (C_in,) signed
                            z = x_prev * M.view(1, 1, -1)
                            zsum = _add_signed_eps(z.sum(dim=-1, keepdim=True))
                            ch_w = z / zsum
                    except Exception:
                        ch_w = None
                if ch_w is None:
                    # Stabiler Fallback: Kanalgewichte über |x| statt signed Summen
                    ch_w = _channel_weights_absx(x_prev)
                R_fx_to_x = rFx_scalar * token_w * ch_w
                path_label = "ffn"
            else:
                # Nicht-FFN: Token-Gewichte
                if index_axis == "token":
                    # Relevanz nur auf den gewählten Token t_sel verteilen
                    token_w = torch.zeros((B, T, 1), device=x_prev.device, dtype=x_prev.dtype)
                    token_w[:, t_sel, 0] = 1.0
                else:
                    # Proportional ||x||^2 (Decoder) oder gleichmäßig (Encoder)
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
                                M = (W_V * WO_j.unsqueeze(-1)).sum(dim=(0, 1))  # (C_in,) signed
                                z = x_prev * M.view(1, 1, -1)
                                zsum = _add_signed_eps(z.sum(dim=-1, keepdim=True))
                                ch_w = z / zsum
                        elif isinstance(W_V, torch.Tensor):
                            Wv_sum = W_V.sum(dim=(0, 1))  # (C,) signed
                            z = x_prev * Wv_sum.view(1, 1, -1)
                            zsum = _add_signed_eps(z.sum(dim=-1, keepdim=True))
                            ch_w = z / zsum
                    except Exception:
                        ch_w = None
                if ch_w is None:
                    # Stabiler Fallback: Kanalgewichte über |x| statt signed Summen
                    ch_w = _channel_weights_absx(x_prev)  # (B,T,C)
                R_fx_to_x = rFx_scalar * token_w * ch_w
                path_label = "fallback"

        # Gesamtrelevanz ergibt sich aus Skip-Anteil + verteilter Transform-Anteil
        # Konservations-Logging vor Endnorm (in doppelter Präzision, um Auslöschung zu vermeiden)
        try:
            sum_Ro = Ro.double().sum()
            sum_rx = rx.double().sum()
            sum_rFx = rFx.double().sum()
            delta_res = float((sum_rx + sum_rFx - sum_Ro).item())
            # relative Toleranz bezogen auf |sumRo| oder 1.0, damit bei sehr kleinen Summen nicht über-empfindlich
            tol_base = float(sum_Ro.abs().item())
            tol = 1e-3 * max(tol_base, 1.0)
            if not (abs(delta_res) <= tol):
                # Herabstufen auf DEBUG: direkt danach erzwingen wir Batch-Konservierung;
                # diese Meldung ist rein diagnostisch und soll nicht als Warnung erscheinen.
                self._logger.debug(
                    f"LRP conservation drift before backshare: Δres={delta_res:.3e} (sumRo={float(sum_Ro.item()):.3e}) path={path_label}"
                )
        except Exception:
            delta_res = float("nan")
    # Rohkomponenten für Diagnose/Kombinationen rekonstruieren
        # Transform-Pattern (skalierungsinvariant): vermeide Division durch 0
        s_eff = rFx_scalar.clamp_min(1e-12)
        R_fx_pattern = R_fx_to_x / s_eff  # (B,T,C) skaliert auf 1
        R_transform_raw = R_fx_pattern * Ro_scalar  # so als ob immer Ro-Skalar verwendet

        # Skip-Komponente identitätsgemäß: genau rx_full (nicht der evtl. genullte rx)
        Rx_skip_raw = rx_full

        # Effektive Summe gemäß Modus
        if rx.shape == R_fx_to_x.shape:
            R_prev_raw = rx + R_fx_to_x
        elif torch.allclose(rx.abs().sum(), torch.zeros((), device=rx.device, dtype=rx.dtype)):
            R_prev_raw = R_fx_to_x
        else:
            R_prev_raw = rx + R_fx_to_x

        # Erzwinge Konservierung auch nach Backshare pro Batch
        try:
            s_prev_b = R_prev_raw.sum(dim=(1, 2), keepdim=True)  # (B,1,1)
            s_Ro_b = Ro.sum(dim=(1, 2), keepdim=True)            # (B,1,1)
            bad_mask2 = (~torch.isfinite(s_prev_b)) | (s_prev_b.abs() < 1e-12)
            scale_b2 = torch.where(bad_mask2, torch.ones_like(s_prev_b), s_Ro_b / s_prev_b)
            R_prev_raw = R_prev_raw * scale_b2
        except Exception:
            pass

        # Diagnose nach angleichung (sollte ~0 sein)
        try:
            sum_prev = R_prev_raw.double().sum()
            delta_prev = float((sum_prev - sum_Ro).item())
            self._logger.debug(f"LRP: path={path_label}, index_axis={index_axis}, conservative={conservative_residual}, Δres={delta_res:.3e}, Δprev={delta_prev:.3e}")
        except Exception:
            pass

        # 4) Normierung
        R_prev = R_prev_raw
        if norm == "sum1":
            s = (R_prev.sum() + 1e-12)
            R_prev = R_prev / s
        elif norm == "sumAbs1":
            s = (R_prev.abs().sum() + 1e-12)
            R_prev = R_prev / s
        elif norm == "none":
            pass
        else:
            raise ValueError("norm muss 'sum1', 'sumAbs1' oder 'none' sein")

        # g_spec zur Nachvollziehbarkeit
        if index_axis == "channel":
            gspec = f"y[t={t_sel}, c={c_sel}]"
        else:
            gspec = f"y[token={t_sel}, channels=*]"
        return LRPResult(
            R_prev=R_prev.detach(),
            g_spec=gspec,
            R_prev_raw=R_prev_raw.detach(),
            R_transform_raw=R_transform_raw.detach(),
            Rx_skip_raw=Rx_skip_raw.detach(),
        )


__all__ = ["LRPEngine", "LRPResult"]
