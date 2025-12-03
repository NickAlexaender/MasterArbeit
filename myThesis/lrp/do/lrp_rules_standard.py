"""
LRP-Regeln für Residual, LayerNorm und Attention-Value-Pfad.

Dieses Modul etabliert die "Grammatik" von LRP, implementiert als:
    R_in = Σ R_out · (Beitrag des Neurons)

Implementierte Regeln:
    - ε-Regel: R_j = Σ_k (a_j · w_jk) / (Σ_j a_j · w_jk + ε) · R_k
    - γ-Regel: R_j = Σ_k (a_j · (w_jk + γ·w_jk⁺)) / (Σ_j a_j · (w_jk + γ·w_jk⁺) + ε) · R_k
    - LayerNorm: Korrekte Behandlung der Normalisierungsstatistiken (μ, σ²)

Hinweis: Diese Implementierungen sind robuste, numerisch stabile Varianten
und dienen als Bausteine für die LRPEngine.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .tensor_ops import safe_divide


# =============================================================================
# Numerische Stabilität (verwendet safe_divide aus tensor_ops)
# =============================================================================


# =============================================================================
# Standard LRP-Regeln (ε-Regel und γ-Regel)
# =============================================================================


def lrp_epsilon_rule(
    a_in: Tensor,
    weight: Tensor,
    R_out: Tensor,
    bias: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """Standard ε-Regel für lineare Layer (Linear, Conv2d).

    Implementiert die generische LRP-Regel:
        R_j = Σ_k (a_j · w_jk) / (Σ_j a_j · w_jk + ε) · R_k

    Die ε-Regel ist numerisch stabil und verteilt Relevanz proportional
    zum Beitrag jeder Eingabe zur Ausgabe.

    Args:
        a_in: Eingabe-Aktivierungen (B, ..., in_features).
        weight: Gewichtsmatrix (out_features, in_features).
        R_out: Ausgabe-Relevanz (B, ..., out_features).
        bias: Optionaler Bias-Vektor (wird zum Nenner addiert).
        eps: Stabilisierungsterm für numerische Sicherheit.

    Returns:
        R_in: Eingabe-Relevanz (B, ..., in_features).

    Mathematik:
        z_k = Σ_j a_j · w_jk + b_k    (Preaktivierung)
        R_j = Σ_k (a_j · w_jk / (z_k + ε·sign(z_k))) · R_k
    """
    # z = a @ W^T + b  (Preaktivierung)
    z = torch.einsum("...i,oi->...o", a_in, weight)
    if bias is not None:
        z = z + bias

    # Gewichtete Relevanz: (a_j * w_jk) / z_k * R_k
    # Für jeden Eingabekanal j und Ausgabekanal k
    s = safe_divide(R_out, z, eps)  # (..., out_features)

    # Rückpropagation: R_j = a_j * Σ_k (w_jk * s_k)
    c = torch.einsum("...o,oi->...i", s, weight)  # (..., in_features)

    return a_in * c


def lrp_gamma_rule(
    a_in: Tensor,
    weight: Tensor,
    R_out: Tensor,
    bias: Optional[Tensor] = None,
    gamma: float = 0.25,
    eps: float = 1e-6,
) -> Tensor:
    """γ-Regel für lineare Layer mit verstärkten positiven Gewichten.

    Die γ-Regel erhöht den Einfluss positiver Beiträge:
        w' = w + γ · w⁺   wobei w⁺ = max(0, w)

    Dies führt zu schärferen, weniger verrauschten Attributionen,
    besonders nützlich in tieferen Schichten des Netzwerks.

    Args:
        a_in: Eingabe-Aktivierungen (B, ..., in_features).
        weight: Gewichtsmatrix (out_features, in_features).
        R_out: Ausgabe-Relevanz (B, ..., out_features).
        bias: Optionaler Bias-Vektor.
        gamma: Verstärkungsfaktor für positive Gewichte (Standard: 0.25).
        eps: Stabilisierungsterm.

    Returns:
        R_in: Eingabe-Relevanz (B, ..., in_features).

    Mathematik:
        w'_jk = w_jk + γ · max(0, w_jk)
        z'_k = Σ_j a_j · w'_jk + b_k
        R_j = Σ_k (a_j · w'_jk / (z'_k + ε)) · R_k
    """
    # Modifizierte Gewichte: w' = w + γ * w⁺
    w_pos = weight.clamp(min=0)
    w_modified = weight + gamma * w_pos

    # Modifizierter Bias (falls vorhanden)
    b_modified = None
    if bias is not None:
        b_pos = bias.clamp(min=0)
        b_modified = bias + gamma * b_pos

    # Anwendung der ε-Regel mit modifizierten Gewichten
    return lrp_epsilon_rule(a_in, w_modified, R_out, b_modified, eps)


def lrp_alpha_beta_rule(
    a_in: Tensor,
    weight: Tensor,
    R_out: Tensor,
    bias: Optional[Tensor] = None,
    alpha: float = 2.0,
    beta: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """α-β-Regel: Separate Behandlung positiver und negativer Beiträge.

    Verteilt Relevanz getrennt über positive und negative Aktivierungen:
        R_j = α · R_j⁺ - β · R_j⁻

    Typische Wahl: α=2, β=1 (LRP-α₂β₁) oder α=1, β=0 (LRP-z⁺).

    Args:
        a_in: Eingabe-Aktivierungen.
        weight: Gewichtsmatrix.
        R_out: Ausgabe-Relevanz.
        bias: Optionaler Bias.
        alpha: Gewichtung positiver Beiträge (Standard: 2.0).
        beta: Gewichtung negativer Beiträge (Standard: 1.0).
        eps: Stabilisierungsterm.

    Returns:
        R_in: Eingabe-Relevanz.

    Note:
        Für Konservativität sollte α - β = 1 gelten.
    """
    a_pos = a_in.clamp(min=0)
    a_neg = a_in.clamp(max=0)
    w_pos = weight.clamp(min=0)
    w_neg = weight.clamp(max=0)

    # Positive Beiträge: a⁺·w⁺ + a⁻·w⁻
    z_pos = torch.einsum("...i,oi->...o", a_pos, w_pos) + \
            torch.einsum("...i,oi->...o", a_neg, w_neg)
    if bias is not None:
        z_pos = z_pos + bias.clamp(min=0)

    # Negative Beiträge: a⁺·w⁻ + a⁻·w⁺
    z_neg = torch.einsum("...i,oi->...o", a_pos, w_neg) + \
            torch.einsum("...i,oi->...o", a_neg, w_pos)
    if bias is not None:
        z_neg = z_neg + bias.clamp(max=0)

    # Relevanz-Verteilung
    s_pos = safe_divide(R_out, z_pos, eps)
    s_neg = safe_divide(R_out, z_neg, eps)

    # Rückpropagation
    c_pos = torch.einsum("...o,oi->...i", s_pos, w_pos)
    c_neg = torch.einsum("...o,oi->...i", s_neg, w_neg)

    R_in_pos = a_pos * c_pos + a_neg * torch.einsum("...o,oi->...i", s_pos, w_neg)
    R_in_neg = a_pos * torch.einsum("...o,oi->...i", s_neg, w_neg) + a_neg * c_neg

    return alpha * R_in_pos - beta * R_in_neg


def lrp_linear(
    layer: nn.Linear,
    a_in: Tensor,
    R_out: Tensor,
    rule: str = "epsilon",
    eps: float = 1e-6,
    gamma: float = 0.25,
    alpha: float = 2.0,
    beta: float = 1.0,
) -> Tensor:
    """LRP für nn.Linear Layer mit wählbarer Regel.

    Args:
        layer: Das lineare Layer.
        a_in: Eingabe-Aktivierungen.
        R_out: Ausgabe-Relevanz.
        rule: "epsilon", "gamma", oder "alpha_beta".
        eps, gamma, alpha, beta: Regelparameter.

    Returns:
        R_in: Eingabe-Relevanz.
    """
    weight = layer.weight.detach()
    bias = layer.bias.detach() if layer.bias is not None else None

    if rule == "epsilon":
        return lrp_epsilon_rule(a_in, weight, R_out, bias, eps)
    elif rule == "gamma":
        return lrp_gamma_rule(a_in, weight, R_out, bias, gamma, eps)
    elif rule == "alpha_beta":
        return lrp_alpha_beta_rule(a_in, weight, R_out, bias, alpha, beta, eps)
    else:
        raise ValueError(f"Unbekannte LRP-Regel: {rule}")


# =============================================================================
# Residual Split Strategien
# =============================================================================


class ResidualSplitStrategy:
    """Basis-Klasse für Residual-Split-Strategien."""

    IDENTITY = "identity"
    ENERGY = "energy"
    EPSILON = "epsilon"
    DOTPOS = "dotpos"
    ZSIGN = "zsign"


def residual_split(
    x: Tensor,
    Fx: Tensor,
    Ry: Tensor,
    mode: str = "epsilon",
    eps: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """Teile Relevanz Ry zwischen Skip-Pfad x und Transformationspfad F(x) auf.

    Bei Residual-Verbindungen y = x + F(x) muss die Relevanz Ry auf beide
    Pfade verteilt werden. Diese Funktion implementiert verschiedene Strategien.

    Strategien:
        - identity: Pass-through (Ry wird komplett an x weitergegeben, F(x) = 0).
        - epsilon: ε-Regel basierte Verteilung proportional zu x/(x+F(x)).
        - energy: Proportional zu ||x||² vs. ||F(x)||² je Token.
        - dotpos: Proportional zu positiven Skalarprodukten mit y = x + F(x).
        - zsign: Vorzeichenbewahrende z-Regel auf Tokenebene.

    Args:
        x: Skip-Pfad Aktivierungen (B, T, C) oder (B, C).
        Fx: Transformationspfad F(x) Aktivierungen.
        Ry: Relevanz am Ausgang y = x + F(x).
        mode: Strategie ("identity", "epsilon", "energy", "dotpos", "zsign").
        eps: Stabilisierungsterm.

    Returns:
        (Rx, RFx): Relevanz-Tupel für Skip- und Transformationspfad.
    """
    # Vereinheitliche Form (B, T, C)
    squeeze_output = False
    if x.dim() == 2:
        x = x.unsqueeze(1)
        squeeze_output = True
    if Fx.dim() == 2:
        Fx = Fx.unsqueeze(1)
    if Ry.dim() == 2:
        Ry = Ry.unsqueeze(1)

    if mode == "identity":
        # Pass-through: Gesamte Relevanz geht an den Skip-Pfad
        rx = Ry
        rFx = torch.zeros_like(Ry)

    elif mode == "epsilon":
        # ε-Regel: Proportional zum Beitrag jedes Pfades zur Summe
        # y = x + F(x), also z_x = x, z_Fx = F(x)
        y = x + Fx

        # Relevanz-Verteilung basierend auf relativem Beitrag
        rx = safe_divide(x, y, eps) * Ry
        rFx = safe_divide(Fx, y, eps) * Ry

    elif mode == "energy":
        # Energie-basierte Verteilung
        ex = (x.pow(2)).sum(dim=-1, keepdim=True) + eps
        ef = (Fx.pow(2)).sum(dim=-1, keepdim=True) + eps
        ratio_x = ex / (ex + ef)
        rx = ratio_x * Ry
        rFx = Ry - rx

    elif mode == "dotpos":
        # Positive Skalarprodukt-basierte Verteilung
        def posdot(a: Tensor, b: Tensor) -> Tensor:
            return torch.clamp((a * b), min=0).sum(dim=-1, keepdim=True) + eps

        y = x + Fx
        px = posdot(x, y)
        pf = posdot(Fx, y)
        ratio_x = px / (px + pf)
        rx = ratio_x * Ry
        rFx = Ry - rx

    elif mode == "zsign":
        # Vorzeichenbewahrende z-Regel auf Tokenebene
        zx = x.sum(dim=-1, keepdim=True)
        zf = Fx.sum(dim=-1, keepdim=True)
        rx = safe_divide(zx, zx + zf, eps) * Ry
        rFx = Ry - rx

    else:
        raise ValueError(f"Unbekannte Residual-Split-Strategie: {mode}")

    if squeeze_output:
        rx = rx.squeeze(1)
        rFx = rFx.squeeze(1)

    return rx, rFx


# =============================================================================
# LayerNorm LRP-Regel (Korrekte Behandlung der Normalisierungsstatistiken)
# =============================================================================


def layernorm_lrp(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    R_out: Tensor,
    rule: str = "taylor",
    ln_eps: float = 1e-5,
    lrp_eps: float = 1e-6,
) -> Tensor:
    """LRP-Regel für LayerNorm mit korrekter Behandlung der Normalisierungsstatistiken.

    LayerNorm ist nicht-linear aufgrund der Division durch die Varianz:
        y = γ · (x - μ) / σ + β

    wobei μ = mean(x), σ = sqrt(var(x) + ε).

    Das naive Durchreichen der Relevanz ignoriert die Abhängigkeit von μ und σ²
    von allen Eingaben und führt zu verrauschten Attributionen.

    Implementierte Strategien:
        - taylor: Taylor-Expansion 1. Ordnung (Gradient × Input).
        - local_linear: Behandelt LN als lokal-linear mit festem μ, σ.
        - zsign: Vorzeichenbewahrende z-Regel auf den normierten Beiträgen.
        - conservative: Konservative Regel mit vollständiger Jacobi-Berechnung.

    Args:
        x: Eingabe vor LayerNorm (B, T, C) oder (B, C).
        gamma: Skalierungsparameter γ (C,).
        beta: Verschiebungsparameter β (C,).
        R_out: Ausgabe-Relevanz (B, T, C) oder (B, C).
        rule: LRP-Regel ("taylor", "local_linear", "zsign", "conservative").
        ln_eps: Epsilon für LayerNorm (numerische Stabilität in σ).
        lrp_eps: Epsilon für LRP-Regel.

    Returns:
        R_in: Eingabe-Relevanz mit gleicher Form wie x.

    Mathematik (Taylor-Expansion):
        Die Taylor-Expansion 1. Ordnung um den Arbeitspunkt x₀ ergibt:
        R_j = x_j · ∂y/∂x_j · (R_out_j / y_j)

        Für LayerNorm: ∂y_j/∂x_i = γ_j/σ · (δ_ij - 1/C - (x_j-μ)(x_i-μ)/(Cσ²))
    """
    # Vereinheitliche Form (B, T, C)
    squeeze_output = False
    if x.dim() == 2:
        x = x.unsqueeze(1)
        squeeze_output = True
    if R_out.dim() == 2:
        R_out = R_out.unsqueeze(1)

    B, T, C = x.shape
    gamma = gamma.view(1, 1, C)
    beta = beta.view(1, 1, C) if beta is not None else 0

    # LayerNorm-Statistiken berechnen
    mu = x.mean(dim=-1, keepdim=True)  # (B, T, 1)
    x_centered = x - mu  # (B, T, C)
    var = x_centered.pow(2).mean(dim=-1, keepdim=True)  # (B, T, 1)
    sigma = (var + ln_eps).sqrt()  # (B, T, 1)

    # Normalisierte Eingabe
    x_norm = x_centered / sigma  # (B, T, C)

    # LayerNorm-Ausgabe (für Verifikation)
    y = gamma * x_norm + beta

    if rule == "taylor":
        # Taylor-Expansion 1. Ordnung (Gradient × Input Methode)
        # Berechne den exakten Gradienten ∂y/∂x via Autograd
        x_req = x.detach().requires_grad_(True)
        mu_loc = x_req.mean(dim=-1, keepdim=True)
        x_c_loc = x_req - mu_loc
        var_loc = x_c_loc.pow(2).mean(dim=-1, keepdim=True)
        sigma_loc = (var_loc + ln_eps).sqrt()
        y_loc = gamma * (x_c_loc / sigma_loc) + beta

        # Relevanz-gewichteter Skalar für Gradientenberechnung
        # Wir verwenden y · R_out als Proxy für die "wichtigen" Ausgaben
        target = (y_loc * R_out).sum()

        grad = torch.autograd.grad(
            outputs=target,
            inputs=x_req,
            retain_graph=False,
            create_graph=False,
        )[0]

        # Gradient × Input Relevanz
        R_in = x.detach() * grad

        # Skalierung zur Relevanz-Erhaltung (optional, aber empfohlen)
        R_sum_out = R_out.sum(dim=-1, keepdim=True)
        R_sum_in = R_in.sum(dim=-1, keepdim=True)
        R_in = R_in * safe_divide(R_sum_out, R_sum_in, lrp_eps)

    elif rule == "local_linear":
        # Behandle LayerNorm als lokal-linearen Layer mit festem μ, σ
        # y_j = (γ_j / σ) · x_j - (γ_j · μ / σ) + β_j
        # Effektives Gewicht: w_j = γ_j / σ (diagonal)
        # Effektiver Bias: b_j = -γ_j · μ / σ + β_j

        w_eff = gamma / sigma  # (B, T, C)

        # LRP ε-Regel: R_j = (w_j · x_j) / y · R_out
        R_in = safe_divide(w_eff * x, y, lrp_eps) * R_out

    elif rule == "zsign":
        # Vorzeichenbewahrende z-Regel auf den normierten Beiträgen
        # z_j = (γ_j / σ) · (x_j - μ)
        z = (gamma / sigma) * x_centered  # (B, T, C)

        # Gesamtrelevanz pro Token
        R_tok = R_out.sum(dim=-1, keepdim=True)

        # Verteilung proportional zu z (z-Regel über Kanäle je Token)
        R_in = safe_divide(z, z.sum(dim=-1, keepdim=True), lrp_eps) * R_tok

    elif rule == "conservative":
        # Konservative Regel: Vollständige Jacobi-Matrix-Berechnung
        # Dies ist rechenintensiv, aber exakt konservativ

        # Für Effizienz: Nutze die analytische Form der Jacobi-Matrix
        # ∂y_j/∂x_i = (γ_j/σ) · [δ_ij - 1/C - (x_j-μ)(x_i-μ)/(Cσ²)]

        inv_sigma = 1.0 / sigma  # (B, T, 1)
        inv_C = 1.0 / C

        # Term 1: Diagonale (δ_ij)
        term1 = gamma * inv_sigma  # (B, T, C)

        # Term 2: Konstante Subtraktion (-1/C)
        term2_coeff = -gamma * inv_sigma * inv_C  # (B, T, C)
        term2 = term2_coeff.sum(dim=-1, keepdim=True).expand_as(x)  # Broadcast

        # Term 3: Varianz-Korrektur (-(x_j-μ)(x_i-μ)/(Cσ²))
        # Für R_in_i = Σ_j ∂y_j/∂x_i · R_out_j
        # Term 3 Beitrag: -1/(Cσ²) · (x_i-μ) · Σ_j γ_j · (x_j-μ) · R_out_j
        weighted_sum = (gamma * x_centered * R_out).sum(dim=-1, keepdim=True)  # (B, T, 1)
        term3 = -inv_C * (inv_sigma.pow(2)) * x_centered * weighted_sum  # (B, T, C)

        # Gesamt-Gradient (Zeilen der Jacobi × R_out)
        # R_in_i = Σ_j J_ji · R_out_j
        # = Σ_j [term1_j·δ_ij + term2_coeff_j + term3_coeff_ij] · R_out_j
        grad_times_R = term1 * R_out + (term2_coeff * R_out).sum(dim=-1, keepdim=True) + term3

        # Gradient × Input für LRP
        R_in = x * grad_times_R

        # Relevanz-Normalisierung zur Konservativität
        R_sum_out = R_out.sum(dim=-1, keepdim=True)
        R_sum_in = R_in.sum(dim=-1, keepdim=True)
        R_in = R_in * safe_divide(R_sum_out, R_sum_in, lrp_eps)

    else:
        raise ValueError(f"Unbekannte LayerNorm-LRP-Regel: {rule}")

    if squeeze_output:
        R_in = R_in.squeeze(1)

    return R_in


def layernorm_backshare(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    Ry: Tensor,
    rule: str = "taylor",
    eps: float = 1e-6,
) -> Tensor:
    """Verteile Relevanz über LayerNorm-Eingang x (Legacy-Interface).

    Diese Funktion ist ein Wrapper um layernorm_lrp für Rückwärtskompatibilität.

    Unterstützte Regeln:
        - "taylor": Taylor-Expansion (empfohlen, neu).
        - "local_linear": Lokal-lineare Approximation (neu).
        - "zsign": Vorzeichenbewahrende z-Regel (aktualisiert).
        - "xmu": Proportional zu |x-μ| (Legacy, nicht empfohlen).
        - "abs-grad-xmu": Absolutgradient × |x-μ| (Legacy).

    Args:
        x: Eingabe vor LayerNorm.
        gamma: γ-Parameter.
        beta: β-Parameter.
        Ry: Ausgabe-Relevanz.
        rule: LRP-Regel.
        eps: Stabilisierungsterm.

    Returns:
        R_in: Eingabe-Relevanz.
    """
    # Neue Regeln direkt an layernorm_lrp weiterleiten
    if rule in ("taylor", "local_linear", "zsign", "conservative"):
        return layernorm_lrp(x, gamma, beta, Ry, rule=rule, lrp_eps=eps)

    # Legacy-Regeln
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if Ry.dim() == 2:
        Ry = Ry.unsqueeze(1)

    if rule == "xmu":
        # Legacy: Einfache Proportionalität zu |x-μ| (ignoriert γ, σ)
        xm = (x - x.mean(dim=-1, keepdim=True)).abs()
        w = xm / (xm.sum(dim=-1, keepdim=True) + 1e-12)
        return w * Ry

    elif rule == "abs-grad-xmu":
        # Legacy: Absolutgradient × |x-μ|
        x_req = x.detach().requires_grad_(True)
        xmu = x_req - x_req.mean(dim=-1, keepdim=True)
        sigma = xmu.pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        y = gamma * (xmu / sigma) + beta
        L = (y * Ry).sum()
        g = torch.autograd.grad(
            outputs=L, inputs=x_req,
            retain_graph=False, create_graph=False, allow_unused=True
        )[0]
        if g is None:
            g = torch.ones_like(x_req)
        g = g.abs()
        w = g * xmu.abs()
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-12)
        return w * Ry

    else:
        raise ValueError(f"Unbekannte LayerNorm-Regel: {rule}")


# =============================================================================
# Attention Value Path LRP
# =============================================================================


def value_path_split(
    Ro_j: Tensor,
    attn_weights: Tensor,
    Vproj: Tensor,
    W_O_j: Tensor,
    use_abs: bool | None = None,
    eps: float = 1e-6,
    ) -> Tensor:
    """Verteilt Relevanz von Ausgabetoken t auf Quell-Token s über den Value-Pfad.

    Parameter:
    - Ro_j: (B,T,1) Relevanz für Zielkanal j pro Ziel-Token t
    - attn_weights: (B,H,T,S)
    - Vproj: (B,H,S,Dh)  Value-Projektionen je Head
    - W_O_j: (H,Dh,1)    Spaltenvektor der Output-Projektion für Kanal j
    - use_abs: None ⇒ vorzeichenbewahrende z-Regel; True ⇒ |.|-Gewichte; False ⇒ quadratische Gewichte

    Rückgabe:
    - Rs: (B,S,1) Relevanz auf die Quell-Token s aggregiert über t
    """
    B, H, T, S = attn_weights.shape
    # Effektiver signed Beitrag Z_{t,s} = sum_h a_{t,s,h} * <V_{s,h}, W_O_j[h]>
    # score_h: (B,S,1); nach Broadcast -> (B,T,S,1)
    scores = []
    for h in range(H):
        vs = Vproj[:, h]          # (B,S,Dh)
        wo = W_O_j[h]             # (Dh,1)
        score = torch.einsum("bsd,dk->bsk", vs, wo)  # (B,S,1)
        score = score.unsqueeze(1).expand(-1, T, -1, -1)  # (B,T,S,1)
        a = attn_weights[:, h].unsqueeze(-1)            # (B,T,S,1)
        scores.append(a * score)                        # signed
    Z = torch.stack(scores, dim=1).sum(dim=1)           # (B,T,S,1)

    if use_abs is True:
        C = Z.abs()
        Csum = C.sum(dim=2, keepdim=True) + 1e-12
        w = C / Csum
        Rs = (w * Ro_j).sum(dim=1)
        return Rs
    elif use_abs is False:
        C = Z.pow(2)
        Csum = C.sum(dim=2, keepdim=True) + 1e-12
        w = C / Csum
        Rs = (w * Ro_j).sum(dim=1)
        return Rs
    else:
        # Vorzeichenbewahrende z-Regel über s je t
        w = safe_divide(Z, Z.sum(dim=2, keepdim=True), eps)  # (B,T,S,1)
        Rs = (w * Ro_j).sum(dim=1)                           # (B,S,1)
        return Rs


def value_path_split_deform(
    Ro_j: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
    spatial_shapes: Tensor,
    level_start_index: Tensor,
) -> Tensor:
    """Verteile Relevanz von Ziel-Tokens t zurück auf Quelle S der MSDeformAttn.

    Bilineares "Splatting" der attention_weights auf 4 Nachbarn je Sample-Punkt.

    Parameter:
    - Ro_j: (B,T,1) Relevanz auf Ziel-Tokens (z. B. aus rFx_scalar für gewählten Kanal)
    - sampling_locations: (B,T,H,L,P,2) normierte Koordinaten je Level in [0,1]
    - attention_weights: (B,T,H,L,P) zugehörige Gewichte (>=0, Summe über P typ. 1)
    - spatial_shapes: (L,2) je Level (H_l, W_l)
    - level_start_index: (L,) Start-Offsets pro Level in der flachen S-Dimension

    Rückgabe:
    - Rs: (B,S,1) Relevanz auf Quelle (über T/H/L/P aggregiert)
    """
    B, T, Hh, L, P, _ = sampling_locations.shape
    device = sampling_locations.device
    dtype = sampling_locations.dtype

    # Gesamt-Tokenzahl S bestimmen
    spatial_shapes = spatial_shapes.to(torch.long)
    level_start_index = level_start_index.to(torch.long)
    last_h, last_w = spatial_shapes[-1]
    S_total = int((level_start_index[-1] + last_h * last_w).item())

    Rs = torch.zeros((B, S_total, 1), device=device, dtype=Ro_j.dtype)

    # Broadcast Relevanz über Köpfe/Punkte
    Ro_broadcast = Ro_j.view(B, T, 1, 1)  # (B,T,1,1)

    for l in range(L):
        Hl = int(spatial_shapes[l, 0].item())
        Wl = int(spatial_shapes[l, 1].item())
        base = int(level_start_index[l].item())

        loc = sampling_locations[:, :, :, l]  # (B,T,Hh,P,2)
        aw = attention_weights[:, :, :, l]    # (B,T,Hh,P)

        # Normierte -> Pixelkoordinaten (wie bei grid_sample Bilinear)
        x = loc[..., 0] * Wl - 0.5  # (B,T,Hh,P)
        y = loc[..., 1] * Hl - 0.5

        x0 = torch.floor(x).to(torch.long)
        y0 = torch.floor(y).to(torch.long)
        x1 = x0 + 1
        y1 = y0 + 1

        # Clamp in gültigen Bereich
        x0c = x0.clamp(0, Wl - 1)
        x1c = x1.clamp(0, Wl - 1)
        y0c = y0.clamp(0, Hl - 1)
        y1c = y1.clamp(0, Hl - 1)

        fx = (x - x0.to(x.dtype)).clamp(0, 1)
        fy = (y - y0.to(y.dtype)).clamp(0, 1)

        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy

        # Basisgewicht aus Attention * Relevanz
        base_w = (aw * Ro_broadcast).to(Rs.dtype)  # (B,T,Hh,P)

        def _scatter(nei_w: Tensor, xx: Tensor, yy: Tensor):
            idx = (yy * Wl + xx).view(B, -1) + base
            vals = (base_w * nei_w).view(B, -1)
            # Summe über T/Hh/P -> add in S-Dimension
            for b in range(B):
                Rs[b, idx[b], 0] += vals[b]

        _scatter(w00, x0c, y0c)
        _scatter(w10, x1c, y0c)
        _scatter(w01, x0c, y1c)
        _scatter(w11, x1c, y1c)

    return Rs


__all__ = [
    # Standard LRP-Regeln
    "lrp_epsilon_rule",
    "lrp_gamma_rule",
    "lrp_alpha_beta_rule",
    "lrp_linear",
    # Residual-Split
    "ResidualSplitStrategy",
    "residual_split",
    # LayerNorm LRP
    "layernorm_lrp",
    "layernorm_backshare",  # Legacy-Interface
    # Attention Value Path
    "value_path_split",
    "value_path_split_deform",
]
