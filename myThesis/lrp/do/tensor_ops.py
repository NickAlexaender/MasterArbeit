"""
Tensor utility functions used across the LRP analysis code.

Contains strictly stateless math helper functions:
- safe_divide: Handles division by zero for LRP stability.
- rearrange_activations: Converts between Detectron2's (N, C, H, W) and Transformer (N, L, C) formats.
- compute_jacobian: Helper to compute local Jacobians for Gradient-Times-Input (GTI) on unstable layers.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
import warnings
import torch
import torch.nn as nn
from torch import Tensor


# =============================================================================
# Safe Division for LRP Stability
# =============================================================================


def safe_divide(
    numerator: Tensor,
    denominator: Tensor,
    eps: float = 1e-9,
    signed_eps: bool = True,
) -> Tensor:
    """Sichere Division zur Vermeidung von NaN/Inf in LRP-Berechnungen.

    LRP-Regeln erfordern häufig Divisionen der Form a / z, wobei z die
    Summe gewichteter Eingaben ist. Bei z ≈ 0 entstehen numerische
    Instabilitäten. Diese Funktion stabilisiert durch Hinzufügen eines
    kleinen Epsilon-Wertes.

    Args:
        numerator: Zähler-Tensor.
        denominator: Nenner-Tensor (kann nahe Null sein).
        eps: Minimaler Absolutwert zur Stabilisierung.
        signed_eps: Falls True, wird das Vorzeichen des Nenners beibehalten
                    (standard LRP-ε-Regel). Falls False, wird |eps| addiert.

    Returns:
        Stabilisierter Quotient numerator / (denominator + ε).

    Example:
        >>> z = torch.tensor([1.0, 0.0, -0.5, 1e-15])
        >>> safe_divide(torch.ones(4), z)
        tensor([1.0000, 1e9, -2.0000, 1e9])  # stabil, kein Inf/NaN
    """
    if signed_eps:
        # Vorzeichen-erhaltende Stabilisierung (LRP-ε Standardregel)
        stabilizer = eps * torch.sign(denominator) + (denominator == 0).float() * eps
    else:
        stabilizer = eps
    return numerator / (denominator + stabilizer)


# =============================================================================
# Activation Rearrangement (Detectron2 <-> Transformer Format)
# =============================================================================


def rearrange_activations(
    tensor: Tensor,
    source_format: str,
    target_format: str,
    spatial_shape: Optional[Tuple[int, int]] = None,
) -> Tensor:
    """Konvertiert Aktivierungen zwischen Detectron2- und Transformer-Formaten.

    Unterstützte Formate:
    - "NCHW": Detectron2/CNN-Standard (Batch, Channels, Height, Width)
    - "NLC":  Transformer-Standard (Batch, Length/Tokens, Channels)
    - "NC":   Flaches Format (Batch, Channels) – nur für Ausgaben ohne Raum

    Args:
        tensor: Eingabe-Tensor.
        source_format: Quellformat ("NCHW", "NLC", oder "NC").
        target_format: Zielformat ("NCHW", "NLC", oder "NC").
        spatial_shape: (H, W) erforderlich bei NLC -> NCHW Konversion.

    Returns:
        Tensor im Zielformat.

    Raises:
        ValueError: Bei unbekannten Formaten oder fehlender spatial_shape.

    Example:
        >>> x = torch.randn(2, 256, 14, 14)  # NCHW
        >>> y = rearrange_activations(x, "NCHW", "NLC")
        >>> y.shape
        torch.Size([2, 196, 256])
    """
    source_format = source_format.upper()
    target_format = target_format.upper()

    if source_format == target_format:
        return tensor

    # NCHW -> NLC
    if source_format == "NCHW" and target_format == "NLC":
        if tensor.dim() != 4:
            raise ValueError(f"NCHW erwartet 4D-Tensor, erhalten: {tensor.dim()}D")
        N, C, H, W = tensor.shape
        # (N, C, H, W) -> (N, H, W, C) -> (N, H*W, C)
        return tensor.permute(0, 2, 3, 1).reshape(N, H * W, C)

    # NLC -> NCHW
    if source_format == "NLC" and target_format == "NCHW":
        if tensor.dim() != 3:
            raise ValueError(f"NLC erwartet 3D-Tensor, erhalten: {tensor.dim()}D")
        if spatial_shape is None:
            raise ValueError("spatial_shape (H, W) erforderlich für NLC -> NCHW")
        N, L, C = tensor.shape
        H, W = spatial_shape
        if H * W != L:
            raise ValueError(f"spatial_shape {spatial_shape} passt nicht zu L={L}")
        # (N, L, C) -> (N, H, W, C) -> (N, C, H, W)
        return tensor.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()

    # NC -> NLC (füge Token-Dimension hinzu)
    if source_format == "NC" and target_format == "NLC":
        if tensor.dim() != 2:
            raise ValueError(f"NC erwartet 2D-Tensor, erhalten: {tensor.dim()}D")
        return tensor.unsqueeze(1)  # (N, C) -> (N, 1, C)

    # NLC -> NC (entferne Token-Dimension durch Pooling)
    if source_format == "NLC" and target_format == "NC":
        if tensor.dim() != 3:
            raise ValueError(f"NLC erwartet 3D-Tensor, erhalten: {tensor.dim()}D")
        return tensor.mean(dim=1)  # (N, L, C) -> (N, C)

    # NCHW -> NC (Global Average Pooling)
    if source_format == "NCHW" and target_format == "NC":
        if tensor.dim() != 4:
            raise ValueError(f"NCHW erwartet 4D-Tensor, erhalten: {tensor.dim()}D")
        return tensor.mean(dim=(2, 3))  # (N, C, H, W) -> (N, C)

    # NC -> NCHW
    if source_format == "NC" and target_format == "NCHW":
        if tensor.dim() != 2:
            raise ValueError(f"NC erwartet 2D-Tensor, erhalten: {tensor.dim()}D")
        if spatial_shape is None:
            raise ValueError("spatial_shape (H, W) erforderlich für NC -> NCHW")
        H, W = spatial_shape
        N, C = tensor.shape
        # Broadcast auf alle Positionen
        return tensor.view(N, C, 1, 1).expand(N, C, H, W).contiguous()

    raise ValueError(f"Unbekannte Formatkonversion: {source_format} -> {target_format}")


# =============================================================================
# Jacobian Computation for Gradient-Times-Input (GTI)
# =============================================================================


def compute_jacobian(
    func: Callable[[Tensor], Tensor],
    x: Tensor,
    create_graph: bool = False,
) -> Tensor:
    """Berechnet die Jacobi-Matrix für eine Funktion f: R^n -> R^m.

    Nützlich für Gradient-Times-Input (GTI) Methode bei instabilen Layern,
    wo klassische LRP-Regeln versagen können.

    Args:
        func: Differenzierbare Funktion f(x).
        x: Eingabe-Tensor der Form (batch, *input_dims).
        create_graph: Falls True, wird der Berechnungsgraph für höhere
                      Ableitungen beibehalten.

    Returns:
        Jacobi-Matrix der Form (batch, *output_dims, *input_dims).

    Note:
        Für hochdimensionale Tensoren kann dies speicherintensiv sein.
        Erwäge torch.func.jacrev/jacfwd für effizientere Berechnung.

    Example:
        >>> linear = nn.Linear(4, 3)
        >>> x = torch.randn(2, 4, requires_grad=True)
        >>> J = compute_jacobian(linear, x)
        >>> J.shape
        torch.Size([2, 3, 4])
    """
    x = x.detach().requires_grad_(True)
    y = func(x)

    batch_size = x.shape[0]
    x_flat_dim = x[0].numel()
    y_flat_dim = y[0].numel()

    # Jacobi-Matrix initialisieren
    jacobian = torch.zeros(batch_size, y_flat_dim, x_flat_dim, device=x.device, dtype=x.dtype)

    # Für jede Output-Dimension die Gradienten berechnen
    y_flat = y.view(batch_size, -1)
    for i in range(y_flat_dim):
        grad_outputs = torch.zeros_like(y_flat)
        grad_outputs[:, i] = 1.0
        grad_outputs = grad_outputs.view_as(y)

        grads = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if grads is not None:
            jacobian[:, i, :] = grads.view(batch_size, -1)

    # Forme zurück in ursprüngliche Dimensionen
    output_shape = y[0].shape
    input_shape = x[0].shape
    return jacobian.view(batch_size, *output_shape, *input_shape)


def gradient_times_input(
    func: Callable[[Tensor], Tensor],
    x: Tensor,
    target_indices: Optional[Union[int, Tensor]] = None,
) -> Tensor:
    """Gradient-Times-Input (GTI) Relevanz für instabile Layer.

    GTI ist eine einfache aber effektive Attributionsmethode:
    R = x * (∂y/∂x)

    Für Layer wo LRP-ε/γ instabil sind (z.B. Layer Normalization,
    Batch Normalization), bietet GTI eine robuste Alternative.

    Args:
        func: Differenzierbare Funktion.
        x: Eingabe-Tensor.
        target_indices: Optionale Ziel-Output-Indizes für selektive Attribution.
                        Falls None, werden alle Outputs berücksichtigt.

    Returns:
        Relevanz-Tensor mit gleicher Form wie x.

    Example:
        >>> layer = nn.LayerNorm(256)
        >>> x = torch.randn(2, 100, 256)
        >>> R = gradient_times_input(layer, x)
        >>> R.shape
        torch.Size([2, 100, 256])
    """
    x = x.detach().requires_grad_(True)
    y = func(x)

    if target_indices is not None:
        # Selektive Gradientenberechnung
        if isinstance(target_indices, int):
            y_target = y.flatten(start_dim=1)[:, target_indices].sum()
        else:
            y_target = y.flatten(start_dim=1).gather(1, target_indices.view(-1, 1)).sum()
    else:
        y_target = y.sum()

    grads = torch.autograd.grad(
        outputs=y_target,
        inputs=x,
        create_graph=False,
        retain_graph=False,
    )[0]

    return x * grads


# =============================================================================
# Legacy Helper Functions (Internal Use)
# =============================================================================


def _flatten(obj: Any) -> List[Any]:
    """Flacht verschachtelte Strukturen (tuple/list) ab, behält Reihenfolge."""
    if isinstance(obj, (list, tuple)):
        res: List[Any] = []
        for it in obj:
            res.extend(_flatten(it))
        return res
    return [obj]


def _iter_tensors(obj: Any) -> Iterable[Tensor]:
    for x in _flatten(obj):
        if isinstance(x, torch.Tensor):
            yield x


def _first_tensor(obj: Any) -> Tensor:
    for t in _iter_tensors(obj):
        return t
    raise ValueError("Keine Tensor-Ausgabe im Layer gefunden")


def _to_BTC(t: Tensor) -> Tensor:
    """Bringe Tensor robust in Form (B, T, C).

    Note:
        Für explizite Formatkonvertierung bevorzuge `rearrange_activations()`.

    Regeln:
    - 4D: (B, C, H, W) -> (B, H*W, C) via rearrange_activations
    - 3D: Unverändert zurückgeben (erwartet (B, T, C)). Keine heuristische Permutation.
      Falls tatsächlich (T,B,C) vorliegt, bitte explizit vor dem Aufruf transponieren.
      Bei Verdacht (erste Achse deutlich größer als zweite) wird eine Warnung ausgegeben.
    - 2D: (B, C) -> (B, 1, C) via rearrange_activations
    """
    if t.dim() == 4:
        return rearrange_activations(t, "NCHW", "NLC")
    if t.dim() == 3:
        if t.shape[0] > t.shape[1]:
            warnings.warn(
                "_to_BTC: 3D-Input wird unverändert angenommen (B,T,C). "
                "Die erste Achse ist größer als die zweite – prüfen, ob (T,B,C) vorliegt und ggf. explizit transponieren.",
                stacklevel=2,
            )
        return t
    if t.dim() == 2:
        return rearrange_activations(t, "NC", "NLC")
    raise ValueError(f"Unerwartete Tensorform: {tuple(t.shape)}")


def aggregate_channel_relevance(R_in: Tensor) -> Tensor:
    """Aggregiere Eingangsrelevanz zu einem Vektor (C_in,)."""
    if R_in.dim() == 4:  # (B, C, H, W)
        return R_in.sum(dim=(0, 2, 3)).detach().cpu()
    if R_in.dim() == 3:  # (B, L, C)
        return R_in.sum(dim=(0, 1)).detach().cpu()
    if R_in.dim() == 2:  # (B, C)
        return R_in.sum(dim=0).detach().cpu()
    raise ValueError(f"Unerwartete R_in-Form: {tuple(R_in.shape)}")


def build_target_relevance(
    layer_output: Tensor,
    feature_index: int,
    token_reduce: str,
    target_norm: str = "sum1",
    index_axis: str = "channel",
) -> Tensor:
    """Erzeuge eine Start-Relevanz R_out ohne Gradienten.

    index_axis:
    - "channel": feature_index adressiert den Kanal (C-Achse)
    - "token":   feature_index adressiert den Token/Query (T-Achse)

    token_reduce wirkt nur bei index_axis="channel" und steuert die Verteilung über Tokens.
    """
    y = _to_BTC(layer_output)  # (B, T, C)
    B, T, C = y.shape

    base = torch.zeros_like(y)

    if index_axis == "channel":
        if feature_index < 0 or feature_index >= C:
            raise IndexError(
                f"feature_index {feature_index} außerhalb [0, {C-1}] (axis=channel)"
            )
        feat = y[..., feature_index]
        if token_reduce == "mean":
            w = torch.ones_like(feat)
        elif token_reduce == "max":
            # Echte Max-Auswahl: One-Hot pro Batch auf den Token mit maximaler Aktivierung
            idx = torch.argmax(feat, dim=1, keepdim=True)  # (B,1)
            w = torch.zeros_like(feat).scatter_(1, idx, 1.0)
        else:
            raise ValueError("token_reduce muss 'mean' oder 'max' sein")
        # Normierung über alle Tokens/Batch
        s = w.sum().clamp_min(1e-12)
        if target_norm == "sum1":
            w = w / s
        elif target_norm == "sumT":
            w = w / s * float(T)
        elif target_norm == "none":
            pass
        else:
            raise ValueError("target_norm muss 'sum1', 'sumT' oder 'none' sein")
        base[..., feature_index] = w
    elif index_axis == "token":
        if feature_index < 0 or feature_index >= T:
            raise IndexError(
                f"feature_index {feature_index} außerhalb [0, {T-1}] (axis=token)"
            )
        # Verteile Gewicht gleichmäßig über Kanäle für den gewählten Token
        w_tok = torch.ones_like(y[:, feature_index, :])  # (B, C)
        s = w_tok.sum().clamp_min(1e-12)
        if target_norm == "sum1":
            w_tok = w_tok / s
        elif target_norm == "sumT":
            # für Token-Modus interpretieren wir 'T' als C-Anzahl
            w_tok = w_tok / s * float(C)
        elif target_norm == "none":
            pass
        else:
            raise ValueError("target_norm muss 'sum1', 'sumT' oder 'none' sein")
        base[:, feature_index, :] = w_tok
    else:
        raise ValueError("index_axis muss 'channel' oder 'token' sein")

    # Form zurück wie layer_output
    if layer_output.dim() == 4:
        if index_axis == "token":
            raise ValueError("index_axis='token' wird für 4D-Outputs nicht unterstützt")
        # (B,T,C)->(B,C,H,W)
        B2, C2, H, W = layer_output.shape
        assert B2 == B and C2 == C and H * W == T
        base = base.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    return base
