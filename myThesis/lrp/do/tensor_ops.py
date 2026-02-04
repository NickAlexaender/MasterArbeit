from __future__ import annotations
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union
import warnings
import torch
import torch.nn as nn
from torch import Tensor


# wie sichern wir Division zur Vermeidung von NaN/Inf in LRP-Berechnungen

def safe_divide(
    numerator: Tensor,
    denominator: Tensor,
    eps: float = 1e-9,
    signed_eps: bool = True,
) -> Tensor:
    if signed_eps:
        # Vorzeichen-erhaltende Stabilisierung (LRP-ε Standardregel)
        stabilizer = eps * torch.sign(denominator) + (denominator == 0).float() * eps
    else:
        stabilizer = eps
    return numerator / (denominator + stabilizer)

# Wir müssen Aktivierungen konvertieren zwischen Detectron2- und Transformer-Formaten

def rearrange_activations(
    tensor: Tensor,
    source_format: str,
    target_format: str,
    spatial_shape: Optional[Tuple[int, int]] = None,
) -> Tensor:
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


# Wo instabile Layer versagen könnten, kann GTI nützlich sein

def compute_jacobian(
    func: Callable[[Tensor], Tensor],
    x: Tensor,
    create_graph: bool = False,
) -> Tensor:
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

# GTI ist eine einfache aber effektive Attributionsmethode: R = x * (∂y/∂x)

def gradient_times_input(
    func: Callable[[Tensor], Tensor],
    x: Tensor,
    target_indices: Optional[Union[int, Tensor]] = None,
) -> Tensor:
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


def _flatten(obj: Any) -> List[Any]:
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
