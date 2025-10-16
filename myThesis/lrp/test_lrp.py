"""
Sanity tests for LRP engine: MLP conservation and simple attention smoke test.
Run: python -m myThesis.lrp.test_lrp
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import LRPConfig, LRPTracer, propagate_lrp


def test_mlp_epsilon(device: str = "cpu"):
    torch.manual_seed(0)
    mlp = nn.Sequential(
        nn.Linear(4, 3, bias=False),
        nn.ReLU(),
        nn.Linear(3, 2, bias=False),
    ).to(device)
    x = torch.randn(5, 4, device=device)
    tracer = LRPTracer(LRPConfig(rule_linear="epsilon", epsilon=1e-6, verbose_checks=True))
    tracer.add_from_model(mlp)
    with torch.no_grad():
        y = mlp(x)
    R_out = F.relu(y)
    last_lin = [m for m in mlp.modules() if isinstance(m, nn.Linear)][-1]
    rels = propagate_lrp(mlp, tracer, last_lin, R_out, LRPConfig(rule_linear="epsilon"))
    R_in = rels.get(last_lin)
    assert R_in is not None
    diff = float(abs(R_in.sum() - R_out.sum()).cpu())
    print("epsilon diff:", diff)
    assert diff < 1e-3


def test_mlp_zplus(device: str = "cpu"):
    torch.manual_seed(0)
    mlp = nn.Sequential(
        nn.Linear(4, 3, bias=False),
        nn.ReLU(),
        nn.Linear(3, 2, bias=False),
    ).to(device)
    x = torch.rand(5, 4, device=device)  # non-negative for z+
    tracer = LRPTracer(LRPConfig(rule_linear="zplus", epsilon=1e-6, verbose_checks=True))
    tracer.add_from_model(mlp)
    with torch.no_grad():
        y = mlp(x)
    R_out = F.relu(y)
    last_lin = [m for m in mlp.modules() if isinstance(m, nn.Linear)][-1]
    rels = propagate_lrp(mlp, tracer, last_lin, R_out, LRPConfig(rule_linear="zplus"))
    R_in = rels.get(last_lin)
    assert R_in is not None
    diff = float(abs(R_in.sum() - R_out.sum()).cpu())
    print("zplus diff:", diff)
    assert diff < 1e-3


def test_mlp_alphabeta(device: str = "cpu"):
    torch.manual_seed(0)
    mlp = nn.Sequential(
        nn.Linear(4, 3, bias=False),
        nn.ReLU(),
        nn.Linear(3, 2, bias=False),
    ).to(device)
    x = torch.randn(5, 4, device=device)
    tracer = LRPTracer(LRPConfig(rule_linear="alphabeta", alpha=1.0, beta=0.0, verbose_checks=True))
    tracer.add_from_model(mlp)
    with torch.no_grad():
        y = mlp(x)
    R_out = F.relu(y)
    last_lin = [m for m in mlp.modules() if isinstance(m, nn.Linear)][-1]
    rels = propagate_lrp(mlp, tracer, last_lin, R_out, LRPConfig(rule_linear="alphabeta", alpha=1.0, beta=0.0))
    R_in = rels.get(last_lin)
    assert R_in is not None
    diff = float(abs(R_in.sum() - R_out.sum()).cpu())
    print("alphabeta diff:", diff)
    assert diff < 1e-3


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    test_mlp_epsilon(dev)
    test_mlp_zplus(dev)
    test_mlp_alphabeta(dev)
    print("All LRP tests passed.")
