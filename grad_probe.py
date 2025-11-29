"""
Small utilities for measuring and adjusting gradient interactions across objectives.

These mirror the minimal functionality expected by train.py.
"""

from typing import Dict, List, Tuple

import torch

# Allow running as a script (no package) or as a module
try:
    from .utils import robust_cosine
except ImportError:
    from utils import robust_cosine


def cosine_matrix(grads: Dict[str, torch.Tensor]) -> Tuple[List[str], torch.Tensor]:
    """Return objective names and pairwise cosine matrix."""
    keys = list(grads.keys())
    n = len(keys)
    M = torch.zeros((n, n), device=next(iter(grads.values())).device)
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            M[i, j] = robust_cosine(grads[ki], grads[kj])
    return keys, M


def conflict_fraction(M: torch.Tensor, tau: float = -0.1) -> float:
    """Fraction of off-diagonal pairs with cosine below threshold tau."""
    if M.numel() == 0 or M.shape[0] <= 1:
        return 0.0
    n = M.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=M.device)
    vals = M[mask]
    return float((vals < tau).sum().item() / vals.numel())


def project_nonconflicting(g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
    """
    PCGrad-style projection: if gradients conflict (negative dot), project g1
    onto the normal plane of g2; otherwise return g1 unchanged.
    """
    if g1.numel() == 0 or g2.numel() == 0:
        return g1
    dot = torch.dot(g1, g2)
    if dot < 0:
        g1 = g1 - dot / (g2.norm() ** 2 + 1e-12) * g2
    return g1
