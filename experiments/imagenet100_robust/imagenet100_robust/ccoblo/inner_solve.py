from __future__ import annotations

import torch

from .smoothing_torch import dh_theta, ddh_theta


@torch.no_grad()
def solve_s_star_bisect(
    gvals: torch.Tensor,
    *,
    delta: float,
    theta: float,
    eps: float,
    steps: int,
    s_init: float | None = None,
) -> float:
    """
    Solve for s* such that dG/ds = 0 with a monotone bisection.

      dG/ds = 1 - (1/(δM)) Σ dh_theta(g_i - s) + eps*s

    gvals should be 1D (M,).
    """
    # Work in fp32 for numerical stability (esp. under AMP).
    g = gvals.detach().flatten().float()
    m = float(g.numel())
    if m <= 0:
        raise ValueError("empty gvals")

    def dG_ds(s: torch.Tensor) -> torch.Tensor:
        t = g - s
        return 1.0 - (1.0 / (delta * m)) * dh_theta(t, theta).sum() + eps * s

    lo = g.min() - 10.0
    hi = g.max() + 10.0
    d_lo = dG_ds(lo)
    d_hi = dG_ds(hi)
    for _ in range(80):
        if d_lo <= 0.0 and d_hi >= 0.0:
            break
        width = hi - lo
        if d_lo > 0.0:
            lo = lo - 2.0 * width
            d_lo = dG_ds(lo)
        if d_hi < 0.0:
            hi = hi + 2.0 * width
            d_hi = dG_ds(hi)
    else:
        # Fallback: use the empirical (1-δ)-quantile of g as a robust initializer.
        q = torch.quantile(g, 1.0 - float(delta))
        return float(q.item())

    for _ in range(int(steps)):
        mid = 0.5 * (lo + hi)
        if dG_ds(mid) <= 0.0:
            lo = mid
        else:
            hi = mid
    return float((0.5 * (lo + hi)).item())


@torch.no_grad()
def d2G_ds2(gvals: torch.Tensor, *, delta: float, theta: float, eps: float, s_star: float) -> float:
    g = gvals.detach().flatten()
    m = float(g.numel())
    t = g - float(s_star)
    return float(((1.0 / (delta * m)) * ddh_theta(t, theta).sum() + eps).item())
