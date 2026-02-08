from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .examples import Example, make_example
from .smoothing import ddh_theta, dh_theta, h_theta


@dataclass(frozen=True)
class RunResult:
    x: np.ndarray
    s: float
    history: dict[str, list[float]]


def _g_hat_and_partials(
    example: Example,
    x: np.ndarray,
    s: float,
    z_batch: np.ndarray,
    *,
    delta: float,
    theta: float,
    eps: float,
) -> tuple[float, float, float, np.ndarray]:
    """
    Returns:
      G_hat(x,s), dG/ds, d2G/ds2, d2G/dxds (vector)
    """
    gvals = example.g(x, z_batch)
    t = gvals - s
    M = float(len(z_batch))

    G = float(s + (1.0 / (delta * M)) * np.sum(h_theta(t, theta)) + 0.5 * eps * s * s)

    dh = dh_theta(t, theta)
    dG_ds = float(1.0 - (1.0 / (delta * M)) * np.sum(dh) + eps * s)

    ddh = ddh_theta(t, theta)
    d2G_ds2 = float((1.0 / (delta * M)) * np.sum(ddh) + eps)

    # d2G/dxds = -(1/(δM)) Σ ddh(t_j) * ∂g/∂x (at z_j)
    grad_g = example.grad_g(x, z_batch)  # shape (M, d)
    d2G_dxds = -((1.0 / (delta * M)) * (ddh[:, None] * grad_g).sum(axis=0))

    return G, dG_ds, d2G_ds2, d2G_dxds


def run_ccoblo(
    *,
    example_name: str,
    dim: int,
    seed: int,
    delta: float,
    mu: float,
    theta: float,
    eps: float,
    M: int,
    outer_steps: int,
    inner_steps: int,
    eta: float,
    gamma: float,
    x0: list[float] | None = None,
    x_min: float | list[float] | None = None,
    x_max: float | list[float] | None = None,
    resample_z_each_outer: bool = False,
    eval_samples: int = 10_000,
) -> RunResult:
    rng = np.random.default_rng(seed)
    example = make_example(example_name, dim)

    if x0 is None:
        x = rng.normal(scale=0.1, size=(example.dim,))
    else:
        x = np.array(x0, dtype=float).reshape((example.dim,))

    if x_min is not None:
        x_min_arr = np.broadcast_to(np.array(x_min, dtype=float), x.shape)
    else:
        x_min_arr = None
    if x_max is not None:
        x_max_arr = np.broadcast_to(np.array(x_max, dtype=float), x.shape)
    else:
        x_max_arr = None

    s = 0.0
    z_batch = example.sample_z(rng, M)

    hist_f: list[float] = []
    hist_pen: list[float] = []
    hist_s: list[float] = []
    hist_ec: list[float] = []

    for _k in range(outer_steps):
        if resample_z_each_outer:
            z_batch = example.sample_z(rng, M)

        # inner loop: approximately minimize G_hat(x, s) over scalar s
        s_inner = float(s)
        for _ in range(inner_steps):
            _, dG_ds, _, _ = _g_hat_and_partials(
                example, x, s_inner, z_batch, delta=delta, theta=theta, eps=eps
            )
            s_inner = float(s_inner - gamma * dG_ds)
        s = float(s_inner)

        # implicit gradient ds*/dx (scalar s -> vector)
        _, _, d2G_ds2, d2G_dxds = _g_hat_and_partials(
            example, x, s, z_batch, delta=delta, theta=theta, eps=eps
        )
        ds_dx = -(d2G_dxds / d2G_ds2)

        # outer objective: f(x) + ([s]_+)^2 / (2 mu)
        fval = float(example.f(x))
        s_pos = max(s, 0.0)
        pen = float((s_pos * s_pos) / (2.0 * mu))

        grad = example.grad_f(x)
        if s > 0.0:
            grad = grad + (s / mu) * ds_dx

        x = x - eta * grad
        if x_min_arr is not None or x_max_arr is not None:
            x = np.clip(x, -np.inf if x_min_arr is None else x_min_arr, np.inf if x_max_arr is None else x_max_arr)

        # evaluation: empirical coverage on fresh samples
        z_eval = example.sample_z(rng, eval_samples)
        ec = float(np.mean(example.g(x, z_eval) <= 0.0))

        hist_f.append(fval)
        hist_pen.append(pen)
        hist_s.append(s)
        hist_ec.append(ec)

    return RunResult(x=x, s=s, history={"f": hist_f, "penalty": hist_pen, "s": hist_s, "ec": hist_ec})
