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


def _solve_s_star(
    *,
    gvals: np.ndarray,
    delta: float,
    theta: float,
    eps: float,
    inner_steps: int,
    inner_solver: str,
    gamma: float,
    s_init: float,
) -> float:
    """
    Solve (approximately) for s* = argmin_s G_hat(s) where
      dG/ds = 1 - (1/(δM)) Σ dh_theta(g_j - s) + eps*s
    which is strictly increasing in s (since d2G/ds2 >= eps > 0).
    """
    M = float(len(gvals))

    def dG_ds(s: float) -> float:
        t = gvals - s
        dh = dh_theta(t, theta)
        return float(1.0 - (1.0 / (delta * M)) * np.sum(dh) + eps * s)

    def d2G_ds2(s: float) -> float:
        t = gvals - s
        ddh = ddh_theta(t, theta)
        return float((1.0 / (delta * M)) * np.sum(ddh) + eps)

    if inner_solver == "gd":
        s = float(s_init)
        for _ in range(inner_steps):
            s -= float(gamma) * dG_ds(s)
        return float(s)

    if inner_solver == "newton":
        s = float(s_init)
        for _ in range(inner_steps):
            s -= float(gamma) * (dG_ds(s) / d2G_ds2(s))
        return float(s)

    if inner_solver == "bisect":
        lo = float(np.min(gvals) - 1.0)
        hi = float(np.max(gvals) + 1.0)

        # Expand until we bracket the root: dG(lo) <= 0 <= dG(hi)
        d_lo = dG_ds(lo)
        d_hi = dG_ds(hi)
        for _ in range(50):
            if d_lo <= 0.0 and d_hi >= 0.0:
                break
            width = hi - lo
            if d_lo > 0.0:
                lo -= 2.0 * width
                d_lo = dG_ds(lo)
            if d_hi < 0.0:
                hi += 2.0 * width
                d_hi = dG_ds(hi)
        else:
            raise RuntimeError("failed to bracket root for dG/ds=0")

        for _ in range(inner_steps):
            mid = 0.5 * (lo + hi)
            if dG_ds(mid) <= 0.0:
                lo = mid
            else:
                hi = mid
        return float(0.5 * (lo + hi))

    raise ValueError(f"unknown inner_solver: {inner_solver}")


def run_ccoblo(
    *,
    example_name: str,
    dim: int,
    seed: int,
    delta: float,
    mu: float,
    mu_final: float | None = None,
    mu_schedule: str = "constant",
    theta: float,
    eps: float,
    M: int,
    outer_steps: int,
    inner_steps: int,
    eta: float,
    gamma: float,
    inner_solver: str = "bisect",
    outer_optimizer: str = "sgd",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
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

    # outer optimizer state
    adam_m = np.zeros_like(x)
    adam_v = np.zeros_like(x)

    def mu_at(k: int) -> float:
        if mu_final is None or mu_schedule == "constant" or outer_steps <= 1:
            return float(mu)
        if mu_schedule == "linear":
            t = k / (outer_steps - 1)
            return float(mu + t * (mu_final - mu))
        if mu_schedule == "geom":
            t = k / (outer_steps - 1)
            if mu <= 0 or mu_final <= 0:
                raise ValueError("geom mu_schedule requires mu>0 and mu_final>0")
            return float(mu * ((mu_final / mu) ** t))
        raise ValueError(f"unknown mu_schedule: {mu_schedule}")

    for _k in range(outer_steps):
        mu_k = mu_at(_k)
        if resample_z_each_outer:
            z_batch = example.sample_z(rng, M)

        gvals = example.g(x, z_batch)
        s = _solve_s_star(
            gvals=gvals,
            delta=delta,
            theta=theta,
            eps=eps,
            inner_steps=inner_steps,
            inner_solver=inner_solver,
            gamma=gamma,
            s_init=float(s),
        )

        # implicit gradient ds*/dx (scalar s -> vector)
        _, _, d2G_ds2, d2G_dxds = _g_hat_and_partials(
            example, x, s, z_batch, delta=delta, theta=theta, eps=eps
        )
        ds_dx = -(d2G_dxds / d2G_ds2)

        # outer objective: f(x) + ([s]_+)^2 / (2 mu)
        fval = float(example.f(x))
        s_pos = max(s, 0.0)
        pen = float((s_pos * s_pos) / (2.0 * mu_k))

        grad = example.grad_f(x)
        if s > 0.0:
            grad = grad + (s / mu_k) * ds_dx

        if outer_optimizer == "sgd":
            x = x - eta * grad
        elif outer_optimizer == "adam":
            adam_m = adam_beta1 * adam_m + (1.0 - adam_beta1) * grad
            adam_v = adam_beta2 * adam_v + (1.0 - adam_beta2) * (grad * grad)
            m_hat = adam_m / (1.0 - (adam_beta1 ** (_k + 1)))
            v_hat = adam_v / (1.0 - (adam_beta2 ** (_k + 1)))
            x = x - eta * m_hat / (np.sqrt(v_hat) + adam_eps)
        else:
            raise ValueError(f"unknown outer_optimizer: {outer_optimizer}")
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
