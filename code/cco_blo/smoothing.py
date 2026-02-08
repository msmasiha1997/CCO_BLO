from __future__ import annotations

import numpy as np


def h_theta(x: np.ndarray, theta: float) -> np.ndarray:
    x = np.asarray(x)
    out = np.maximum(x, 0.0)
    mask = (x >= -theta) & (x <= theta)
    if np.any(mask):
        xm = x[mask]
        out[mask] = (
            -(xm**4) / (16.0 * theta**3)
            + (3.0 * xm**2) / (8.0 * theta)
            + xm / 2.0
            + (3.0 * theta) / 16.0
        )
    return out


def dh_theta(x: np.ndarray, theta: float) -> np.ndarray:
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    out[x > theta] = 1.0
    mask = (x >= -theta) & (x <= theta)
    if np.any(mask):
        xm = x[mask]
        out[mask] = -(xm**3) / (4.0 * theta**3) + (3.0 * xm) / (4.0 * theta) + 0.5
    return out


def ddh_theta(x: np.ndarray, theta: float) -> np.ndarray:
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    mask = (x >= -theta) & (x <= theta)
    if np.any(mask):
        xm = x[mask]
        out[mask] = -(3.0 * xm**2) / (4.0 * theta**3) + (3.0) / (4.0 * theta)
    return out
