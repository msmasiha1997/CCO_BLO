from __future__ import annotations

import torch


def h_theta(x: torch.Tensor, theta: float) -> torch.Tensor:
    # piecewise quartic smoothing of max(x,0)
    out = torch.clamp_min(x, 0.0)
    theta_t = x.new_tensor(float(theta))
    mask = (x >= -theta_t) & (x <= theta_t)
    if mask.any():
        xm = x[mask]
        # Compute in fp32 for stability / autocast compatibility, then cast back.
        xm_f = xm.float()
        th_f = xm_f.new_tensor(float(theta))
        out_m = (
            -(xm_f**4) / (16.0 * th_f**3)
            + (3.0 * xm_f**2) / (8.0 * th_f)
            + xm_f / 2.0
            + (3.0 * th_f) / 16.0
        ).to(dtype=x.dtype)
        out = out.clone()
        out[mask] = out_m
    return out


def dh_theta(x: torch.Tensor, theta: float) -> torch.Tensor:
    theta_t = x.new_tensor(float(theta))
    out = torch.zeros_like(x, dtype=x.dtype)
    out = torch.where(x > theta_t, torch.ones_like(out), out)
    mask = (x >= -theta_t) & (x <= theta_t)
    if mask.any():
        xm = x[mask]
        xm_f = xm.float()
        th_f = xm_f.new_tensor(float(theta))
        out_m = (-(xm_f**3) / (4.0 * th_f**3) + (3.0 * xm_f) / (4.0 * th_f) + 0.5).to(dtype=x.dtype)
        out = out.clone()
        out[mask] = out_m
    return out


def ddh_theta(x: torch.Tensor, theta: float) -> torch.Tensor:
    theta_t = x.new_tensor(float(theta))
    out = torch.zeros_like(x, dtype=x.dtype)
    mask = (x >= -theta_t) & (x <= theta_t)
    if mask.any():
        xm = x[mask]
        xm_f = xm.float()
        th_f = xm_f.new_tensor(float(theta))
        out_m = (-(3.0 * xm_f**2) / (4.0 * th_f**3) + (3.0) / (4.0 * th_f)).to(dtype=x.dtype)
        out = out.clone()
        out[mask] = out_m
    return out
