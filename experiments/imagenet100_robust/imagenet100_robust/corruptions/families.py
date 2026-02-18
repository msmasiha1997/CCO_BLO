from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..data.transforms import clamp_01, unnormalize


FAMILIES = ("noise", "blur", "weather", "digital")


@dataclass(frozen=True)
class CorruptionParams:
    severity: int


def _severity_to_sigma(severity: int) -> float:
    # Map {1..5} -> sigma in a mild range.
    return {1: 0.02, 2: 0.04, 3: 0.06, 4: 0.08, 5: 0.10}[int(severity)]


def _severity_to_blur_kernel(severity: int) -> int:
    return {1: 3, 2: 5, 3: 7, 4: 9, 5: 11}[int(severity)]


def _gaussian_kernel2d(ks: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel


def _apply_depthwise_conv(x: torch.Tensor, kernel2d: torch.Tensor) -> torch.Tensor:
    # x: (B,C,H,W), kernel2d: (K,K)
    b, c, _, _ = x.shape
    k = kernel2d.shape[0]
    weight = kernel2d.expand(c, 1, k, k).contiguous()
    return F.conv2d(x, weight, padding=k // 2, groups=c)


def corrupt_batch(
    x_normed: torch.Tensor,
    family: str,
    *,
    severity: int,
) -> torch.Tensor:
    """
    Apply a random corruption family to a batch.

    Input is assumed to be ImageNet-normalized (mean/std). We unnormalize to [0,1],
    corrupt, then renormalize in the caller (model expects normalized inputs).
    """
    family = str(family).lower()
    if family not in FAMILIES:
        raise ValueError(f"unknown corruption family: {family}")

    x = unnormalize(x_normed)
    x = clamp_01(x)

    if family == "noise":
        sigma = _severity_to_sigma(severity)
        x = x + sigma * torch.randn_like(x)
        return clamp_01(x)

    if family == "blur":
        ks = _severity_to_blur_kernel(severity)
        sigma = 0.3 * ks
        k2d = _gaussian_kernel2d(ks, sigma, x.device, x.dtype)
        x = _apply_depthwise_conv(x, k2d)
        return clamp_01(x)

    if family == "weather":
        # Simple "fog": blend towards a light gray with severity-controlled alpha + low-freq noise.
        alpha = {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50}[int(severity)]
        gray = torch.full_like(x, 0.85)
        noise = torch.randn_like(x) * 0.05
        # Smooth noise a bit (cheap average pooling).
        noise = F.avg_pool2d(noise, kernel_size=5, stride=1, padding=2)
        x = (1 - alpha) * x + alpha * clamp_01(gray + noise)
        return clamp_01(x)

    if family == "digital":
        # Pixelation via downsample/upsample + mild quantization.
        scale = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.35, 5: 0.25}[int(severity)]
        b, c, h, w = x.shape
        hh = max(1, int(round(h * scale)))
        ww = max(1, int(round(w * scale)))
        x_small = F.interpolate(x, size=(hh, ww), mode="bilinear", align_corners=False)
        x = F.interpolate(x_small, size=(h, w), mode="bilinear", align_corners=False)
        # Quantize to N levels.
        levels = {1: 256, 2: 128, 3: 64, 4: 32, 5: 16}[int(severity)]
        x = torch.round(x * (levels - 1)) / (levels - 1)
        return clamp_01(x)

    raise RuntimeError("unreachable")


def sample_severity(severity_min: int, severity_max: int) -> int:
    return random.randint(int(severity_min), int(severity_max))

