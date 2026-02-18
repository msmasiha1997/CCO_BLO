from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn.functional as F

from ..corruptions import corrupt_batch
from ..data.transforms import clamp_01, normalize, unnormalize


def _severity_to_brightness(severity: int) -> float:
    return {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50}[int(severity)]


def _severity_to_contrast(severity: int) -> float:
    return {1: 0.10, 2: 0.20, 3: 0.30, 4: 0.40, 5: 0.50}[int(severity)]


def _op_brightness(x01: torch.Tensor, severity: int) -> torch.Tensor:
    a = _severity_to_brightness(severity)
    # Randomly brighten or darken.
    sign = -1.0 if random.random() < 0.5 else 1.0
    return clamp_01(x01 + sign * a)


def _op_contrast(x01: torch.Tensor, severity: int) -> torch.Tensor:
    a = _severity_to_contrast(severity)
    mean = x01.mean(dim=(2, 3), keepdim=True)
    factor = 1.0 + (a if random.random() < 0.5 else -a)
    return clamp_01((x01 - mean) * factor + mean)


def _op_sharpness(x01: torch.Tensor, severity: int) -> torch.Tensor:
    # Unsharp mask-like: x + λ(x - blur(x))
    lam = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}[int(severity)]
    # cheap blur: avgpool
    blur = F.avg_pool2d(x01, kernel_size=3, stride=1, padding=1)
    return clamp_01(x01 + lam * (x01 - blur))


def _op_identity(x01: torch.Tensor, severity: int) -> torch.Tensor:  # noqa: ARG001
    return x01


def _available_ops() -> list[Callable[[torch.Tensor, int], torch.Tensor]]:
    # These are simple tensor ops + our differentiable corruption families.
    return [
        _op_identity,
        _op_brightness,
        _op_contrast,
        _op_sharpness,
        lambda x, s: corrupt_batch(normalize(x), "noise", severity=s),
        lambda x, s: corrupt_batch(normalize(x), "blur", severity=s),
        lambda x, s: corrupt_batch(normalize(x), "weather", severity=s),
        lambda x, s: corrupt_batch(normalize(x), "digital", severity=s),
    ]


@torch.no_grad()
def augmix_batch(
    x01: torch.Tensor,
    *,
    severity: int,
    mixture_width: int,
    mixture_depth: int,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    AugMix-style mixing implemented with tensor operations.

    This is intended as a practical baseline that uses the same corruption primitives
    as the chance-constrained experiment (noise/blur/weather/digital + simple color ops).

    Inputs:
      x01: unnormalized images in [0,1], shape (B,C,H,W)
    """
    b = x01.size(0)
    ops = _available_ops()

    ws = torch.distributions.Dirichlet(torch.full((mixture_width,), float(alpha), device=x01.device)).sample()
    m = torch.distributions.Beta(float(alpha), float(alpha)).sample().to(x01.device)

    mix = torch.zeros_like(x01)
    for i in range(int(mixture_width)):
        x_aug = x01
        depth = int(mixture_depth) if int(mixture_depth) > 0 else random.randint(1, 3)
        for _ in range(depth):
            op = random.choice(ops)
            x_aug = op(x_aug, int(severity))
        mix = mix + ws[i] * x_aug

    out = (1.0 - m) * x01 + m * mix
    return clamp_01(out)

