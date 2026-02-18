from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
from PIL import Image, ImageEnhance


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).contiguous()
    return arr.float().div(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device)[:, None, None]
    return (x - mean) / std


@dataclass(frozen=True)
class TrainTransform:
    image_size: int

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # RandomResizedCrop (approx)
        w, h = img.size
        scale = random.uniform(0.08, 1.0)
        aspect = math.exp(random.uniform(math.log(3 / 4), math.log(4 / 3)))
        crop_w = int(round(math.sqrt(scale * w * h * aspect)))
        crop_h = int(round(math.sqrt(scale * w * h / aspect)))
        crop_w = max(1, min(crop_w, w))
        crop_h = max(1, min(crop_h, h))
        x0 = random.randint(0, w - crop_w) if w > crop_w else 0
        y0 = random.randint(0, h - crop_h) if h > crop_h else 0
        img = img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((self.image_size, self.image_size), Image.BILINEAR)

        # RandomHorizontalFlip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        x = pil_to_tensor(img)
        x = normalize(x)
        return x


@dataclass(frozen=True)
class ValTransform:
    image_size: int

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # Resize shorter side to 256 then center crop.
        w, h = img.size
        short = min(w, h)
        if short == 0:
            raise ValueError("invalid image size")
        scale = 256 / short
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        left = (new_w - self.image_size) // 2
        top = (new_h - self.image_size) // 2
        img = img.crop((left, top, left + self.image_size, top + self.image_size))

        x = pil_to_tensor(img)
        x = normalize(x)
        return x


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device)[:, None, None]
    return x * std + mean


def clamp_01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def enhance_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def enhance_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def enhance_sharpness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Sharpness(img).enhance(factor)

