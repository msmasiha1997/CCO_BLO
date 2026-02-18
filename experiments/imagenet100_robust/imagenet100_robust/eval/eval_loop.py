from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..core.config import Config
from ..corruptions import FAMILIES, corrupt_batch, sample_severity
from ..data.dataset import ImageFolderDataset
from ..data.transforms import ValTransform, clamp_01, normalize, unnormalize
from ..methods.losses import _margin
from ..models.resnet import resnet18, resnet50


def _fscore_ec(ec: float, target: float) -> float:
    return float(2.0 * min(ec, target) / (ec + target + 1e-12))


@torch.no_grad()
def evaluate_run(cfg: Config, run_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = run_dir / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = str(cfg.train.model_name).lower().strip()
    if model_name == "resnet50":
        model = resnet50(num_classes=int(cfg.data.num_classes))
    elif model_name == "resnet18":
        model = resnet18(num_classes=int(cfg.data.num_classes))
    else:
        raise ValueError(f"unknown model_name: {cfg.train.model_name!r}")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    val_ds = ImageFolderDataset(
        cfg.data.data_root / "val",
        transform=ValTransform(cfg.data.image_size),
        index_file=cfg.data.data_root / ".index_val.csv",
        write_index=False,
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg.data.batch_size), shuffle=False, num_workers=int(cfg.data.num_workers))

    # Clean acc
    clean_acc = []
    for x_norm, y in val_loader:
        x_norm = x_norm.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x_norm)
        clean_acc.append(float((logits.argmax(1) == y).float().mean().item()))
    clean_acc = float(sum(clean_acc) / max(1, len(clean_acc)))

    # Robust under training corruption distribution (uniform family + uniform severity).
    fam_stats: dict[str, dict[str, float]] = {f: {"acc": 0.0, "ec": 0.0, "n": 0.0} for f in FAMILIES}
    kappa = float(cfg.method.params.get("kappa", 0.0)) if cfg.method.name == "ccoblo" else 0.0
    delta = float(cfg.method.params.get("delta", 0.05)) if cfg.method.name == "ccoblo" else 0.05
    target = 1.0 - float(delta)

    for x_norm, y in val_loader:
        x_norm = x_norm.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        x = clamp_01(unnormalize(x_norm))
        for fam in FAMILIES:
            sev = sample_severity(cfg.corruptions.severity_min, cfg.corruptions.severity_max)
            x_cor = normalize(corrupt_batch(x_norm, fam, severity=sev))
            logits = model(x_cor)
            acc = float((logits.argmax(1) == y).float().mean().item())
            margin = _margin(logits, y)
            ec = float((margin >= float(kappa)).float().mean().item())
            fam_stats[fam]["acc"] += acc * y.size(0)
            fam_stats[fam]["ec"] += ec * y.size(0)
            fam_stats[fam]["n"] += float(y.size(0))

    out: dict[str, Any] = {"clean_acc": clean_acc}
    for fam, s in fam_stats.items():
        n = max(1.0, s["n"])
        acc = s["acc"] / n
        ec = s["ec"] / n
        out[f"{fam}_acc"] = acc
        out[f"{fam}_ec"] = ec
        out[f"{fam}_f_ec"] = _fscore_ec(ec, target)

    (run_dir / "eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
