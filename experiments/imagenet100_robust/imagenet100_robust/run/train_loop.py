from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..core.config import Config
from ..core.io import ensure_dir
from ..corruptions import FAMILIES, corrupt_batch, sample_severity
from ..data.dataset import ImageFolderDataset
from ..data.transforms import TrainTransform, ValTransform, clamp_01, normalize, unnormalize
from ..methods.losses import (
    BatchOutput,
    ccoblo_per_family_loss,
    cvar_loss,
    corruption_aug_loss,
    erm_loss,
    groupdro_weights_update,
    top1_accuracy,
    trades_loss,
)
from ..methods.augmix import augmix_batch
from ..models.resnet import resnet18, resnet50


@dataclass(frozen=True)
class TrainArtifacts:
    best_path: str
    last_path: str
    metrics: list[dict[str, Any]]


def _make_optimizer(cfg: Config, model: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.train.optimizer != "sgd":
        raise ValueError(f"unsupported optimizer: {cfg.train.optimizer}")
    return torch.optim.SGD(
        model.parameters(),
        lr=float(cfg.train.lr),
        momentum=float(cfg.train.momentum),
        weight_decay=float(cfg.train.weight_decay),
        nesterov=True,
    )


@torch.no_grad()
def _eval_clean(model: torch.nn.Module, loader: DataLoader, device: torch.device, *, max_batches: int = 0) -> float:
    model.eval()
    accs = []
    for bidx, (x, y) in enumerate(loader):
        if int(max_batches) > 0 and bidx >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        accs.append(float((logits.argmax(1) == y).float().mean().item()))
    model.train()
    return float(sum(accs) / max(1, len(accs)))


def train(cfg: Config, outdir: Path) -> TrainArtifacts:
    ensure_dir(outdir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ImageFolderDataset(
        cfg.data.data_root / "train",
        transform=TrainTransform(cfg.data.image_size),
        index_file=cfg.data.data_root / ".index_train.csv",
    )
    val_ds = ImageFolderDataset(
        cfg.data.data_root / "val",
        transform=ValTransform(cfg.data.image_size),
        index_file=cfg.data.data_root / ".index_val.csv",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.data.batch_size),
        shuffle=True,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.data.batch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
    )

    model_name = str(cfg.train.model_name).lower().strip()
    if model_name == "resnet50":
        model = resnet50(num_classes=int(cfg.data.num_classes)).to(device)
    elif model_name == "resnet18":
        model = resnet18(num_classes=int(cfg.data.num_classes)).to(device)
    else:
        raise ValueError(f"unknown model_name: {cfg.train.model_name!r}")

    if str(cfg.train.init_checkpoint).strip():
        ckpt_path = Path(str(cfg.train.init_checkpoint))
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        print(json.dumps({"init_checkpoint": str(ckpt_path)}))

    opt = _make_optimizer(cfg, model)
    method_amp = bool(cfg.method.params.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.train.amp) and bool(method_amp) and device.type == "cuda")
    grad_clip_norm = max(0.0, float(cfg.train.grad_clip_norm))
    max_train_batches = int(cfg.train.max_train_batches)
    max_eval_batches = int(cfg.train.max_eval_batches)
    skip_nonfinite = bool(cfg.method.params.get("skip_nonfinite", True))

    best_acc = -1.0
    best_path = outdir / "best.pt"
    last_path = outdir / "last.pt"
    metrics: list[dict[str, Any]] = []

    # GroupDRO state
    group_names = [g for g in (cfg.method.params.get("groups") or list(FAMILIES))]
    group_w = torch.ones(len(group_names), device=device) / max(1, len(group_names))

    for epoch in range(int(cfg.train.epochs)):
        t0 = time.time()
        model.train()
        running = {"loss": 0.0, "acc": 0.0, "batches": 0, "skipped": 0}
        n = 0

        for bidx, (x_norm, y) in enumerate(train_loader):
            if max_train_batches > 0 and bidx >= max_train_batches:
                break
            x_norm = x_norm.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # Keep an unnormalized [0,1] copy for corruptions/adversarial steps.
            x = clamp_01(unnormalize(x_norm))

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                if cfg.method.name == "erm":
                    logits = model(x_norm)
                    bout: BatchOutput = erm_loss(logits, y)
                elif cfg.method.name == "aug":
                    fam = str(cfg.method.params.get("family") or random.choice(list(FAMILIES)))
                    bout = corruption_aug_loss(
                        model,
                        x_norm,
                        y,
                        severity_min=int(cfg.corruptions.severity_min),
                        severity_max=int(cfg.corruptions.severity_max),
                        family=fam,
                    )
                elif cfg.method.name == "augmix":
                    p = cfg.method.params
                    mix1 = augmix_batch(
                        x,
                        severity=int(p.get("severity", 3)),
                        mixture_width=int(p.get("mixture_width", 3)),
                        mixture_depth=int(p.get("mixture_depth", -1)),
                    )
                    mix2 = augmix_batch(
                        x,
                        severity=int(p.get("severity", 3)),
                        mixture_width=int(p.get("mixture_width", 3)),
                        mixture_depth=int(p.get("mixture_depth", -1)),
                    )
                    x_all = torch.cat([normalize(x), normalize(mix1), normalize(mix2)], dim=0)
                    logits_all = model(x_all)
                    logits_clean, logits_mix1, logits_mix2 = logits_all.chunk(3, dim=0)
                    loss_ce = torch.nn.functional.cross_entropy(logits_clean, y)
                    p_clean = torch.softmax(logits_clean.float(), dim=1)
                    p_mix1 = torch.softmax(logits_mix1.float(), dim=1)
                    p_mix2 = torch.softmax(logits_mix2.float(), dim=1)
                    p_m = (p_clean + p_mix1 + p_mix2) / 3.0
                    log_p_m = torch.log(torch.clamp_min(p_m, 1e-7))
                    jsd = (
                        torch.nn.functional.kl_div(log_p_m, p_clean, reduction="batchmean")
                        + torch.nn.functional.kl_div(log_p_m, p_mix1, reduction="batchmean")
                        + torch.nn.functional.kl_div(log_p_m, p_mix2, reduction="batchmean")
                    ) / 3.0
                    loss = loss_ce + float(p.get("jsd_weight", 12.0)) * jsd
                    bout = BatchOutput(loss=loss, logs={"acc": top1_accuracy(logits_clean, y), "ce": float(loss_ce.item())})
                elif cfg.method.name == "trades":
                    p = cfg.method.params
                    bout = trades_loss(
                        model,
                        x,
                        y,
                        epsilon=float(p["epsilon"]),
                        step_size=float(p["step_size"]),
                        steps=int(p["steps"]),
                        beta=float(p["beta"]),
                    )
                elif cfg.method.name == "cvar":
                    # CVaR over corrupted loss; corruption family sampled per batch.
                    p = cfg.method.params
                    fam = str(p.get("family") or random.choice(list(FAMILIES)))
                    sev = sample_severity(cfg.corruptions.severity_min, cfg.corruptions.severity_max)
                    x_cor = normalize(corrupt_batch(x_norm, fam, severity=sev))
                    logits = model(x_cor)
                    per = torch.nn.functional.cross_entropy(logits.float(), y, reduction="none")
                    loss_tail = cvar_loss(per, alpha=float(p.get("alpha", 0.05)))
                    clean_w = float(p.get("clean_weight", 0.0))
                    tail_w = float(p.get("tail_weight", 1.0))
                    loss = tail_w * loss_tail
                    if clean_w > 0.0:
                        logits_clean = model(normalize(x))
                        loss = loss + clean_w * torch.nn.functional.cross_entropy(logits_clean, y)
                    bout = BatchOutput(loss=loss, logs={"acc": top1_accuracy(logits, y), "family": fam, "severity": sev})
                elif cfg.method.name == "groupdro":
                    # Compute group losses for each family using one forward pass.
                    fams = group_names
                    x_list = []
                    for fam in fams:
                        sev = sample_severity(cfg.corruptions.severity_min, cfg.corruptions.severity_max)
                        x_list.append(normalize(corrupt_batch(x_norm, fam, severity=sev)))
                    x_cor = torch.cat(x_list, dim=0)
                    y_rep = y.repeat(len(fams))
                    logits = model(x_cor)
                    per = torch.nn.functional.cross_entropy(logits, y_rep, reduction="none").view(len(fams), -1).mean(dim=1)
                    group_w = groupdro_weights_update(group_w, per, eta=float(cfg.method.params.get("eta", 0.05)))
                    loss = (group_w.detach() * per).sum()
                    bout = BatchOutput(loss=loss, logs={"acc": float((logits.argmax(1) == y_rep).float().mean().item())})
                elif cfg.method.name == "ccoblo":
                    p = cfg.method.params
                    bout = ccoblo_per_family_loss(
                        model,
                        x,
                        y,
                        families=list(p.get("families") or list(FAMILIES)),
                        delta=float(p.get("delta", 0.05)),
                        kappa=float(p.get("kappa", 0.0)),
                        mu=float(p.get("mu", 0.01)),
                        theta=float(p.get("theta", 0.1)),
                        eps=float(p.get("eps", 0.001)),
                        inner_steps=int(p.get("inner_steps", 60)),
                        severity_min=int(cfg.corruptions.severity_min),
                        severity_max=int(cfg.corruptions.severity_max),
                        alpha_max=float(p.get("alpha_max", 20.0)),
                        denom_floor=float(p.get("denom_floor", 1e-3)),
                        proxy_scale=float(p.get("proxy_scale", 1.0))
                        * (
                            min(1.0, float(epoch + 1) / float(max(1, int(p.get("warmup_epochs", 0)))))
                            if int(p.get("warmup_epochs", 0)) > 0
                            else 1.0
                        ),
                        proxy_reduction=str(p.get("proxy_reduction", "sum")),
                        ce_weight=float(p.get("ce_weight", 1.0)),
                        robust_ce_weight=float(p.get("robust_ce_weight", 0.0)),
                        robust_ce_mode=str(p.get("robust_ce_mode", "mean")),
                        robust_ce_alpha=float(p.get("robust_ce_alpha", 0.25)),
                        smooth_gap_weight=float(p.get("smooth_gap_weight", 0.0)),
                        smooth_gap_tau=float(p.get("smooth_gap_tau", 0.25)),
                    )
                else:
                    raise ValueError(f"unknown method: {cfg.method.name}")

            if skip_nonfinite and not torch.isfinite(bout.loss):
                opt.zero_grad(set_to_none=True)
                running["skipped"] += 1
                continue

            scaler.scale(bout.loss).backward()
            if grad_clip_norm > 0.0:
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if skip_nonfinite and not torch.isfinite(grad_norm):
                    opt.zero_grad(set_to_none=True)
                    scaler.update()
                    running["skipped"] += 1
                    continue
            scaler.step(opt)
            scaler.update()

            bsz = int(y.size(0))
            running["loss"] += float(bout.loss.item()) * bsz
            running["acc"] += float(bout.logs.get("acc", bout.logs.get("acc_clean", 0.0))) * bsz
            running["batches"] += 1
            n += bsz

        train_loss = running["loss"] / max(1, n)
        train_acc = running["acc"] / max(1, n)
        val_acc = _eval_clean(model, val_loader, device, max_batches=max_eval_batches)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc_clean": val_acc,
            "train_batches": int(running["batches"]),
            "skipped_batches": int(running["skipped"]),
            "seconds": float(time.time() - t0),
        }
        metrics.append(row)
        print(json.dumps(row))

        torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": {"method": cfg.method.name}}, last_path)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "cfg": {"method": cfg.method.name}}, best_path)

    return TrainArtifacts(best_path=str(best_path), last_path=str(last_path), metrics=metrics)
