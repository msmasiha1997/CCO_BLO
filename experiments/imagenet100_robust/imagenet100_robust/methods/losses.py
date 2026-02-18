from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ..ccoblo.inner_solve import d2G_ds2, solve_s_star_bisect
from ..ccoblo.smoothing_torch import dh_theta
from ..corruptions import corrupt_batch, sample_severity
from ..data.transforms import clamp_01, normalize, unnormalize


def top1_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == y).float().mean().item())


def _margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    y_logits = logits.gather(1, y[:, None]).squeeze(1)
    tmp = logits.clone()
    tmp[torch.arange(logits.size(0), device=logits.device), y] = -1e9
    other = tmp.max(dim=1).values
    return y_logits - other


def violation_from_logits(logits: torch.Tensor, y: torch.Tensor, *, kappa: float) -> torch.Tensor:
    return float(kappa) - _margin(logits, y)


@dataclass(frozen=True)
class BatchOutput:
    loss: torch.Tensor
    logs: dict[str, Any]


def erm_loss(logits: torch.Tensor, y: torch.Tensor) -> BatchOutput:
    loss = F.cross_entropy(logits, y)
    return BatchOutput(loss=loss, logs={"acc": top1_accuracy(logits, y), "ce": float(loss.item())})


def corruption_aug_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    severity_min: int,
    severity_max: int,
    family: str,
) -> BatchOutput:
    sev = sample_severity(severity_min, severity_max)
    x_cor01 = corrupt_batch(x, family, severity=sev)
    logits = model(normalize(x_cor01))
    out = erm_loss(logits, y)
    return BatchOutput(loss=out.loss, logs={**out.logs, "family": family, "severity": sev})


def cvar_loss(per_example_loss: torch.Tensor, *, alpha: float) -> torch.Tensor:
    # alpha is tail probability, e.g. 0.05 means average top 5% losses.
    losses = per_example_loss.float()
    finite = torch.isfinite(losses)
    if not finite.any():
        return losses.new_tensor(float("nan"))
    losses = losses[finite]
    b = losses.numel()
    k = max(1, int(round(float(alpha) * b)))
    vals, _ = torch.topk(losses, k=k, largest=True)
    return vals.mean()


def groupdro_weights_update(weights: torch.Tensor, group_losses: torch.Tensor, *, eta: float) -> torch.Tensor:
    log_w = torch.log(weights + 1e-12) + float(eta) * group_losses.detach()
    log_w = log_w - log_w.max()
    w = torch.exp(log_w)
    return w / (w.sum() + 1e-12)


def trades_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epsilon: float,
    step_size: float,
    steps: int,
    beta: float,
) -> BatchOutput:
    # TRADES: CE(clean) + beta * KL(clean || adv)
    model.eval()
    with torch.no_grad():
        logits_clean = model(normalize(x)).detach()
    x_adv = x.detach() + 0.001 * torch.randn_like(x)
    x_adv = clamp_01(x_adv)

    for _ in range(int(steps)):
        x_adv.requires_grad_(True)
        logits_adv = model(normalize(x_adv))
        loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(logits_clean, dim=1), reduction="batchmean")
        grad = torch.autograd.grad(loss_kl, x_adv, only_inputs=True)[0]
        x_adv = x_adv.detach() + float(step_size) * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x - float(epsilon)), x + float(epsilon))
        x_adv = clamp_01(x_adv)

    model.train()
    logits_clean = model(normalize(x))
    logits_adv = model(normalize(x_adv))
    loss_ce = F.cross_entropy(logits_clean, y)
    loss_kl = F.kl_div(F.log_softmax(logits_adv, dim=1), F.softmax(logits_clean.detach(), dim=1), reduction="batchmean")
    loss = loss_ce + float(beta) * loss_kl
    return BatchOutput(loss=loss, logs={"acc": top1_accuracy(logits_clean, y), "trades_ce": float(loss_ce.item())})


def ccoblo_per_family_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    families: list[str],
    delta: float,
    kappa: float,
    mu: float,
    theta: float,
    eps: float,
    inner_steps: int,
    severity_min: int,
    severity_max: int,
    alpha_max: float = 20.0,
    denom_floor: float = 1e-3,
    proxy_scale: float = 1.0,
    proxy_reduction: str = "sum",
    ce_weight: float = 1.0,
    robust_ce_weight: float = 0.0,
    robust_ce_mode: str = "mean",
    robust_ce_alpha: float = 0.25,
    smooth_gap_weight: float = 0.0,
    smooth_gap_tau: float = 0.25,
) -> BatchOutput:
    """
    Implements the CCO-BLO update direction for per-family chance constraints.

    Outer objective = CE(clean) + penalty on quantiles.
    We implement the penalty gradient via an implicit-diff proxy:
      grad penalty_j = -(s_pos / (mu * d2G_ds2)) * grad_theta(dG_ds)
    by adding the pseudo-loss:  -(s_pos / (mu * d2G_ds2)).detach() * dG_ds
    """
    # Inputs to this method are in [0,1] (unnormalized), to support corruptions / adversarial steps.
    logits_clean = model(normalize(x))
    ce = F.cross_entropy(logits_clean, y)

    proxy_terms: list[torch.Tensor] = []
    logs: dict[str, Any] = {"acc_clean": top1_accuracy(logits_clean, y), "ce": float(ce.item())}

    # Build a single concatenated corrupted batch for one forward pass.
    x_list: list[torch.Tensor] = []
    for j, fam in enumerate(families):
        sev = sample_severity(severity_min, severity_max)
        xj = corrupt_batch(normalize(x), fam, severity=sev)  # returns [0,1]
        x_list.append(normalize(xj))
        logs[f"severity_{fam}"] = int(sev)

    x_cor = torch.cat(x_list, dim=0)
    y_rep = y.repeat(len(families))
    logits_cor = model(x_cor)
    viol = violation_from_logits(logits_cor, y_rep, kappa=float(kappa))
    viol = viol.view(len(families), x.size(0))

    per_ex_cor_ce = F.cross_entropy(logits_cor, y_rep, reduction="none").view(len(families), x.size(0))
    per_family_cor_ce = per_ex_cor_ce.mean(dim=1)
    robust_mode = str(robust_ce_mode).lower()
    if robust_mode == "max":
        robust_ce = per_family_cor_ce.max()
    elif robust_mode == "cvar":
        k = max(1, int(round(float(robust_ce_alpha) * len(families))))
        robust_ce = torch.topk(per_family_cor_ce, k=k, largest=True).values.mean()
    else:
        robust_ce = per_family_cor_ce.mean()

    gap_terms: list[torch.Tensor] = []
    target_cov = 1.0 - float(delta)
    tau = max(float(smooth_gap_tau), 1e-4)
    for j, fam in enumerate(families):
        gvals = viol[j]  # (B,)
        cov_soft = torch.sigmoid(-gvals / tau).mean()
        gap = torch.relu(gvals.new_tensor(target_cov) - cov_soft)
        gap_terms.append(gap.square())
        logs[f"cov_soft_{fam}"] = float(cov_soft.detach().item())
        logs[f"gap_{fam}"] = float(gap.detach().item())

        s_star = solve_s_star_bisect(gvals, delta=float(delta), theta=float(theta), eps=float(eps), steps=int(inner_steps))
        s_pos = max(0.0, float(s_star))
        if s_pos <= 0.0:
            logs[f"s_{fam}"] = float(s_star)
            continue

        # dG/ds as a scalar tensor with gradient through gvals (and thus model params).
        m = float(gvals.numel())
        t = gvals - float(s_star)
        dGds = 1.0 - (1.0 / (float(delta) * m)) * dh_theta(t, float(theta)).sum() + float(eps) * float(s_star)
        dGds = torch.nan_to_num(dGds, nan=0.0, posinf=0.0, neginf=0.0)

        denom_raw = d2G_ds2(gvals, delta=float(delta), theta=float(theta), eps=float(eps), s_star=float(s_star))
        if not math.isfinite(denom_raw):
            logs[f"s_{fam}"] = float(s_star)
            logs[f"alpha_{fam}"] = 0.0
            logs[f"denom_{fam}"] = float("nan")
            continue

        denom = max(float(denom_floor), float(denom_raw))
        mu_safe = max(float(mu), 1e-8)
        alpha = float(s_pos) / (mu_safe * denom)
        if float(alpha_max) > 0.0:
            alpha = min(alpha, float(alpha_max))
        if not math.isfinite(alpha) or alpha <= 0.0:
            logs[f"s_{fam}"] = float(s_star)
            logs[f"alpha_{fam}"] = 0.0
            logs[f"denom_{fam}"] = float(denom)
            continue

        proxy_terms.append(float(alpha) * dGds)

        logs[f"s_{fam}"] = float(s_star)
        logs[f"alpha_{fam}"] = float(alpha)
        logs[f"denom_{fam}"] = float(denom)

    smooth_gap = ce.new_zeros(())
    if gap_terms:
        smooth_gap = torch.stack(gap_terms).mean()

    loss = float(ce_weight) * ce + float(robust_ce_weight) * robust_ce + float(smooth_gap_weight) * smooth_gap
    proxy = ce.new_zeros(())
    if proxy_terms:
        stacked = torch.stack(proxy_terms)
        if str(proxy_reduction).lower() == "mean":
            proxy = stacked.mean()
        else:
            proxy = stacked.sum()
        loss = loss - float(proxy_scale) * proxy

    logs["proxy"] = float(proxy.detach().item())
    logs["smooth_gap"] = float(smooth_gap.detach().item())
    logs["robust_ce"] = float(robust_ce.detach().item())
    logs["proxy_scale"] = float(proxy_scale)
    return BatchOutput(loss=loss, logs=logs)
