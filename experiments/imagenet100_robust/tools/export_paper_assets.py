from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


FAMILIES = ["noise", "blur", "weather", "digital"]


@dataclass(frozen=True)
class RunSummary:
    label: str
    clean_acc: float
    fam_acc: dict[str, float]
    fam_f: dict[str, float]

    @property
    def mean_corrupt_acc(self) -> float:
        return float(np.mean([self.fam_acc[f] for f in FAMILIES]))

    @property
    def min_corrupt_acc(self) -> float:
        return float(np.min([self.fam_acc[f] for f in FAMILIES]))

    @property
    def mean_f(self) -> float:
        return float(np.mean([self.fam_f[f] for f in FAMILIES]))

    @property
    def min_f(self) -> float:
        return float(np.min([self.fam_f[f] for f in FAMILIES]))


def _load_eval(eval_path: Path, label: str) -> RunSummary:
    d = json.loads(eval_path.read_text(encoding="utf-8"))
    return RunSummary(
        label=label,
        clean_acc=float(d["clean_acc"]),
        fam_acc={f: float(d[f"{f}_acc"]) for f in FAMILIES},
        fam_f={f: float(d[f"{f}_f_ec"]) for f in FAMILIES},
    )


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def _write_table_tex(out_path: Path, runs: list[RunSummary]) -> None:
    lines = []
    for r in runs:
        lines.append(
            " & ".join(
                [
                    r.label,
                    _pct(r.clean_acc),
                    _pct(r.mean_corrupt_acc),
                    _pct(r.min_corrupt_acc),
                    _pct(r.mean_f),
                    _pct(r.min_f),
                ]
            )
            + r" \\"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_acc(out_path: Path, runs: list[RunSummary]) -> None:
    labels = [r.label for r in runs]
    clean = [r.clean_acc for r in runs]
    mean_cor = [r.mean_corrupt_acc for r in runs]

    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    ax.bar(x - w / 2, clean, width=w, label="Clean Top-1")
    ax.bar(x + w / 2, mean_cor, width=w, label="Mean corrupted Top-1 (4 families)")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_f_heatmap(out_path: Path, runs: list[RunSummary]) -> None:
    mat = np.array([[r.fam_f[f] for f in FAMILIES] for r in runs], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(FAMILIES)))
    ax.set_xticklabels([f.title() for f in FAMILIES])
    ax.set_yticks(np.arange(len(runs)))
    ax.set_yticklabels([r.label for r in runs])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$F_{\mathrm{EC}}$")
    ax.set_title(r"Reliability vs target coverage ($F_{\mathrm{EC}}$)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs/imagenet100_robust", type=Path)
    ap.add_argument("--paper-root", default="paper/uai2026-template", type=Path)
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run spec 'label=dir_name' (dir under runs-root). Can be passed multiple times.",
    )
    args = ap.parse_args()

    if not args.run:
        raise SystemExit("provide at least one --run label=dir_name")

    runs: list[RunSummary] = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"bad --run: {spec!r} (expected label=dir_name)")
        label, dname = spec.split("=", 1)
        eval_path = Path(args.runs_root) / dname / "eval.json"
        if not eval_path.exists():
            raise SystemExit(f"missing {eval_path}")
        runs.append(_load_eval(eval_path, label=label))

    plots_dir = Path(args.paper_root) / "Plots"
    tables_dir = Path(args.paper_root) / "tables"
    _plot_acc(plots_dir / "imagenet100_pilot_acc.png", runs)
    _plot_f_heatmap(plots_dir / "imagenet100_pilot_fec_heatmap.png", runs)
    _write_table_tex(tables_dir / "imagenet100_pilot_rows.tex", runs)

    print("wrote:")
    print(plots_dir / "imagenet100_pilot_acc.png")
    print(plots_dir / "imagenet100_pilot_fec_heatmap.png")
    print(tables_dir / "imagenet100_pilot_rows.tex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
