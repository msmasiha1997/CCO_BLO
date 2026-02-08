from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "code"))

from cco_blo.solver import run_ccoblo  # noqa: E402


def _save_csv(path: Path, history: dict[str, list[float]]) -> None:
    keys = list(history.keys())
    rows = zip(*(history[k] for k in keys))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step"] + keys)
        for i, row in enumerate(rows):
            w.writerow([i] + list(row))


def _plot(history: dict[str, list[float]], outdir: Path) -> None:
    steps = np.arange(len(next(iter(history.values()))))
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(steps, history["f"], label="f(x)")
    ax[0].plot(steps, history["penalty"], label="penalty")
    ax[0].set_title("Objective terms")
    ax[0].set_xlabel("outer step")
    ax[0].legend()

    ax[1].plot(steps, history["ec"], label="empirical coverage")
    ax[1].set_title("Coverage")
    ax[1].set_xlabel("outer step")
    ax[1].set_ylim(0.0, 1.0)
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(outdir / "curves.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config under configs/.")
    ap.add_argument("--out", default="", help="Override output directory name under runs/.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = (REPO_ROOT / cfg_path).resolve()

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    run_name = args.out or cfg.get("run_name") or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = REPO_ROOT / "runs" / run_name

    res = run_ccoblo(**cfg["solver"])
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (outdir / "result.json").write_text(
        json.dumps({"x": res.x.tolist(), "s": res.s, "history_keys": list(res.history.keys())}, indent=2),
        encoding="utf-8",
    )
    _save_csv(outdir / "history.csv", res.history)
    _plot(res.history, outdir)
    print(f"wrote {outdir}")
    print(f"x={res.x}, s={res.s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
