from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _seed_list(spec: dict[str, Any]) -> list[int]:
    if "seeds" in spec:
        return list(map(int, spec["seeds"]))
    start = int(spec.get("seed_start", 0))
    count = int(spec.get("seed_count", 10))
    return list(range(start, start + count))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    seed: int
    f_final: float
    penalty_final: float
    ec_final: float
    s_final: float
    wall_s: float


def _run_case_seed(case_id: str, seed: int, solver_cfg: dict[str, Any]) -> CaseResult:
    # Local import so the script can run even before editable install.
    import sys

    sys.path.insert(0, str(REPO_ROOT / "code"))
    from cco_blo.solver import run_ccoblo  # noqa: E402

    t0 = time.perf_counter()
    res = run_ccoblo(seed=seed, **solver_cfg)
    wall = time.perf_counter() - t0

    f_final = float(res.history["f"][-1])
    pen_final = float(res.history["penalty"][-1])
    ec_final = float(res.history["ec"][-1])
    s_final = float(res.history["s"][-1])

    return CaseResult(
        case_id=case_id,
        seed=seed,
        f_final=f_final,
        penalty_final=pen_final,
        ec_final=ec_final,
        s_final=s_final,
        wall_s=float(wall),
    )


def _write_rows_csv(path: Path, rows: list[CaseResult]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "seed", "f_final", "penalty_final", "ec_final", "s_final", "wall_s"])
        for r in rows:
            w.writerow([r.case_id, r.seed, r.f_final, r.penalty_final, r.ec_final, r.s_final, r.wall_s])


def _summarize(rows: list[CaseResult]) -> list[dict[str, Any]]:
    by_case: dict[str, list[CaseResult]] = {}
    for r in rows:
        by_case.setdefault(r.case_id, []).append(r)

    summary: list[dict[str, Any]] = []
    for case_id, rs in sorted(by_case.items()):
        f = np.array([r.f_final for r in rs], dtype=float)
        pen = np.array([r.penalty_final for r in rs], dtype=float)
        ec = np.array([r.ec_final for r in rs], dtype=float)
        wall = np.array([r.wall_s for r in rs], dtype=float)
        s = np.array([r.s_final for r in rs], dtype=float)
        summary.append(
            {
                "case_id": case_id,
                "n_seeds": int(len(rs)),
                "f_mean": float(f.mean()),
                "f_std": float(f.std(ddof=1)) if len(rs) > 1 else 0.0,
                "pen_mean": float(pen.mean()),
                "ec_mean": float(ec.mean()),
                "ec_std": float(ec.std(ddof=1)) if len(rs) > 1 else 0.0,
                "s_mean": float(s.mean()),
                "wall_mean_s": float(wall.mean()),
            }
        )
    return summary


def _write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    _ensure_dir(path.parent)
    keys = ["case_id", "n_seeds", "f_mean", "f_std", "pen_mean", "ec_mean", "ec_std", "s_mean", "wall_mean_s"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in summary:
            w.writerow([row[k] for k in keys])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Sweep YAML (under configs/).")
    ap.add_argument("--out", default="", help="Override output dir name under runs/.")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    sweep = cfg["sweep"]

    run_name = args.out or sweep.get("run_name") or f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = REPO_ROOT / "runs" / run_name
    _ensure_dir(outdir)
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    all_rows: list[CaseResult] = []
    for case in sweep["cases"]:
        case_id = str(case["id"])
        seeds = _seed_list(case)
        solver_cfg = dict(case["solver"])
        for seed in seeds:
            r = _run_case_seed(case_id, seed, solver_cfg)
            all_rows.append(r)
            print(f"[{case_id}] seed={seed} f={r.f_final:.4g} ec={r.ec_final:.3f} s={r.s_final:.4g} wall={r.wall_s:.2f}s")

    _write_rows_csv(outdir / "rows.csv", all_rows)
    summary = _summarize(all_rows)
    _write_summary_csv(outdir / "summary.csv", summary)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

