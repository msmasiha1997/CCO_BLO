from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _fscore_ec(ec: float, target: float) -> float:
    # Symmetric "closeness to target" score in [0,1], equals 1 iff ec == target.
    if ec < 0 or target <= 0:
        return float("nan")
    return float(2.0 * min(ec, target) / (ec + target + 1e-12))


def _make_seed_list(spec: dict[str, Any]) -> list[int]:
    start = int(spec.get("start", 0))
    count = int(spec.get("count", 10))
    return list(range(start, start + count))


def _patch_fixed_sampling(problem: Any, z_train: np.ndarray, z_eval: np.ndarray) -> None:
    import types

    samples_num = int(problem.samples_num)

    def z_samples(self, n: int | None = None) -> np.ndarray:
        if n is None or n == -1 or int(n) == samples_num:
            return z_train
        n = int(n)
        if n <= len(z_eval):
            return z_eval[:n]
        reps = int(np.ceil(n / len(z_eval)))
        tiled = np.tile(z_eval, (reps, 1)) if z_eval.ndim == 2 else np.tile(z_eval, reps)
        return tiled[:n]

    problem.z_samples = types.MethodType(z_samples, problem)


def _make_problem(problem_cfg: dict[str, Any]) -> Any:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "code"))
    from cco.problems.cpp_problem import CPPProblem  # noqa: E402
    from cco.problems.high_dimension_problem import HighDimensionProblem  # noqa: E402

    kind = problem_cfg["kind"]
    if kind == "cpp_problem":
        return CPPProblem(
            initial_x=float(problem_cfg["initial_x"]),
            samples_num=int(problem_cfg["samples_num"]),
            max_iter_s=int(problem_cfg["max_iter_s"]),
            max_iter_x=int(problem_cfg["max_iter_x"]),
            theta=float(problem_cfg["theta"]),
            epsilon=float(problem_cfg["epsilon"]),
            delta=float(problem_cfg["delta"]),
            lr=float(problem_cfg["lr"]),
            mu=float(problem_cfg["mu"]),
            abstol=float(problem_cfg["abstol"]) if problem_cfg.get("abstol") is not None else None,
        )

    if kind == "high_dimension":
        dim = int(problem_cfg["dim"])
        init = problem_cfg["initial_x"]
        if init == "zeros":
            initial_x = np.zeros(dim, dtype=float)
        else:
            initial_x = np.array(init, dtype=float)
        return HighDimensionProblem(
            initial_x=initial_x,
            samples_num=int(problem_cfg["samples_num"]),
            max_iter_s=int(problem_cfg["max_iter_s"]),
            max_iter_x=int(problem_cfg["max_iter_x"]),
            theta=float(problem_cfg["theta"]),
            epsilon=float(problem_cfg["epsilon"]),
            delta=float(problem_cfg["delta"]),
            lr=float(problem_cfg["lr"]),
            mu=float(problem_cfg["mu"]),
            abstol=float(problem_cfg["abstol"]) if problem_cfg.get("abstol") is not None else None,
        )

    raise ValueError(f"unknown problem kind: {kind}")


def _run_core(problem: Any) -> tuple[np.ndarray, float]:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "code"))
    from cco.core.optimizer import Optimizer  # noqa: E402

    opt = Optimizer(problem=problem, zero_order_method=False, empirical_coverage=False, verbose=False)
    t0 = time.perf_counter()
    x_final = opt.run()
    wall = time.perf_counter() - t0
    x_final = np.asarray(x_final, dtype=float).reshape(problem.dimension)
    return x_final, float(wall)


def _run_taco(problem: Any) -> tuple[np.ndarray, float]:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "code"))
    from cco.taco.problems.toy_problem import ToyProblem  # noqa: E402
    from cco.taco.chance_optimizer import Optimizer as TacoOptimizer  # noqa: E402

    taco_problem = ToyProblem(problem=problem)
    pb_numba_compliant = True
    other_selected_params = {
        "nb_iterations": int(problem.max_iter_x),
        "bund_mu_high": 0.001,
        "bund_mu_low": 0.001,
        "bund_max_size_bundle_set": 31,
        "superquantile_smoothing_param": 0.1,
    }
    optimizer = TacoOptimizer(
        problem=taco_problem,
        p=1.0 - float(problem.delta),
        starting_point=np.concatenate([np.asarray(problem.initial_x, dtype=float).reshape(-1), [0.0]]),
        numba=pb_numba_compliant,
        params=other_selected_params,
    )
    t0 = time.perf_counter()
    result = optimizer.run(verbose=False)
    wall = time.perf_counter() - t0
    x_final = np.asarray(result[:-1], dtype=float).reshape(problem.dimension)
    return x_final, float(wall)


def _run_cpp(problem: Any, method: str, cpp_time_limit_s: int) -> tuple[np.ndarray | None, float, str | None]:
    import sys

    sys.path.insert(0, str(REPO_ROOT / "code"))
    import cco.cpp.configuration as cpp_config  # noqa: E402
    from cco.cpp.resources.solver import solve  # noqa: E402

    cpp_config.time_limit = int(cpp_time_limit_s)

    x_dim = problem.dimension[0]
    delta = float(problem.delta)
    training_Ys = problem.z_samples()

    # No extra deterministic constraints for synthetic problems.
    hs: list[Callable[..., Any]] = []
    gs: list[Callable[..., Any]] = []
    result, solver_time = solve(x_dim, delta, training_Ys, hs, gs, problem.cpp_chance_function, problem.cpp_objective_function, method)

    if isinstance(result, str):
        return None, float(solver_time), result
    x_final = np.asarray(result, dtype=float).reshape(problem.dimension)
    return x_final, float(solver_time), None


@dataclass(frozen=True)
class Row:
    case_id: str
    method: str
    seed: int
    objective: float
    ec: float
    fscore_ec: float
    wall_s: float
    status: str


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Benchmark YAML (under configs/).")
    ap.add_argument("--out", default="", help="Override output dir name under runs/.")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    bench = cfg["benchmark"]

    run_name = args.out or bench.get("run_name") or f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = REPO_ROOT / "runs" / run_name
    _ensure_dir(outdir)
    (outdir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    eval_samples = int(bench.get("eval_samples", 100000))
    cpp_time_limit_s = int(bench.get("cpp_time_limit_s", 600))
    methods = list(bench["methods"])

    rows: list[Row] = []

    for case in bench["cases"]:
        case_id = str(case["id"])
        problem_cfg = dict(case["problem"])
        target = 1.0 - float(problem_cfg["delta"])

        for seed in _make_seed_list(case["seeds"]):
            # Deterministic fixed samples for *all* methods in this (case, seed).
            np.random.seed(seed)
            problem = _make_problem(problem_cfg)
            z_train = problem.z_samples()
            np.random.seed(seed + 12345)
            z_eval = problem.z_samples(n=eval_samples)
            _patch_fixed_sampling(problem, z_train=z_train, z_eval=z_eval)

            for method in methods:
                status = "ok"
                try:
                    if method == "Core":
                        x_final, wall_s = _run_core(problem)
                    elif method == "TACO":
                        x_final, wall_s = _run_taco(problem)
                    elif method in ("CPP-MIP", "CPP-KKT", "SA", "SAA"):
                        x_final, wall_s, err = _run_cpp(problem, method, cpp_time_limit_s)
                        if x_final is None:
                            status = err or "error"
                            # write placeholder row with NaNs
                            rows.append(
                                Row(
                                    case_id=case_id,
                                    method=method,
                                    seed=seed,
                                    objective=float("nan"),
                                    ec=float("nan"),
                                    fscore_ec=float("nan"),
                                    wall_s=wall_s,
                                    status=status,
                                )
                            )
                            print(f"[{case_id}] {method} seed={seed} status={status} wall={wall_s:.2f}s")
                            continue
                    else:
                        raise ValueError(f"unknown method: {method}")

                    obj = float(problem.f_function(x_final))
                    chance_vals = problem.chance_function(x_final, z_eval)
                    ec = float(np.mean(chance_vals <= 0.0))
                    fscore = _fscore_ec(ec, target)

                    rows.append(
                        Row(
                            case_id=case_id,
                            method=method,
                            seed=seed,
                            objective=obj,
                            ec=ec,
                            fscore_ec=fscore,
                            wall_s=wall_s,
                            status=status,
                        )
                    )
                    print(f"[{case_id}] {method} seed={seed} obj={obj:.4g} ec={ec:.3f} f={fscore:.3f} wall={wall_s:.2f}s")
                except Exception as e:
                    status = f"exception:{type(e).__name__}"
                    rows.append(
                        Row(
                            case_id=case_id,
                            method=method,
                            seed=seed,
                            objective=float("nan"),
                            ec=float("nan"),
                            fscore_ec=float("nan"),
                            wall_s=0.0,
                            status=status,
                        )
                    )
                    print(f"[{case_id}] {method} seed={seed} status={status}")

    # Write rows.csv
    with (outdir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "method", "seed", "objective", "ec", "fscore_ec", "wall_s", "status"])
        for r in rows:
            w.writerow([r.case_id, r.method, r.seed, r.objective, r.ec, r.fscore_ec, r.wall_s, r.status])

    # Summarize
    summary: list[dict[str, Any]] = []
    for case_id in sorted({r.case_id for r in rows}):
        for method in methods:
            rs = [r for r in rows if r.case_id == case_id and r.method == method]
            ok = [r for r in rs if r.status == "ok" and np.isfinite(r.objective) and np.isfinite(r.ec)]
            if not ok:
                summary.append({"case_id": case_id, "method": method, "n_ok": 0, "n_total": len(rs)})
                continue
            obj = np.array([r.objective for r in ok], dtype=float)
            ec = np.array([r.ec for r in ok], dtype=float)
            fs = np.array([r.fscore_ec for r in ok], dtype=float)
            wall = np.array([r.wall_s for r in ok], dtype=float)
            summary.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "n_ok": int(len(ok)),
                    "n_total": int(len(rs)),
                    "objective_mean": float(obj.mean()),
                    "objective_std": float(obj.std(ddof=1)) if len(ok) > 1 else 0.0,
                    "ec_mean": float(ec.mean()),
                    "ec_std": float(ec.std(ddof=1)) if len(ok) > 1 else 0.0,
                    "fscore_ec_mean": float(fs.mean()),
                    "fscore_ec_std": float(fs.std(ddof=1)) if len(ok) > 1 else 0.0,
                    "wall_mean_s": float(wall.mean()),
                }
            )

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (outdir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "method",
                "n_ok",
                "n_total",
                "objective_mean",
                "objective_std",
                "ec_mean",
                "ec_std",
                "fscore_ec_mean",
                "fscore_ec_std",
                "wall_mean_s",
            ],
        )
        w.writeheader()
        for row in summary:
            w.writerow(row)

    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

