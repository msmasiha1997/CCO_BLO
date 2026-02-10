from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SummaryRow:
    case_id: str
    method: str
    n_ok: int
    n_total: int
    objective_mean: float
    objective_std: float
    ec_mean: float
    ec_std: float
    fscore_ec_mean: float
    fscore_ec_std: float
    wall_mean_s: float


def _read_summary_csv(path: Path) -> list[SummaryRow]:
    rows: list[SummaryRow] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                SummaryRow(
                    case_id=str(row["case_id"]),
                    method=str(row["method"]),
                    n_ok=int(row.get("n_ok", 0) or 0),
                    n_total=int(row.get("n_total", 0) or 0),
                    objective_mean=float(row.get("objective_mean", "nan")),
                    objective_std=float(row.get("objective_std", "nan")),
                    ec_mean=float(row.get("ec_mean", "nan")),
                    ec_std=float(row.get("ec_std", "nan")),
                    fscore_ec_mean=float(row.get("fscore_ec_mean", "nan")),
                    fscore_ec_std=float(row.get("fscore_ec_std", "nan")),
                    wall_mean_s=float(row.get("wall_mean_s", "nan")),
                )
            )
    return rows


def _fmt(x: float) -> str:
    if x != x:  # nan
        return "--"
    ax = abs(x)
    if ax != 0.0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.2e}"
    return f"{x:.4g}"


def _escape_tex(s: str) -> str:
    return s.replace("_", "\\_")


def _write_case_table(
    out: list[str],
    *,
    case_id: str,
    rows: list[SummaryRow],
    methods_order: list[str] | None,
    caption_prefix: str,
    label_prefix: str,
) -> None:
    out.append("\\begin{table}[!t]")
    out.append("  \\centering")
    out.append(f"  \\caption{{{caption_prefix} { _escape_tex(case_id) }.}}\\label{{{label_prefix}:{_escape_tex(case_id)}}}")
    out.append("  \\begin{tabular}{lrrrr}")
    out.append("    \\toprule")
    out.append("    Method & Obj.$\\downarrow$ & EC$\\uparrow$ & $F_{\\mathrm{EC}}\\uparrow$ & Time$\\downarrow$\\\\")
    out.append("    \\midrule")

    by_method: dict[str, SummaryRow] = {r.method: r for r in rows}
    methods = methods_order or sorted(by_method.keys())
    for m in methods:
        r = by_method.get(m)
        if r is None:
            out.append(f"    {_escape_tex(m)} & -- & -- & -- & --\\\\")
            continue
        suffix = "" if r.n_total <= 0 else f" ({r.n_ok}/{r.n_total})"
        out.append(
            "    "
            + " & ".join(
                [
                    _escape_tex(m) + suffix,
                    _fmt(r.objective_mean),
                    _fmt(r.ec_mean),
                    _fmt(r.fscore_ec_mean),
                    _fmt(r.wall_mean_s),
                ]
            )
            + "\\\\"
        )
    out.append("    \\bottomrule")
    out.append("  \\end{tabular}")
    out.append("\\end{table}")
    out.append("")


def _group_case_ids(case_ids: list[str]) -> list[tuple[str, list[str]]]:
    groups: dict[str, list[str]] = {"CPP": [], "HD": [], "HT": [], "Other": []}
    for cid in case_ids:
        if cid.startswith("cpp") or cid.startswith("example") or cid.startswith("cpp1"):
            groups["CPP"].append(cid)
        elif cid.startswith("hd") or cid.startswith("gaussian") or cid.startswith("high"):
            groups["HD"].append(cid)
        elif cid.startswith("ht") or cid.startswith("heavy"):
            groups["HT"].append(cid)
        else:
            groups["Other"].append(cid)
    ordered: list[tuple[str, list[str]]] = []
    for k in ["CPP", "HD", "HT", "Other"]:
        if groups[k]:
            ordered.append((k, sorted(groups[k])))
    return ordered


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary",
        default=str(REPO_ROOT / "results" / "synthetic" / "latest" / "summary.csv"),
        help="Path to benchmark summary.csv (from scripts/benchmark_synthetic_baselines.py).",
    )
    ap.add_argument(
        "--out",
        default=str(REPO_ROOT / "paper" / "uai2026-template" / "tables" / "synthetic_results.tex"),
        help="Output .tex file to write (overwrites).",
    )
    ap.add_argument(
        "--methods",
        default="Core,TACO,CPP-MIP,CPP-KKT,SA,SAA,KKTBL,RCPP-MIP,RCPP-KKT",
        help="Comma-separated method order for each table.",
    )
    ap.add_argument(
        "--cases",
        default="",
        help="Optional comma-separated case_id filter. If empty, include all cases.",
    )
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_path = Path(args.out)
    methods_order = [m.strip() for m in str(args.methods).split(",") if m.strip()]

    rows = _read_summary_csv(summary_path)
    if not rows:
        raise SystemExit(f"no rows in {summary_path}")

    if args.cases.strip():
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        rows = [r for r in rows if r.case_id in keep]

    by_case: dict[str, list[SummaryRow]] = {}
    for r in rows:
        by_case.setdefault(r.case_id, []).append(r)

    out_lines: list[str] = []
    out_lines.append("% This file is auto-generated by scripts/make_latex_tables.py")
    out_lines.append("% Do not edit by hand.")
    out_lines.append("")

    for group, case_ids in _group_case_ids(sorted(by_case.keys())):
        for cid in case_ids:
            _write_case_table(
                out_lines,
                case_id=cid,
                rows=by_case[cid],
                methods_order=methods_order,
                caption_prefix=f"Synthetic ({group}) case",
                label_prefix="tab:synthetic",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

