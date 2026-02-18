from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs/imagenet100_robust", type=Path)
    ap.add_argument("--pattern", default="imagenet100_pilot_", help="Only include run dirs whose name starts with this.")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    rows = []
    for d in sorted(runs_root.iterdir()):
        if not d.is_dir():
            continue
        if args.pattern and not d.name.startswith(args.pattern):
            continue
        p = d / "eval.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        rows.append((d.name, data))

    if not rows:
        print("no eval.json found")
        return 0

    # Header
    families = ["noise", "blur", "weather", "digital"]
    cols = ["run", "clean_acc"] + [f"{f}_acc" for f in families] + [f"{f}_ec" for f in families] + [
        f"{f}_f_ec" for f in families
    ]
    print(",".join(cols))
    for name, d in rows:
        vals = [name, f"{d.get('clean_acc', float('nan')):.4f}"]
        for f in families:
            vals.append(f"{d.get(f'{f}_acc', float('nan')):.4f}")
        for f in families:
            vals.append(f"{d.get(f'{f}_ec', float('nan')):.4f}")
        for f in families:
            vals.append(f"{d.get(f'{f}_f_ec', float('nan')):.4f}")
        print(",".join(vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

