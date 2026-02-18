from __future__ import annotations

import argparse
from pathlib import Path

from imagenet100_robust.core.config import Config, load_config
from imagenet100_robust.eval.eval_loop import evaluate_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory under runs/imagenet100_robust/...")
    ap.add_argument("--data-root", required=True, help="Path to ImageNet-100 root (train/ + val/).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir / "config.yaml").with_data_root(Path(args.data_root))
    evaluate_run(cfg, run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

