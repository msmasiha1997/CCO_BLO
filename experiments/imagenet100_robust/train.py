from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from imagenet100_robust.core.config import Config, dump_config, load_config
from imagenet100_robust.core.io import ensure_dir, write_jsonl
from imagenet100_robust.core.seed import set_seed
from imagenet100_robust.run.train_loop import TrainArtifacts, train


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config.")
    ap.add_argument("--data-root", required=True, help="Path to ImageNet-100 root (train/ + val/).")
    ap.add_argument("--out-root", default="runs/imagenet100_robust", help="Base output directory.")
    ap.add_argument("--run-name", default="", help="Override run name.")
    args = ap.parse_args()

    cfg: Config = load_config(Path(args.config))
    cfg = cfg.with_data_root(Path(args.data_root))

    run_name = args.run_name or cfg.run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = Path(args.out_root) / run_name
    ensure_dir(outdir)

    set_seed(cfg.seed)
    (outdir / "config.yaml").write_text(yaml.safe_dump(dump_config(cfg), sort_keys=False), encoding="utf-8")

    artifacts: TrainArtifacts = train(cfg, outdir)
    (outdir / "artifacts.json").write_text(json.dumps(asdict(artifacts), indent=2), encoding="utf-8")
    write_jsonl(outdir / "metrics.jsonl", artifacts.metrics)
    print(f"wrote {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
