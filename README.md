# CCO with BLO (UAI draft)

This repo contains:
- `paper/`: UAI LaTeX source.
- `code/cco_blo/`: reference implementation for synthetic CCO+BLO experiments.
- `scripts/`: runnable entrypoints.
- `configs/`: experiment configs (YAML).
- `runs/`: outputs (ignored by git).

## Quickstart (synthetic)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_synthetic.py --config configs/synth_example1.yaml
python scripts/run_synthetic.py --config configs/synth_gaussian_d10_m100.yaml
```

Outputs (CSV + PNG) go under `runs/`.

## Server workflow

1. Push changes from your laptop.
2. On the server:

```bash
git pull
source .venv/bin/activate  # or conda activate ...
python scripts/run_synthetic.py --config configs/synth_gaussian_d30_m100.yaml
```

