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

## Sweeps (synthetic)

```bash
python scripts/sweep_synthetic.py --config configs/sweep_replicate_slides.yaml
python scripts/sweep_synthetic.py --config configs/sweep_scaling_gaussian_linear.yaml
```

Each sweep writes `rows.csv` and `summary.csv` under `runs/<run_name>/`.

## Baseline benchmark (synthetic)

Install baseline deps:

```bash
pip install -r requirements-baselines.txt
```

Run the multi-method benchmark (Core/TACO/CPP-*):

```bash
python scripts/benchmark_synthetic_baselines.py --config configs/benchmark_synthetic_baselines.yaml
python scripts/benchmark_synthetic_baselines.py --config configs/benchmark_synthetic_sweeps.yaml
```

## Server workflow

1. Push changes from your laptop.
2. On the server:

```bash
git pull
source .venv/bin/activate  # or conda activate ...
python scripts/run_synthetic.py --config configs/synth_gaussian_d30_m100.yaml
```
