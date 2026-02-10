# Synthetic results (tracked)

`runs/` is git-ignored, so after running sweeps on the server copy the small artifacts needed for the paper into this folder and commit them.

Recommended per-run contents:
- `summary.csv`
- `summary.json`
- `config.yaml`

Then regenerate LaTeX tables for the paper using `scripts/make_latex_tables.py`.

