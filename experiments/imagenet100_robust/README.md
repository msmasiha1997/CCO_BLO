# ImageNet-100 Robustness (Data-Driven) Experiment

This folder contains a self-contained PyTorch experiment for **robust image classification under random corruptions** with:

- Our method: **CCO-BLO (per-corruption chance constraints)**.
- Practical baselines: **ERM**, **corruption augmentation**, **AugMix**, **TRADES**, **CVaR**, **GroupDRO**.

The code is designed to run on EPFL/SCITAS Slurm (see `jobs/`), but can also be run locally.

## Data layout

The training code expects ImageFolder-style directories:

```
<IMAGENET100_ROOT>/
  train/
    <class_0>/*.JPEG
    ...
  val/
    <class_0>/*.JPEG
    ...
```

You can create ImageNet-100 by selecting 100 ImageNet synsets and symlinking/copying them.

- Put the list of 100 synsets in `data/imagenet100_synsets.txt` (one synset per line).
- Then run:
  - `python experiments/imagenet100_robust/tools/make_imagenet100.py --imagenet-root /path/to/imagenet --out-root /path/to/imagenet100 --synsets experiments/imagenet100_robust/data/imagenet100_synsets.txt --link`

## Environment

This experiment needs PyTorch and Pillow.

On the cluster, the provided jobs assume:
- `module load gcc/13.2.0 python/3.11.7`
- an existing venv at `.venv311` with `torch` installed.

Optional (recommended) packages:
- `tqdm`, `pyyaml`

## Run training

Before launching many jobs, precompute dataset indices (avoids repeated filesystem scans):

```
python experiments/imagenet100_robust/tools/build_index.py --data-root /path/to/imagenet100
```

Example (CCO-BLO, per-family constraints):

```
python experiments/imagenet100_robust/train.py \
  --config experiments/imagenet100_robust/configs/core_ccoblo.yaml \
  --data-root /path/to/imagenet100
```

Baselines:

```
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/erm.yaml --data-root /path/to/imagenet100
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/augmix.yaml --data-root /path/to/imagenet100
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/trades.yaml --data-root /path/to/imagenet100
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/cvar.yaml --data-root /path/to/imagenet100
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/groupdro.yaml --data-root /path/to/imagenet100
```

Quick sanity configs (small scale, a few minutes):

```
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/sanity_erm.yaml --data-root /path/to/imagenet100
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/sanity_augmix.yaml --data-root /path/to/imagenet100
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/sanity_trades.yaml --data-root /path/to/imagenet100
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/sanity_cvar.yaml --data-root /path/to/imagenet100
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/sanity_groupdro.yaml --data-root /path/to/imagenet100
python experiments/imagenet100_robust/train.py --config experiments/imagenet100_robust/configs/sanity_ccoblo.yaml --data-root /path/to/imagenet100
```

Outputs are written under `runs/imagenet100_robust/<run_name>/`:
- `config.yaml`, `metrics.jsonl`, `best.pt`, `last.pt`

Useful train config knobs:
- `train.model_name`: `resnet50` (default) or `resnet18` for faster sanity runs.
- `train.max_train_batches`, `train.max_eval_batches`: limit batches per epoch for quick checks.
- `train.grad_clip_norm`: optional gradient clipping.
- `train.init_checkpoint`: optional warm-start path (e.g., ERM checkpoint).

CCO-BLO method knobs (optional):
- `ce_weight`, `robust_ce_weight`, `robust_ce_mode`: blend clean and corrupted CE.
- `smooth_gap_weight`, `smooth_gap_tau`: smooth chance-coverage penalty.
- `proxy_scale`, `alpha_max`, `denom_floor`, `warmup_epochs`: stabilize implicit proxy term.

## Run evaluation

```
python experiments/imagenet100_robust/eval.py --run-dir runs/imagenet100_robust/<run_name> --data-root /path/to/imagenet100
```

This reports:
- Clean Top-1 accuracy
- Robust Top-1 accuracy under the training corruption distribution
- Per-family empirical coverage (chance constraint satisfaction) + F-score
