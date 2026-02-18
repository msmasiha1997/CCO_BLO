from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    data_root: Path
    num_classes: int = 100
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 90
    model_name: str = "resnet50"
    optimizer: str = "sgd"
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    amp: bool = True
    grad_clip_norm: float = 0.0
    max_train_batches: int = 0
    max_eval_batches: int = 0
    init_checkpoint: str = ""


@dataclass(frozen=True)
class MethodConfig:
    name: str
    params: dict[str, Any]


@dataclass(frozen=True)
class CorruptionConfig:
    severity_min: int = 1
    severity_max: int = 5


@dataclass(frozen=True)
class Config:
    run_name: str
    seed: int
    data: DataConfig
    train: TrainConfig
    method: MethodConfig
    corruptions: CorruptionConfig

    def with_data_root(self, data_root: Path) -> "Config":
        return replace(self, data=replace(self.data, data_root=data_root))


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def load_config(path: Path) -> Config:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = cfg.get("data") or {}
    train = cfg.get("train") or {}
    method = cfg.get("method") or {}
    corruptions = cfg.get("corruptions") or {}

    # data_root is provided at runtime via CLI; keep placeholder here.
    data_root = _as_path(data.get("data_root") or ".")

    return Config(
        run_name=str(cfg.get("run_name") or ""),
        seed=int(cfg.get("seed", 0)),
        data=DataConfig(
            data_root=data_root,
            num_classes=int(data.get("num_classes", 100)),
            image_size=int(data.get("image_size", 224)),
            batch_size=int(data.get("batch_size", 64)),
            num_workers=int(data.get("num_workers", 4)),
        ),
        train=TrainConfig(
            epochs=int(train.get("epochs", 90)),
            model_name=str(train.get("model_name", "resnet50")),
            optimizer=str(train.get("optimizer", "sgd")),
            lr=float(train.get("lr", 0.1)),
            momentum=float(train.get("momentum", 0.9)),
            weight_decay=float(train.get("weight_decay", 1e-4)),
            amp=bool(train.get("amp", True)),
            grad_clip_norm=float(train.get("grad_clip_norm", 0.0)),
            max_train_batches=int(train.get("max_train_batches", 0)),
            max_eval_batches=int(train.get("max_eval_batches", 0)),
            init_checkpoint=str(train.get("init_checkpoint", "")),
        ),
        method=MethodConfig(
            name=str(method.get("name") or "erm"),
            params=dict(method.get("params") or {}),
        ),
        corruptions=CorruptionConfig(
            severity_min=int(corruptions.get("severity_min", 1)),
            severity_max=int(corruptions.get("severity_max", 5)),
        ),
    )


def _yamlable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _yamlable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yamlable(v) for v in obj]
    return obj


def dump_config(cfg: Config) -> dict[str, Any]:
    # Avoid PyYAML representer issues with Path objects.
    return _yamlable(
        {
            "run_name": cfg.run_name,
            "seed": cfg.seed,
            "data": {
                "data_root": cfg.data.data_root,
                "num_classes": cfg.data.num_classes,
                "image_size": cfg.data.image_size,
                "batch_size": cfg.data.batch_size,
                "num_workers": cfg.data.num_workers,
            },
            "train": {
                "epochs": cfg.train.epochs,
                "model_name": cfg.train.model_name,
                "optimizer": cfg.train.optimizer,
                "lr": cfg.train.lr,
                "momentum": cfg.train.momentum,
                "weight_decay": cfg.train.weight_decay,
                "amp": cfg.train.amp,
                "grad_clip_norm": cfg.train.grad_clip_norm,
                "max_train_batches": cfg.train.max_train_batches,
                "max_eval_batches": cfg.train.max_eval_batches,
                "init_checkpoint": cfg.train.init_checkpoint,
            },
            "method": {"name": cfg.method.name, "params": cfg.method.params},
            "corruptions": {
                "severity_min": cfg.corruptions.severity_min,
                "severity_max": cfg.corruptions.severity_max,
            },
        }
    )
