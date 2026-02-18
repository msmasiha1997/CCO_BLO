from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image
import torch
from torch.utils.data import Dataset


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    y: int


class ImageFolderDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        root: Path,
        *,
        transform: Callable[[Image.Image], torch.Tensor],
        index_file: Path | None = None,
        write_index: bool = True,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.index_file = Path(index_file) if index_file is not None else None

        classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        if not classes:
            raise ValueError(f"no class folders found under {self.root}")
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        if self.index_file is not None and self.index_file.exists():
            self.samples = self._load_index(self.index_file)
        else:
            self.samples = self._scan_samples(classes)
            if self.index_file is not None and write_index:
                self._write_index(self.index_file, self.samples)

        if not self.samples:
            raise ValueError(f"no images found under {self.root}")

    def _scan_samples(self, classes: list[str]) -> list[Sample]:
        # Avoid Path.rglob across the full tree; ImageNet folders are one level deep.
        samples: list[Sample] = []
        for cls in classes:
            cls_dir = self.root / cls
            with os.scandir(cls_dir) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    p = Path(entry.path)
                    if p.suffix.lower() in IMG_EXTS:
                        samples.append(Sample(path=p, y=self.class_to_idx[cls]))
        return samples

    def _write_index(self, path: Path, samples: list[Sample]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["relpath", "y"])
            for s in samples:
                rel = s.path.relative_to(self.root).as_posix()
                w.writerow([rel, int(s.y)])
        tmp.replace(path)

    def _load_index(self, path: Path) -> list[Sample]:
        out: list[Sample] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rel = str(row["relpath"])
                y = int(row["y"])
                out.append(Sample(path=self.root / rel, y=y))
        return out

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        img = Image.open(s.path).convert("RGB")
        x = self.transform(img)
        return x, int(s.y)
