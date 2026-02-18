from __future__ import annotations

import argparse
from pathlib import Path

import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from imagenet100_robust.data.dataset import ImageFolderDataset
from imagenet100_robust.data.transforms import TrainTransform, ValTransform


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path, help="ImageNet-100 root with train/ and val/.")
    ap.add_argument("--image-size", default=224, type=int)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    ImageFolderDataset(
        data_root / "train",
        transform=TrainTransform(int(args.image_size)),
        index_file=data_root / ".index_train.csv",
        write_index=True,
    )
    ImageFolderDataset(
        data_root / "val",
        transform=ValTransform(int(args.image_size)),
        index_file=data_root / ".index_val.csv",
        write_index=True,
    )

    print(f"wrote {data_root / '.index_train.csv'}")
    print(f"wrote {data_root / '.index_val.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
