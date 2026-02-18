from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path, *, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if link:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    else:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagenet-root", required=True, type=Path, help="Path to ImageNet root.")
    ap.add_argument("--out-root", required=True, type=Path, help="Output path for ImageNet-100.")
    ap.add_argument("--synsets", required=True, type=Path, help="Text file of 100 synsets.")
    ap.add_argument("--link", action="store_true", help="Symlink instead of copying.")
    args = ap.parse_args()

    synsets = [s.strip() for s in args.synsets.read_text(encoding="utf-8").splitlines() if s.strip() and not s.startswith("#")]
    if len(synsets) != 100:
        raise SystemExit(f"expected 100 synsets, got {len(synsets)}")

    for split in ["train", "val"]:
        for syn in synsets:
            src = args.imagenet_root / split / syn
            dst = args.out_root / split / syn
            if not src.exists():
                raise SystemExit(f"missing {src}")
            _link_or_copy(src, dst, link=bool(args.link))

    print(f"wrote ImageNet-100 at {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

