"""Build a Casting image inventory CSV from a restored dataset tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/casting_images_inventory.csv"))
    return parser.parse_args()


def iter_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    rows = []
    if args.source_dir.exists():
        for image_path in iter_images(args.source_dir):
            rows.append(
                {
                    "image_id": f"iqa_casting_{sha256_file(image_path)[:16]}",
                    "relative_path": image_path.relative_to(args.source_dir).as_posix(),
                    "sha256": sha256_file(image_path),
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_id", "relative_path", "sha256"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
