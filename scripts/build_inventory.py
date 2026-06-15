"""Build a Casting image inventory CSV from IQA piece events and image files."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

DEFAULT_EVENTS_MANIFEST = Path("data/metadata/casting_piece_events.csv")
DEFAULT_SOURCE_DIR = Path("data/raw/hss-iad")
DEFAULT_OUTPUT = Path("data/metadata/casting_images_inventory.csv")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-manifest", type=Path, default=DEFAULT_EVENTS_MANIFEST)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def iter_images(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_pipe(value: str | None) -> list[str]:
    return [part for part in (value or "").split("|") if part]


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _gt_mask_path(relative_path: str, is_defective: bool) -> str:
    if not is_defective:
        return ""
    path = Path(relative_path)
    parts = path.parts
    if len(parts) < 4:
        return ""
    source_class = parts[0]
    return str(Path(source_class) / "ground_truth" / "defective" / f"{path.stem}_mask.png").replace("\\", "/")


def _rows_from_events(events_manifest: Path, source_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with events_manifest.open(newline="", encoding="utf-8") as file:
        for event in csv.DictReader(file):
            image_ids = _split_pipe(event.get("image_ids"))
            relative_paths = _split_pipe(event.get("relative_paths"))
            for index, relative_path in enumerate(relative_paths):
                image_id = image_ids[index] if index < len(image_ids) else Path(relative_path).stem
                source_path = source_dir / relative_path
                gt_mask_relative_path = _gt_mask_path(relative_path, _truthy(event.get("is_defective")))
                gt_mask_source_path = source_dir / gt_mask_relative_path if gt_mask_relative_path else None
                rows.append(
                    {
                        "image_id": image_id,
                        "event_id": event.get("event_id", ""),
                        "source_class": event.get("source_class", ""),
                        "split_set": event.get("split_set", ""),
                        "label": event.get("label", ""),
                        "is_defective": str(_truthy(event.get("is_defective"))).lower(),
                        "relative_path": relative_path,
                        "source_path_exists": str(source_path.exists()).lower(),
                        "sha256": sha256_file(source_path) if source_path.exists() else "",
                        "has_gt_mask": str(bool(gt_mask_source_path and gt_mask_source_path.exists())).lower(),
                        "gt_mask_relative_path": gt_mask_relative_path,
                        "gt_mask_source_path_exists": str(bool(gt_mask_source_path and gt_mask_source_path.exists())).lower(),
                        "n_images_for_event": event.get("n_images", ""),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    if args.events_manifest.exists():
        rows = _rows_from_events(args.events_manifest, args.source_dir)
    else:
        rows = [
            {
                "image_id": f"iqa_casting_{sha256_file(image_path)[:16]}",
                "event_id": "",
                "source_class": "",
                "split_set": "",
                "label": "",
                "is_defective": "",
                "relative_path": image_path.relative_to(args.source_dir).as_posix(),
                "source_path_exists": "true",
                "sha256": sha256_file(image_path),
                "has_gt_mask": "",
                "gt_mask_relative_path": "",
                "gt_mask_source_path_exists": "",
                "n_images_for_event": "",
            }
            for image_path in iter_images(args.source_dir)
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [
            "image_id",
            "event_id",
            "source_class",
            "split_set",
            "label",
            "is_defective",
            "relative_path",
            "source_path_exists",
            "sha256",
            "has_gt_mask",
            "gt_mask_relative_path",
            "gt_mask_source_path_exists",
            "n_images_for_event",
        ])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
