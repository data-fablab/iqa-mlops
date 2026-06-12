"""Simulate the IQA MVP lifecycle when replay CSVs are available."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mvp_lifecycle_simulation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "mvp_lifecycle_simulation_report.md"
    report.write_text(
        "# IQA MVP lifecycle simulation\n\n"
        "Simulation placeholder restored after file loss. Regenerate after CSV recovery.\n",
        encoding="utf-8",
    )
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
