"""Prepare a lightweight integrated IQA MVP simulation environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mvp_environment_simulation"))
    parser.add_argument("--deploy-dir", type=Path, default=Path("deploy/mvp_simulation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.deploy_dir.mkdir(parents=True, exist_ok=True)
    service_map = {
        "services": [
            {"name": "iqa_api", "role": "FastAPI contracts"},
            {"name": "iqa_metadata_store", "role": "PostgreSQL metadata store"},
            {"name": "iqa_artifact_store", "role": "MinIO object store"},
        ]
    }
    (args.output_dir / "service_map.json").write_text(json.dumps(service_map, indent=2), encoding="utf-8")
    print(f"Prepared simulation skeleton: {args.output_dir}")


if __name__ == "__main__":
    main()
