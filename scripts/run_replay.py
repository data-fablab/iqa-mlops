"""Run the IQA replay batch boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from iqa.replay import list_replay_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--plan", type=Path, default=Path("data/metadata/casting_flux_replay_plan_natural.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "service": "iqa-replay",
        "scenario_id": args.scenario_id,
        "plan": str(args.plan),
        "known_scenarios": list_replay_scenarios(),
        "status": "planned",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
