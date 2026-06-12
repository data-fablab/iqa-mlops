"""Run the IQA lifecycle batch boundary."""

from __future__ import annotations

import argparse
import json

from iqa.registry import ModelRegistryRef, registered_model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-id", default="production_replay_natural")
    parser.add_argument("--stage", default="candidate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "service": "iqa-trainer",
        "lifecycle": "train_eval_gate_promote",
        "registry": ModelRegistryRef(
            scenario_id=args.scenario_id,
            registered_model_name=registered_model_name(args.scenario_id),
            stage=args.stage,
        ).to_dict(),
        "status": "planned",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
