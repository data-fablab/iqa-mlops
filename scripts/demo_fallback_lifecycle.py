"""Manual fallback: trigger a lifecycle cycle via POST (Issue 27).

Use this when the sensor mis-fires during the live talk. Sends the same
conf the sensor would push, bypassing Prometheus polling.

Usage:
    python -m scripts.demo_fallback_lifecycle --class Casting_class2
    python -m scripts.demo_fallback_lifecycle --class Casting_class3
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import urllib.request
import urllib.error


WARMSTART_CHECKPOINTS = {
    "Casting_class2": ".cache/iqa/models/rd_feature_ae_class2_precuit/checkpoint.pt",
    "Casting_class3": ".cache/iqa/models/rd_feature_ae_class3_precuit/checkpoint.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class", dest="triggering_class", required=True,
                        choices=["Casting_class2", "Casting_class3"])
    parser.add_argument("--airflow-url", default=None,
                        help="Airflow REST API base URL (default: $IQA_AIRFLOW_API_URL or http://localhost:8080/api/v1)")
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def trigger_lifecycle(
    triggering_class: str,
    *,
    airflow_url: str,
    user: str,
    password: str,
    dry_run: bool = False,
) -> dict:
    checkpoint = WARMSTART_CHECKPOINTS.get(triggering_class, "")
    conf = {
        "scenario_id": "drift_domain_extension",
        "trigger_reason": "drift",
        "triggering_class": triggering_class,
        "retrain_scope": "incremental_coverage",
        "drift_confirmed": True,
        "candidate_init_checkpoint": checkpoint,
        "mode": "train-on-trigger",
        "max_events": 8,
        "epochs": 1,
        "max_cycles": 1,
    }

    print(f"Triggering lifecycle for {triggering_class}")
    print(f"  warm-start checkpoint: {checkpoint}")
    print(f"  conf: {json.dumps(conf, indent=2)}")

    if dry_run:
        print("\n[DRY RUN] Would POST to Airflow — skipping.")
        return {"status": "dry_run", "conf": conf}

    url = f"{airflow_url}/dags/iqa_lifecycle/dagRuns"
    payload = json.dumps({"conf": conf}).encode("utf-8")
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {token}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            result = json.loads(resp.read().decode("utf-8"))
            dag_run_id = result.get("dag_run_id", "unknown")
            print(f"\nLifecycle triggered: dag_run_id={dag_run_id}")
            return result
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"\nHTTP {exc.code}: {body}")
        raise


def main() -> None:
    args = parse_args()
    airflow_url = args.airflow_url or os.environ.get(
        "IQA_AIRFLOW_API_URL", "http://localhost:8080/api/v1"
    ).rstrip("/")
    user = args.user or os.environ.get("IQA_AIRFLOW_API_USER", "airflow")
    password = args.password or os.environ.get("IQA_AIRFLOW_API_PASSWORD", "airflow")

    trigger_lifecycle(
        args.triggering_class,
        airflow_url=airflow_url,
        user=user,
        password=password,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
