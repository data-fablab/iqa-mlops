"""Host orchestration driver for the online domain-drift demo (chemin B).

Drives the live Docker stack through the user's narrative for each uncovered
domain class (``Casting_class2`` then ``Casting_class3``):

    Vert (baseline) -> Orange (IqaDriftProxyWarning) -> Rouge (IqaDriftProxy,
    critical) -> iqa_drift_sensor triggers iqa_lifecycle (GPU retrain) ->
    class marked "covered" -> Vert again, alerts resolve.

It is a *demo orchestrator*, not part of the inference path: it (a) streams
``POST /predict`` calls so Prometheus sees a steady prediction rate, and (b)
edits ``deploy/drift-state/state.json`` to move a class between the synthetic
reconstruction bands the live scorer reads (``covered``/``orange``/``red``).

The thresholds live once, in the Prometheus rules
(``deploy/prometheus/rules/iqa_drift_proxy.rules.yml``); this script only
*observes* the resulting ALERTS series and the Airflow run states. Nothing here
duplicates a threshold (ADR 0008 / decision 7).

Run from the repo root with the stack already up::

    .venv/Scripts/python.exe -m scripts.run_online_drift_demo

Use ``--dry-run`` to rehearse the phase machine without waiting on GPU retrains.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = REPO_ROOT / "deploy" / "drift-state" / "state.json"

API_URL = "http://localhost:8000"
PROMETHEUS_URL = "http://localhost:9090"
AIRFLOW_API_URL = "http://localhost:8080/api/v1"
AIRFLOW_AUTH = ("airflow", "airflow")
SERVICE_TOKEN = "dev-service-token"
SCENARIO_ID = "drift_domain_extension"
BASELINE_CLASS = "Casting_class1"

WARNING_ALERT = "IqaDriftProxyWarning"
CRITICAL_ALERT = "IqaDriftProxy"
LIFECYCLE_DAG = "iqa_lifecycle"

# Classes to drift through, in order. Each one runs the full narrative.
DRIFT_CLASSES = ["Casting_class2", "Casting_class3"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# State file (per-class synthetic band the live scorer reads)
# --------------------------------------------------------------------------- #
def read_state() -> dict[str, str]:
    try:
        raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(raw.get("classes", {}))


def write_state(classes: dict[str, str]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"classes": classes}, indent=0) + "\n", encoding="utf-8")
    tmp.replace(STATE_FILE)


def set_band(source_class: str, band: str) -> None:
    classes = read_state()
    classes[source_class] = band
    write_state(classes)
    log(f"state: {source_class} -> {band}  ({classes})")


# --------------------------------------------------------------------------- #
# Prediction streaming (background thread keeps the rate window fed)
# --------------------------------------------------------------------------- #
@dataclass
class Streamer:
    """Continuously POST /predict for the current source class at ~rate/s."""

    rate_per_sec: float = 3.0
    _source_class: str = BASELINE_CLASS
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    last_decision: str = "?"
    sent: int = 0

    def set_class(self, source_class: str) -> None:
        self._source_class = source_class

    def _run(self) -> None:
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {SERVICE_TOKEN}"})
        delay = 1.0 / self.rate_per_sec
        while not self._stop.is_set():
            try:
                resp = session.post(
                    f"{API_URL}/predict",
                    json={
                        "piece_event_id": f"demo_{self.sent}",
                        "scenario_id": SCENARIO_ID,
                        "image_uri": "s3://iqa-raw/demo/frame.png",
                        "source_class": self._source_class,
                    },
                    timeout=10,
                )
                self.sent += 1
                if resp.ok:
                    self.last_decision = resp.json()["prediction"]["decision"]
            except requests.RequestException:
                pass
            self._stop.wait(delay)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


# --------------------------------------------------------------------------- #
# Observation helpers (read-only; thresholds owned by Prometheus rules)
# --------------------------------------------------------------------------- #
def alert_state(alert_name: str) -> str:
    """Return the highest state of ``alert_name`` (firing > pending > inactive)."""
    try:
        payload = requests.get(f"{PROMETHEUS_URL}/api/v1/alerts", timeout=10).json()
    except (requests.RequestException, ValueError):
        return "unknown"
    states = {a.get("state") for a in payload.get("data", {}).get("alerts", []) if a.get("labels", {}).get("alertname") == alert_name}
    for level in ("firing", "pending"):
        if level in states:
            return level
    return "inactive"


def lifecycle_run_ids() -> set[str]:
    try:
        payload = requests.get(
            f"{AIRFLOW_API_URL}/dags/{LIFECYCLE_DAG}/dagRuns",
            params={"order_by": "-execution_date", "limit": "30"},
            auth=AIRFLOW_AUTH,
            timeout=10,
        ).json()
    except (requests.RequestException, ValueError):
        return set()
    return {r["dag_run_id"] for r in payload.get("dag_runs", [])}


def lifecycle_run_state(run_id: str) -> str:
    try:
        payload = requests.get(
            f"{AIRFLOW_API_URL}/dags/{LIFECYCLE_DAG}/dagRuns/{run_id}",
            auth=AIRFLOW_AUTH,
            timeout=10,
        ).json()
    except (requests.RequestException, ValueError):
        return "unknown"
    return payload.get("state", "unknown")


def wait_until(predicate, *, timeout: float, poll: float, desc: str, streamer: Streamer) -> bool:
    """Poll ``predicate`` until true or timeout; logs progress + live decision."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            log(f"OK: {desc}")
            return True
        remaining = int(deadline - time.time())
        log(f"  ... waiting for {desc} (last decision={streamer.last_decision}, sent={streamer.sent}, {remaining}s left)")
        time.sleep(poll)
    log(f"TIMEOUT waiting for {desc}")
    return False


# --------------------------------------------------------------------------- #
# Phase machine
# --------------------------------------------------------------------------- #
def run_class_narrative(source_class: str, streamer: Streamer, *, dry_run: bool) -> bool:
    log(f"==================== DRIFT NARRATIVE: {source_class} ====================")

    # Phase 0 — baseline: this class is unknown but not yet streamed; show Vert.
    set_band(source_class, "covered")
    streamer.set_class(source_class)
    log(f"baseline: streaming {source_class} (covered) -> expect Vert")
    time.sleep(8)

    # Phase 1 — Orange: domain appears, model still covers it poorly.
    set_band(source_class, "orange")
    if not wait_until(
        lambda: alert_state(WARNING_ALERT) == "firing",
        timeout=240, poll=10, desc=f"{WARNING_ALERT} firing (Orange)", streamer=streamer,
    ):
        return False

    # Phase 2 — Rouge: confirmed drift -> critical -> sensor triggers retrain.
    # Generous timeout: after the Orange->red switch the 2m rate window is still
    # Orange-dominated, so the Rouge ratio only crosses 0.5 once ~one window has
    # aged out (~2 min) -- plus the rule's for:30s.
    before = lifecycle_run_ids()
    set_band(source_class, "red")
    if not wait_until(
        lambda: alert_state(CRITICAL_ALERT) == "firing",
        timeout=420, poll=10, desc=f"{CRITICAL_ALERT} firing (Rouge)", streamer=streamer,
    ):
        return False

    # Phase 3 — sensor fires iqa_lifecycle; capture the new run and await success.
    new_run = None
    deadline = time.time() + 180
    while time.time() < deadline and new_run is None:
        fresh = lifecycle_run_ids() - before
        if fresh:
            new_run = sorted(fresh)[-1]
            break
        log(f"  ... waiting for iqa_drift_sensor to trigger {LIFECYCLE_DAG} ({int(deadline - time.time())}s left)")
        time.sleep(10)
    if new_run is None:
        log("TIMEOUT: sensor did not trigger a lifecycle run")
        return False
    log(f"iqa_lifecycle triggered: run_id={new_run}")

    if dry_run:
        log("dry-run: skipping GPU retrain wait")
    else:
        ok = wait_until(
            lambda: lifecycle_run_state(new_run) in {"success", "failed"},
            timeout=1800, poll=20, desc=f"{LIFECYCLE_DAG} run {new_run} to finish", streamer=streamer,
        )
        final = lifecycle_run_state(new_run)
        if not ok or final != "success":
            log(f"lifecycle run ended in state={final} — aborting narrative")
            return False
        log(f"retrain SUCCESS: {source_class} now covered by the model")

    # Phase 4 — recovery: model now covers the class -> Vert; drain the alerts.
    set_band(source_class, "covered")
    log("recovery: streaming covered class (Vert) to drain the Rouge rate window")
    if not wait_until(
        lambda: alert_state(CRITICAL_ALERT) != "firing",
        timeout=300, poll=15, desc=f"{CRITICAL_ALERT} resolved", streamer=streamer,
    ):
        return False
    # Let the warning drain too so the next class starts clean.
    wait_until(
        lambda: alert_state(WARNING_ALERT) != "firing",
        timeout=240, poll=15, desc=f"{WARNING_ALERT} resolved", streamer=streamer,
    )
    log(f"RECOVERED: {source_class} green. Cooling down before next class...")
    if not dry_run:
        time.sleep(130)  # > sensor cooldown_seconds (120) so the next class can trigger
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="skip GPU retrain waits")
    parser.add_argument("--classes", nargs="*", default=DRIFT_CLASSES, help="classes to drift through")
    parser.add_argument("--rate", type=float, default=3.0, help="predictions/sec")
    parser.add_argument("--no-reset", action="store_true",
                        help="keep current state.json (preserve already-covered classes)")
    args = parser.parse_args()

    # Reset preserves already-covered classes so re-runs are additive (a class
    # retrained in a previous run stays Vert). Without --no-reset, classes about
    # to be (re)driven are dropped back to the baseline.
    if args.no_reset:
        log(f"state kept as-is ({read_state()})")
    else:
        kept = {k: v for k, v in read_state().items()
                if v in {"covered", "vert", "green"} and k not in args.classes}
        kept[BASELINE_CLASS] = "covered"
        write_state(kept)
        log(f"state reset -> baseline + kept covered ({kept})")

    streamer = Streamer(rate_per_sec=args.rate)
    streamer.set_class(BASELINE_CLASS)
    streamer.start()
    log(f"streamer started @ {args.rate}/s")

    try:
        for source_class in args.classes:
            if not run_class_narrative(source_class, streamer, dry_run=args.dry_run):
                log(f"narrative for {source_class} did not complete — stopping")
                return 1
        log("==================== DEMO COMPLETE — all classes recovered to Vert ====================")
        return 0
    finally:
        streamer.stop()
        log(f"streamer stopped (sent={streamer.sent})")


if __name__ == "__main__":
    sys.exit(main())
