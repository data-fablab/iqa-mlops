"""Real drift driver (chemin A) — fire the drift proxy from genuine decisions.

Twin of ``scripts/run_online_drift_demo.py`` (the synthetic smoke-test, chemin B),
but it never fakes a score: it replays ``casting_flux_replay_plan_drift.csv`` over
the **real** images of the dataset bind-mounted in ``iqa-inference`` and lets the
``RealFeatureAEScorer`` produce ``Vert/Orange/Rouge`` from the actual reconstruction
error (ADR 0010 §1-2).

For each phase, in order::

    baseline_domain_class1  -> in-distribution  -> Vert  -> ratio proxy ~ 0
    domain_extension_class2 -> out-of-distribution -> Orange/Rouge -> ratio > 0.5
    domain_extension_class3 -> out-of-distribution -> Orange/Rouge -> ratio > 0.5

it resolves each row's ``relative_paths`` to a ``file://`` URI under the container
dataset root, streams ``POST /predict`` so Prometheus sees a steady drift-regime
rate, and **observes** the resulting ``ALERTS`` series (``IqaDriftProxyWarning`` then
``IqaDriftProxy``) plus any ``iqa_lifecycle`` run the sensor triggers.

Unlike the synthetic driver it **never writes** ``deploy/drift-state/state.json``:
the decision comes from pixels, and recovery to Vert (out of scope here) comes only
from a retrained checkpoint. The 0.5 proxy threshold lives once in the Prometheus
rules (``deploy/prometheus/rules/iqa_drift_proxy.rules.yml``); this script only reads
the resulting ratio/alerts and never duplicates a threshold (ADR 0010 §7, dec. 7).

Run from the repo root with the stack already up::

    .venv/Scripts/python.exe -m scripts.run_real_drift_demo

Use ``--max-per-phase`` to cap the distinct images cycled per phase.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = REPO_ROOT / "deploy" / "drift-state" / "state.json"
DEFAULT_PLAN = REPO_ROOT / "data" / "metadata" / "casting_flux_replay_plan_drift.csv"

# Dataset root *inside the iqa-inference container* (docker-compose bind mount):
# the file:// URIs we post must resolve there, not on the host.
DEFAULT_DATASET_ROOT = "/opt/iqa/iqa-mlops/data/raw/hss-iad"

API_URL = "http://localhost:8000"
PROMETHEUS_URL = "http://localhost:9090"
AIRFLOW_API_URL = "http://localhost:8080/api/v1"
AIRFLOW_AUTH = ("airflow", "airflow")
SERVICE_TOKEN = "dev-service-token"
SCENARIO_ID = "drift_domain_extension"

WARNING_ALERT = "IqaDriftProxyWarning"
CRITICAL_ALERT = "IqaDriftProxy"
LIFECYCLE_DAG = "iqa_lifecycle"

# Canonical phase order of the drift plan: baseline (in-distribution) first, then
# the out-of-distribution domain extensions in coverage order.
PHASE_ORDER = ["baseline_domain_class1", "domain_extension_class2", "domain_extension_class3"]
BASELINE_PHASE = PHASE_ORDER[0]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Plan -> file:// resolution (pure; unit-tested)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sample:
    piece_event_id: str
    image_uri: str


def _split_paths(value: str) -> list[str]:
    return [part.strip() for part in value.replace(";", "|").replace(",", "|").split("|") if part.strip()]


def to_file_uri(relative_path: str, dataset_root: str) -> str:
    """Resolve a plan ``relative_path`` to a ``file://`` URI under ``dataset_root``.

    ``dataset_root`` is the absolute path of the dataset *inside the container*
    (e.g. ``/opt/iqa/iqa-mlops/data/raw/hss-iad``); the scorer's
    ``resolve_image_path`` reads ``urlparse(uri).path`` back out.
    """
    root = dataset_root.replace("\\", "/").rstrip("/")
    rel = relative_path.replace("\\", "/").lstrip("/")
    return f"file://{root}/{rel}"


def load_phase_samples(plan_path: Path, dataset_root: str) -> dict[str, list[Sample]]:
    """Group the plan into ``{phase: [Sample]}`` using the first view per row.

    One ``/predict`` per piece event keeps ``piece_event_id`` unique; the streamer
    cycles the list, so distinct-image count only bounds variety, not volume.
    """
    by_phase: dict[str, list[Sample]] = {}
    with plan_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            phase = (row.get("scenario_phase") or "").strip()
            paths = _split_paths(row.get("relative_paths") or row.get("relative_path") or "")
            if not phase or not paths:
                continue
            piece_event_id = (row.get("piece_event_id") or row.get("simulated_event_id") or "").strip()
            by_phase.setdefault(phase, []).append(
                Sample(piece_event_id=piece_event_id, image_uri=to_file_uri(paths[0], dataset_root))
            )
    return by_phase


def ordered_phases(present: dict[str, list[Sample]]) -> list[str]:
    """Phases present in the plan, in canonical order; unknowns appended last."""
    known = [phase for phase in PHASE_ORDER if phase in present]
    extra = sorted(phase for phase in present if phase not in PHASE_ORDER)
    return known + extra


# --------------------------------------------------------------------------- #
# Prediction streaming (one phase's real images, cycled at ~rate/s)
# --------------------------------------------------------------------------- #
@dataclass
class PhaseStreamer:
    """Continuously POST /predict for the current phase's real images."""

    rate_per_sec: float = 3.0
    _samples: list[Sample] = field(default_factory=list)
    _phase: str = ""
    _index: int = 0
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    last_decision: str = "?"
    sent: int = 0
    decisions: dict[str, int] = field(default_factory=dict)

    def set_phase(self, phase: str, samples: list[Sample]) -> None:
        self._phase = phase
        self._samples = samples
        self._index = 0

    def _run(self) -> None:
        session = requests.Session()
        session.headers.update({"Authorization": f"Bearer {SERVICE_TOKEN}"})
        delay = 1.0 / self.rate_per_sec
        while not self._stop.is_set():
            samples = self._samples
            if samples:
                sample = samples[self._index % len(samples)]
                self._index += 1
                try:
                    resp = session.post(
                        f"{API_URL}/predict",
                        json={
                            "piece_event_id": sample.piece_event_id or f"real_{self.sent}",
                            "scenario_id": SCENARIO_ID,
                            "image_uri": sample.image_uri,
                        },
                        timeout=30,
                    )
                    self.sent += 1
                    if resp.ok:
                        decision = resp.json()["prediction"]["decision"]
                        self.last_decision = decision
                        self.decisions[decision] = self.decisions.get(decision, 0) + 1
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
    states = {
        a.get("state")
        for a in payload.get("data", {}).get("alerts", [])
        if a.get("labels", {}).get("alertname") == alert_name
    }
    for level in ("firing", "pending"):
        if level in states:
            return level
    return "inactive"


def proxy_ratio(decision: str) -> float:
    """Read the live drift-regime ratio for ``decision`` (the rule's numerator/denom).

    Observation only — the 0.5 firing threshold stays in the Prometheus rule; this
    just reads the ratio value to log progress and assert class1 ~ 0.
    """
    expr = (
        f'sum(rate(iqa_prediction_total{{scenario_id=~"drift.*",decision="{decision}"}}[2m]))'
        ' / clamp_min(sum(rate(iqa_prediction_total{scenario_id=~"drift.*"}[2m])), 1e-9)'
    )
    try:
        payload = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": expr}, timeout=10).json()
        result = payload.get("data", {}).get("result", [])
        return float(result[0]["value"][1]) if result else 0.0
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return 0.0


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


def _state_fingerprint() -> str | None:
    try:
        return hashlib.sha256(STATE_FILE.read_bytes()).hexdigest()
    except OSError:
        return None


def wait_until(predicate, *, timeout: float, poll: float, desc: str, streamer: PhaseStreamer) -> bool:
    """Poll ``predicate`` until true or timeout; logs progress + live decision."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            log(f"OK: {desc}")
            return True
        remaining = int(deadline - time.time())
        log(
            f"  ... waiting for {desc} "
            f"(last={streamer.last_decision}, sent={streamer.sent}, {remaining}s left)"
        )
        time.sleep(poll)
    log(f"TIMEOUT waiting for {desc}")
    return False


# --------------------------------------------------------------------------- #
# Phase machine
# --------------------------------------------------------------------------- #
def run_baseline_phase(samples: list[Sample], streamer: PhaseStreamer, *, settle: float) -> bool:
    log(f"==================== BASELINE: {BASELINE_PHASE} ({len(samples)} images) ====================")
    streamer.set_phase(BASELINE_PHASE, samples)
    log(f"streaming real class1 images -> expect Vert, ratio proxy ~ 0 ({int(settle)}s settle)")
    time.sleep(settle)
    rouge_ratio = proxy_ratio("Rouge")
    orange_ratio = proxy_ratio("Orange")
    log(f"baseline ratios: Orange={orange_ratio:.3f} Rouge={rouge_ratio:.3f} decisions={streamer.decisions}")
    if alert_state(CRITICAL_ALERT) == "firing":
        log("FAIL: IqaDriftProxy is firing during the class1 baseline (expected quiet)")
        return False
    if rouge_ratio > 0.5 or orange_ratio > 0.5:
        log("FAIL: drift proxy ratio crossed 0.5 on the in-distribution baseline")
        return False
    log("OK: class1 baseline quiet (ratio proxy ~ 0, no alert)")
    return True


def run_extension_phase(phase: str, samples: list[Sample], streamer: PhaseStreamer) -> bool:
    log(f"==================== DRIFT: {phase} ({len(samples)} images) ====================")
    before = lifecycle_run_ids()
    streamer.set_phase(phase, samples)
    log(f"streaming real OOD images for {phase} -> expect Orange then Rouge")

    if not wait_until(
        lambda: alert_state(WARNING_ALERT) == "firing",
        timeout=240, poll=10, desc=f"{WARNING_ALERT} firing (Orange ratio > 0.5)", streamer=streamer,
    ):
        return False
    # The 2m rate window is still baseline/Orange-dominated right after the switch,
    # so the Rouge ratio only crosses 0.5 once ~one window has aged out.
    if not wait_until(
        lambda: alert_state(CRITICAL_ALERT) == "firing",
        timeout=420, poll=10, desc=f"{CRITICAL_ALERT} firing (Rouge ratio > 0.5)", streamer=streamer,
    ):
        return False
    log(f"ratios now: Orange={proxy_ratio('Orange'):.3f} Rouge={proxy_ratio('Rouge'):.3f} decisions={streamer.decisions}")

    # Observe (read-only) whether the sensor triggered a retrain — recovery itself
    # is out of scope for this driver (issues 9/10).
    fresh = lifecycle_run_ids() - before
    if fresh:
        log(f"observed: iqa_drift_sensor triggered {LIFECYCLE_DAG} run(s) {sorted(fresh)}")
    else:
        log(f"note: no new {LIFECYCLE_DAG} run observed yet (sensor may fire after its poll interval)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                        help="container path of the dataset bind-mounted in iqa-inference")
    parser.add_argument("--rate", type=float, default=3.0, help="predictions/sec")
    parser.add_argument("--max-per-phase", type=int, default=None,
                        help="cap the distinct images cycled per phase")
    parser.add_argument("--baseline-settle", type=float, default=150.0,
                        help="seconds to stream class1 before asserting the proxy is quiet")
    parser.add_argument("--phases", nargs="*", default=None,
                        help="subset/override of phases to run (default: all present, canonical order)")
    args = parser.parse_args()

    by_phase = load_phase_samples(args.plan, args.dataset_root)
    if not by_phase:
        log(f"no usable rows in {args.plan}")
        return 1
    if args.max_per_phase is not None:
        by_phase = {phase: samples[: args.max_per_phase] for phase, samples in by_phase.items()}

    phases = args.phases or ordered_phases(by_phase)
    log(f"phases: {phases}  (dataset root {args.dataset_root})")
    if BASELINE_PHASE not in by_phase:
        log(f"WARNING: baseline phase {BASELINE_PHASE} absent from the plan")

    # This driver must never touch the synthetic state file (ADR 0010 §1): capture a
    # fingerprint up front and verify it is unchanged at the end.
    state_before = _state_fingerprint()

    streamer = PhaseStreamer(rate_per_sec=args.rate)
    streamer.start()
    log(f"streamer started @ {args.rate}/s")

    rc = 0
    try:
        for phase in phases:
            samples = by_phase.get(phase)
            if not samples:
                log(f"skip {phase}: no samples")
                continue
            if phase == BASELINE_PHASE:
                ok = run_baseline_phase(samples, streamer, settle=args.baseline_settle)
            else:
                ok = run_extension_phase(phase, samples, streamer)
            if not ok:
                log(f"phase {phase} did not meet expectations — stopping")
                rc = 1
                break
        else:
            log("==================== REAL DRIFT DRIVE COMPLETE — proxy fired on real decisions ====================")
    finally:
        streamer.stop()
        log(f"streamer stopped (sent={streamer.sent}, decisions={streamer.decisions})")

    state_after = _state_fingerprint()
    if state_before != state_after:
        log(f"FAIL: {STATE_FILE} changed during the run (this driver must never write it)")
        return 1
    log("OK: drift-state/state.json untouched (recovery is checkpoint-driven, not state-driven)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
