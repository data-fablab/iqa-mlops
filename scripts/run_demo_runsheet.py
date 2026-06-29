"""One-command runsheet driver — the full autonomous demo class1 -> class2 -> class3.

This is the single entry point behind ``make demo-runsheet``. It stitches together
the pieces that the runsheet (``docs/runsheet_demo_20min.md``) otherwise walks
through by hand, so the whole ~50 min autonomous story runs unattended:

    Phase 0  reset to the clean class1-only baseline + pre-warm one /predict
    Phase A  for each extension class (class2 then class3):
               1. stream that class's real images continuously,
               2. wait for PatchCore to fire the domain-drift alert,
               3. wait for the sensor to trigger ``iqa_lifecycle`` autonomously
                  (fallback: POST the same conf the sensor would, see Issue 27),
               4. keep streaming through retrain/promotion/refresh/restart and
                  wait for *recovery* — the per-class out-of-domain ratio drops as
                  the refreshed PatchCore now covers the class,
               5. stop the streamer before the next class.

The key difference from ``run_dual_drift_demo`` (which the handoff flagged): the
streamer stays up *through* recovery instead of stopping the moment the alert
fires, so the dashboard actually shows class2/class3 going green on live traffic.

This driver is read-only on ``deploy/drift-state/state.json`` (real decisions come
from pixels; recovery comes from the promoted checkpoint, never from that file)
and owns no thresholds — Prometheus rules do. Run from the repo root with the
stack already up (the docker compose overrides from the runsheet pre-reqs)::

    .venv/Scripts/python.exe -m scripts.run_demo_runsheet

Useful flags::

    --classes Casting_class2          # only the first cycle
    --no-reset                        # keep the current artifacts (resume)
    --no-fallback                     # never POST the manual trigger; wait only
    --dry-run                         # print the plan and exit (no streaming)
"""

from __future__ import annotations

import argparse
import sys
import time

import requests

from scripts import demo_fallback_lifecycle, demo_reset
from scripts.run_dual_drift_demo import PATCHCORE_ALERT, patchcore_ratio
from scripts.run_real_drift_demo import (
    API_URL,
    BASELINE_PHASE,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PLAN,
    PROMETHEUS_URL,
    SCENARIO_ID,
    SERVICE_TOKEN,
    PhaseStreamer,
    alert_state,
    lifecycle_run_ids,
    load_phase_samples,
    log,
)

# Extension classes in coverage order, mapped to their replay phase + warm-start.
CLASS_TO_PHASE = {
    "Casting_class2": "domain_extension_class2",
    "Casting_class3": "domain_extension_class3",
}
DEFAULT_CLASSES = ["Casting_class2", "Casting_class3"]


def per_class_ood_ratio(source_class: str) -> float:
    """Live out-of-domain ratio for one class (recovery falls as it gets covered).

    Numerator = out-of-domain PatchCore decisions for ``source_class``; denominator
    = all PatchCore decisions for that class. Both regimes carry ``source_class``
    in ``/metrics`` (see ``main.py``), so this drops toward 0 once the refreshed
    bank covers the class and fresh images come back in-domain.
    """
    expr = (
        f'sum(rate(iqa_domain_drift_total{{regime="out_of_domain",source_class="{source_class}"}}[2m]))'
        f' / clamp_min(sum(rate(iqa_domain_drift_total{{source_class="{source_class}"}}[2m])), 1e-9)'
    )
    try:
        payload = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query", params={"query": expr}, timeout=10
        ).json()
        result = payload.get("data", {}).get("result", [])
        return float(result[0]["value"][1]) if result else 0.0
    except (requests.RequestException, ValueError, KeyError, IndexError):
        return 0.0


def wait_until(predicate, *, timeout: float, poll: float, desc: str, streamer: PhaseStreamer) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            log(f"OK: {desc}")
            return True
        remaining = int(deadline - time.time())
        log(f"  ... waiting for {desc} (last={streamer.last_decision}, sent={streamer.sent}, {remaining}s left)")
        time.sleep(poll)
    log(f"TIMEOUT waiting for {desc}")
    return False


def prewarm(timeout: float) -> bool:
    """Send one class1 /predict so the first real call isn't a 60-90 s cold start."""
    by_phase = load_phase_samples(DEFAULT_PLAN, DEFAULT_DATASET_ROOT)
    samples = by_phase.get(BASELINE_PHASE)
    if not samples:
        log("prewarm skipped: no class1 baseline samples in the plan")
        return False
    sample = samples[0]
    payload = {
        "piece_event_id": sample.piece_event_id or "prewarm",
        "scenario_id": SCENARIO_ID,
        "image_uri": sample.image_uri,
        "source_class": "Casting_class1",
    }
    log(f"pre-warming inference with one class1 /predict (up to {int(timeout)}s cold start)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.post(
                f"{API_URL}/predict",
                json=payload,
                headers={"Authorization": f"Bearer {SERVICE_TOKEN}"},
                timeout=90,
            )
            if resp.ok:
                decision = resp.json().get("prediction", {}).get("decision", "?")
                log(f"OK: prewarm /predict returned decision={decision}")
                return True
        except requests.RequestException:
            pass
        time.sleep(5)
    log("WARN: prewarm did not get a clean response (continuing anyway)")
    return False


def run_class_cycle(
    source_class: str,
    *,
    streamer: PhaseStreamer,
    samples,
    args: argparse.Namespace,
) -> bool:
    """Drift -> autonomous (or fallback) retrain -> recovery, for one class."""
    phase = CLASS_TO_PHASE[source_class]
    log("=" * 78)
    log(f"CYCLE {source_class}  (phase {phase}, {len(samples)} images)")
    log("=" * 78)

    runs_before = lifecycle_run_ids()
    streamer.set_phase(phase, samples)
    log(f"streaming real {source_class} images @ {args.rate}/s")

    # 1) Drift detected by PatchCore.
    if not wait_until(
        lambda: alert_state(PATCHCORE_ALERT) == "firing",
        timeout=args.drift_timeout, poll=10,
        desc=f"{PATCHCORE_ALERT} firing (out-of-domain on {source_class})",
        streamer=streamer,
    ):
        return False
    log(f"  PatchCore out-of-domain ratio={patchcore_ratio():.2f} — drift confirmed for {source_class}")

    # 2) Autonomous sensor trigger, with a manual fallback after sensor-timeout.
    sensor_fired = wait_until(
        lambda: bool(lifecycle_run_ids() - runs_before),
        timeout=args.sensor_timeout, poll=10,
        desc=f"sensor to trigger iqa_lifecycle for {source_class}",
        streamer=streamer,
    )
    if not sensor_fired:
        if args.no_fallback:
            log("FAIL: sensor never fired and --no-fallback was set")
            return False
        log(f"sensor silent after {int(args.sensor_timeout)}s — firing manual fallback (Issue 27)")
        demo_fallback_lifecycle.trigger_lifecycle(
            source_class,
            airflow_url=args.airflow_url,
            user=args.airflow_user,
            password=args.airflow_password,
        )

    # 3) Recovery: keep streaming until the refreshed bank covers the class.
    if not wait_until(
        lambda: per_class_ood_ratio(source_class) < args.recovery_threshold,
        timeout=args.recovery_timeout, poll=15,
        desc=f"{source_class} recovery (out-of-domain ratio < {args.recovery_threshold})",
        streamer=streamer,
    ):
        log("FAIL: no recovery observed (retrain/promotion/refresh/restart may have stalled)")
        return False

    log(f"RECOVERED: {source_class} out-of-domain ratio={per_class_ood_ratio(source_class):.2f} — now in-domain")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classes", nargs="*", default=DEFAULT_CLASSES,
                        help="extension classes to drive, in order (default: class2 class3)")
    parser.add_argument("--rate", type=float, default=6.0, help="predictions/sec while streaming")
    parser.add_argument("--max-per-phase", type=int, default=20,
                        help="distinct images cycled per class")
    parser.add_argument("--recovery-threshold", type=float, default=0.3,
                        help="per-class out-of-domain ratio below which the class counts as recovered")
    parser.add_argument("--drift-timeout", type=float, default=300.0,
                        help="seconds to wait for PatchCore to fire")
    parser.add_argument("--sensor-timeout", type=float, default=150.0,
                        help="seconds to wait for the autonomous sensor before the manual fallback")
    parser.add_argument("--recovery-timeout", type=float, default=1500.0,
                        help="seconds to wait for retrain+promotion+refresh+restart (~17 min/cycle)")
    parser.add_argument("--prewarm-timeout", type=float, default=120.0)
    parser.add_argument("--no-reset", action="store_true", help="skip demo-reset (resume mid-demo)")
    parser.add_argument("--no-prewarm", action="store_true")
    parser.add_argument("--no-fallback", action="store_true",
                        help="never POST the manual lifecycle trigger; wait for the sensor only")
    parser.add_argument("--airflow-url", default="http://localhost:8080/api/v1")
    parser.add_argument("--airflow-user", default="airflow")
    parser.add_argument("--airflow-password", default="airflow")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)

    unknown = [c for c in args.classes if c not in CLASS_TO_PHASE]
    if unknown:
        log(f"unknown class(es): {unknown} (known: {sorted(CLASS_TO_PHASE)})")
        return 2

    log("RUNSHEET DEMO — autonomous class1 -> " + " -> ".join(args.classes))
    log(f"  reset={not args.no_reset} prewarm={not args.no_prewarm} fallback={not args.no_fallback}")
    if args.dry_run:
        for source_class in args.classes:
            log(f"  would drive {source_class} via phase {CLASS_TO_PHASE[source_class]}")
        log("dry run — exiting before any streaming")
        return 0

    # Phase 0 — clean class1-only baseline.
    if not args.no_reset:
        log("Phase 0: demo-reset (restore class1-only baseline + restart api/inference)")
        rc = demo_reset.main([])
        if rc != 0:
            log("FAIL: demo-reset returned non-zero")
            return rc
    if not args.no_prewarm:
        prewarm(args.prewarm_timeout)

    by_phase = load_phase_samples(DEFAULT_PLAN, DEFAULT_DATASET_ROOT)
    if args.max_per_phase is not None:
        by_phase = {p: s[: args.max_per_phase] for p, s in by_phase.items()}

    streamer = PhaseStreamer(rate_per_sec=args.rate)
    streamer.start()
    log(f"streamer started @ {args.rate}/s (stays up through each recovery)")

    rc = 0
    try:
        for source_class in args.classes:
            samples = by_phase.get(CLASS_TO_PHASE[source_class])
            if not samples:
                log(f"skip {source_class}: no samples for phase {CLASS_TO_PHASE[source_class]}")
                rc = 1
                break
            if not run_class_cycle(source_class, streamer=streamer, samples=samples, args=args):
                rc = 1
                break
        else:
            log("=" * 78)
            log("RUNSHEET DEMO COMPLETE — all classes covered, system adapted autonomously")
            log("  verify: PatchCore covered_classes == [class1, class2, class3]")
            log("=" * 78)
    finally:
        streamer.stop()
        log(f"streamer stopped (sent={streamer.sent}, decisions={streamer.decisions})")

    return rc


if __name__ == "__main__":
    sys.exit(main())
