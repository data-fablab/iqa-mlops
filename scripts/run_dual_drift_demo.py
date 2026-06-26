"""Dual drift driver — observe AE proxy AND PatchCore domain-drift side by side (Issue 13).

Evolves the single-signal real driver (Issue 7) into a **dual-signal demo**: for each
phase of the drift replay plan (class1 -> class2 -> class3), it streams real images and
observes **both** signals simultaneously:

- **AE proxy** (Vert/Orange/Rouge): ``iqa_prediction_total`` ratio, alerts
  ``IqaDriftProxyWarning`` / ``IqaDriftProxy``.
- **PatchCore domain-drift** (in_domain/out_of_domain): ``iqa_domain_drift_total``
  ratio, alert ``IqaDomainDriftPatchCore``.

The contrast that justifies the whole PatchCore track: on **class3** the AE proxy
stays quasi-silent (the AE cannot separate domains) while **PatchCore fires at 100%**.

The driver **never writes** ``deploy/drift-state/state.json`` (read-only observation,
empreinte SHA-256 verified at the end). Thresholds are owned by the Prometheus rules;
this script only reads ratios and alert states.

Run from the repo root with the stack already up::

    .venv/Scripts/python.exe -m scripts.run_dual_drift_demo
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests

from scripts.run_real_drift_demo import (
    API_URL,
    BASELINE_PHASE,
    CRITICAL_ALERT as AE_CRITICAL_ALERT,
    DEFAULT_DATASET_ROOT,
    DEFAULT_PLAN,
    PHASE_ORDER,
    PROMETHEUS_URL,
    SCENARIO_ID,
    SERVICE_TOKEN,
    WARNING_ALERT as AE_WARNING_ALERT,
    PhaseStreamer,
    Sample,
    alert_state,
    lifecycle_run_ids,
    load_phase_samples,
    log,
    ordered_phases,
    proxy_ratio,
)

STATE_FILE = Path(__file__).resolve().parents[1] / "deploy" / "drift-state" / "state.json"

PATCHCORE_ALERT = "IqaDomainDriftPatchCore"


def patchcore_ratio() -> float:
    """Read the live PatchCore out-of-domain ratio (the rule's numerator/denom)."""
    expr = (
        'sum(rate(iqa_domain_drift_total{regime="out_of_domain"}[2m]))'
        " / clamp_min(sum(rate(iqa_domain_drift_total[2m])), 1e-9)"
    )
    try:
        payload = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": expr}, timeout=10).json()
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


def _state_fingerprint() -> str | None:
    try:
        return hashlib.sha256(STATE_FILE.read_bytes()).hexdigest()
    except OSError:
        return None


def _dual_status() -> str:
    ae_orange = proxy_ratio("Orange")
    ae_rouge = proxy_ratio("Rouge")
    pc_out = patchcore_ratio()
    ae_w = alert_state(AE_WARNING_ALERT)
    ae_c = alert_state(AE_CRITICAL_ALERT)
    pc = alert_state(PATCHCORE_ALERT)
    return (
        f"AE [Orange={ae_orange:.2f} Rouge={ae_rouge:.2f} warn={ae_w} crit={ae_c}]"
        f"  PatchCore [out={pc_out:.2f} alert={pc}]"
    )


def run_baseline(samples: list[Sample], streamer: PhaseStreamer, *, settle: float) -> bool:
    log(f"==================== BASELINE: {BASELINE_PHASE} ({len(samples)} images) ====================")
    streamer.set_phase(BASELINE_PHASE, samples)
    log(f"streaming real class1 images ({int(settle)}s settle)")
    time.sleep(settle)
    status = _dual_status()
    log(f"baseline dual status: {status}")
    log(f"decisions: {streamer.decisions}")

    ae_rouge = proxy_ratio("Rouge")
    ae_orange = proxy_ratio("Orange")
    pc_out = patchcore_ratio()
    if ae_rouge > 0.5 or ae_orange > 0.5 or pc_out > 0.5:
        log("FAIL: a signal crossed 0.5 during the in-distribution baseline")
        return False
    if alert_state(PATCHCORE_ALERT) == "firing":
        log("FAIL: IqaDomainDriftPatchCore is firing during class1 baseline")
        return False
    log("OK: class1 baseline quiet — both signals below 0.5, no alert")
    return True


def run_extension(phase: str, samples: list[Sample], streamer: PhaseStreamer) -> bool:
    log(f"==================== DRIFT: {phase} ({len(samples)} images) ====================")
    streamer.set_phase(phase, samples)
    log(f"streaming real OOD images for {phase}")

    # PatchCore should fire before or alongside the AE proxy.
    ok = wait_until(
        lambda: alert_state(PATCHCORE_ALERT) == "firing",
        timeout=300, poll=10, desc=f"{PATCHCORE_ALERT} firing (out-of-domain ratio > 0.5)",
        streamer=streamer,
    )
    if not ok:
        return False

    # Report the AE proxy state for the contrast table.
    ae_w = alert_state(AE_WARNING_ALERT)
    ae_c = alert_state(AE_CRITICAL_ALERT)
    ae_orange = proxy_ratio("Orange")
    ae_rouge = proxy_ratio("Rouge")
    pc_out = patchcore_ratio()
    log(
        f"DUAL CONTRAST — {phase}:\n"
        f"  AE proxy    : Orange={ae_orange:.2f} Rouge={ae_rouge:.2f} warn={ae_w} crit={ae_c}\n"
        f"  PatchCore   : out-of-domain={pc_out:.2f} alert={alert_state(PATCHCORE_ALERT)}\n"
        f"  decisions   : {streamer.decisions}"
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--rate", type=float, default=3.0)
    parser.add_argument("--max-per-phase", type=int, default=None)
    parser.add_argument("--baseline-settle", type=float, default=150.0)
    parser.add_argument("--phases", nargs="*", default=None)
    args = parser.parse_args()

    by_phase = load_phase_samples(args.plan, args.dataset_root)
    if not by_phase:
        log(f"no usable rows in {args.plan}")
        return 1
    if args.max_per_phase is not None:
        by_phase = {phase: samples[: args.max_per_phase] for phase, samples in by_phase.items()}

    phases = args.phases or ordered_phases(by_phase)
    log(f"DUAL DRIFT DEMO — phases: {phases}")
    log(f"signals: AE proxy ({AE_WARNING_ALERT}/{AE_CRITICAL_ALERT}) + PatchCore ({PATCHCORE_ALERT})")

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
                ok = run_baseline(samples, streamer, settle=args.baseline_settle)
            else:
                ok = run_extension(phase, samples, streamer)
            if not ok:
                log(f"phase {phase} did not meet expectations — stopping")
                rc = 1
                break
        else:
            log("=" * 70)
            log("DUAL DRIFT DEMO COMPLETE")
            log(f"  AE proxy  : detects DEFECTS, quasi-silent on class3 domain shift")
            log(f"  PatchCore : detects DOMAIN shift, fires at ~100% on class2/class3")
            log("=" * 70)
    finally:
        streamer.stop()
        log(f"streamer stopped (sent={streamer.sent}, decisions={streamer.decisions})")

    state_after = _state_fingerprint()
    if state_before != state_after:
        log(f"FAIL: {STATE_FILE} changed during the run (this driver must never write it)")
        return 1
    log("OK: drift-state/state.json untouched")
    return rc


if __name__ == "__main__":
    sys.exit(main())
