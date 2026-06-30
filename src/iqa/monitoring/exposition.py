"""Prometheus exposition for the IQA lifecycle and drift observability subsystem.

This module owns the *stateful* lifecycle/drift exposition that the API serves
on ``/metrics`` for Grafana. It ingests events emitted by the Airflow batches
(lifecycle, drift), **sanitises** them — rejecting non-observable / sensitive
fields (``image``, ``path``, ``uri``, ``mask``, ``heatmap``, ``piece_event`` …),
a security-governance invariant — accumulates the observable state, and renders
the Prometheus text format.

The interface is intentionally narrow::

    record_lifecycle_event(event) -> None
    record_drift_event(event) -> None
    render_prometheus_lines() -> list[str]
    reset() -> None

Sanitisation failures raise :class:`ObservabilityRejection`; the API translates
them into HTTP 422 responses. Keeping the seam FastAPI-free lets the
sanitisation and rendering invariants be tested without a FastAPI ``TestClient``.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle via iqa.api
    from iqa.api.schemas import DriftEventRequest, LifecycleEventRequest


OBSERVABILITY_TRANSIENT_TTL_SECONDS = float(
    os.environ.get("IQA_OBSERVABILITY_TRANSIENT_TTL_SECONDS", "30")
)

LIFECYCLE_EPOCH_METRIC_ALIASES = {
    "pixel_aupimo_1e-5_1e-3": "pixel_aupimo",
    "pixel_aupimo": "pixel_aupimo",
    "pixel_ap": "pixel_ap",
    "image_ap": "image_ap",
    "false_negatives": "false_negatives",
}
LIFECYCLE_GATE_VALUE_METRICS = {
    "pixel_aupimo",
    "pixel_ap",
    "image_ap",
    "image_recall",
    "false_negatives",
}
LIFECYCLE_ALLOWED_METRICS = {
    "pixel_aupimo_1e-5_1e-3",
    "pixel_aupimo",
    "pixel_ap",
    "image_ap",
    "false_negatives",
    "events_processed",
    "cycles_completed",
    "localization_metric_delta",
    "classification_metric_delta",
    "classification_fn_delta",
    "gate_metric_delta",
    "gate_fn_delta",
}
LIFECYCLE_SENSITIVE_KEYS = (
    "image",
    "path",
    "uri",
    "mask",
    "heatmap",
    "piece_event",
    "relative",
)

DRIFT_ACTIVE_MODEL_ALLOWED_FIELDS = {
    "version",
    "registry_model_name",
    "registered_model_version",
    "registry_stage",
    "runtime_contract_status",
}
DRIFT_ALLOWED_METRICS = {
    "drift_score",
    "window_events",
    "window_index",
    "first_confirmed_window_index",
    "alert_rate",
    "red_rate",
    "unexpected_red_rate",
    "roi_fail_rate",
    "oracle_fn_rate",
    "domain_ratio",
    "domain_score",
    "degradation_score",
}
DRIFT_ALLOWED_STATUSES = {"clear", "suspected", "confirmed"}
DRIFT_ALLOWED_MODEL_ROLES = {"classification", "localization"}

_SENSITIVE_VALUE_MARKERS = (
    "://",
    ":\\",
    "\\\\",
    "/.cache/",
    "\\.cache\\",
    "/data/",
    "\\data\\",
    "/models/",
    "\\models\\",
)


class ObservabilityRejection(Exception):
    """A lifecycle/drift event was rejected by the exposition sanitiser.

    Carries the fields the API needs to render an HTTP error so the module
    stays free of any FastAPI dependency.
    """

    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        message: str,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.reason = reason


def _reject(
    *,
    status_code: int,
    error_code: str,
    message: str,
    reason: str | None = None,
) -> None:
    raise ObservabilityRejection(
        status_code=status_code,
        error_code=error_code,
        message=message,
        reason=reason,
    )


# --- Prometheus label formatting (shared with the API filtered metrics) -------


def _metric_label_value(value: Any) -> str:
    if value is None or value == "":
        value = "unknown"
    text = str(value)
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def metric_labels(labels: tuple[tuple[str, Any], ...]) -> str:
    """Render ``key="value"`` Prometheus label pairs from a label tuple."""
    return ",".join(f'{name}="{_metric_label_value(value)}"' for name, value in labels)


def _cycle_number(cycle_id: Any) -> int:
    text = str(cycle_id or "")
    if text.startswith("cycle_"):
        text = text.rsplit("_", 1)[-1]
    try:
        return int(text)
    except ValueError:
        return 0


def _finite_metric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def _observability_is_recent(updated_at: Any) -> bool:
    if OBSERVABILITY_TRANSIENT_TTL_SECONDS <= 0:
        return True
    try:
        timestamp = float(updated_at)
    except (TypeError, ValueError):
        return False
    return time.time() - timestamp <= OBSERVABILITY_TRANSIENT_TTL_SECONDS


def _lifecycle_base_labels(current: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return (
        ("scenario_id", current.get("scenario_id")),
        ("lifecycle_run_id", current.get("lifecycle_run_id")),
        ("cycle_id", current.get("cycle_id")),
        ("candidate_version", current.get("candidate_version")),
        ("candidate_init_policy", current.get("candidate_init_policy")),
    )


# --- Lifecycle metric-name parsing -------------------------------------------


def _is_allowed_lifecycle_metric_name(name: str) -> bool:
    if name in LIFECYCLE_ALLOWED_METRICS:
        return True
    if _parse_gate_value_metric_name(name) is not None:
        return True
    if _parse_gate_delta_metric_name(name) is not None:
        return True
    return False


def _parse_gate_value_metric_name(name: str) -> tuple[str, str, str] | None:
    prefix = "gate_"
    if not name.startswith(prefix):
        return None
    remainder = name[len(prefix) :]
    for role in ("localization", "classification"):
        role_prefix = f"{role}_"
        if not remainder.startswith(role_prefix):
            continue
        model_and_metric = remainder[len(role_prefix) :]
        for model in ("active", "candidate"):
            model_prefix = f"{model}_"
            if not model_and_metric.startswith(model_prefix):
                continue
            metric = model_and_metric[len(model_prefix) :]
            if metric in LIFECYCLE_GATE_VALUE_METRICS:
                return role, model, metric
    return None


def _parse_gate_delta_metric_name(name: str) -> tuple[str, str] | None:
    prefix = "gate_delta_"
    if not name.startswith(prefix):
        return None
    remainder = name[len(prefix) :]
    for role in ("localization", "classification"):
        role_prefix = f"{role}_"
        if not remainder.startswith(role_prefix):
            continue
        metric = remainder[len(role_prefix) :]
        if metric in LIFECYCLE_GATE_VALUE_METRICS:
            return role, metric
    return None


def _lifecycle_gate_labels(
    event: LifecycleEventRequest,
    *,
    role: str,
    model: str,
    metric_name: str,
) -> tuple[tuple[str, Any], ...]:
    return (
        ("scenario_id", event.scenario_id),
        ("lifecycle_run_id", event.lifecycle_run_id),
        ("cycle_id", event.cycle_id or "unknown"),
        ("role", role),
        ("model", model),
        ("metric", metric_name),
    )


def _lifecycle_gate_delta_labels(
    event: LifecycleEventRequest,
    *,
    role: str,
    metric_name: str,
) -> tuple[tuple[str, Any], ...]:
    return (
        ("scenario_id", event.scenario_id),
        ("lifecycle_run_id", event.lifecycle_run_id),
        ("cycle_id", event.cycle_id or "unknown"),
        ("role", role),
        ("metric", metric_name),
    )


# --- Sanitisation (security governance invariant) ----------------------------


def _contains_sensitive_value(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _SENSITIVE_VALUE_MARKERS)


def _reject_sensitive_lifecycle_payload(payload: dict[str, Any]) -> None:
    def visit(value: Any, *, key: str = "") -> None:
        lowered_key = key.lower()
        if any(marker in lowered_key for marker in LIFECYCLE_SENSITIVE_KEYS):
            _reject(
                status_code=422,
                error_code="sensitive_lifecycle_field",
                message="Lifecycle event contains a sensitive or non-observable field.",
                reason=f"Lifecycle field {key!r} is not accepted by the metrics API.",
            )
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                if key == "metrics" and not _is_allowed_lifecycle_metric_name(str(child_key)):
                    _reject(
                        status_code=422,
                        error_code="unsupported_lifecycle_metric",
                        message="Lifecycle event contains an unsupported metric.",
                        reason=f"Metric {child_key!r} is not accepted by the metrics API.",
                    )
                visit(child_value, key="" if key == "metrics" else str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key=key)
            return
        if isinstance(value, str) and _contains_sensitive_value(value):
            _reject(
                status_code=422,
                error_code="sensitive_lifecycle_value",
                message="Lifecycle event contains a sensitive value.",
                reason=f"Lifecycle value for {key!r} looks like a path or URI.",
            )

    visit(payload)


def _reject_sensitive_drift_payload(payload: dict[str, Any]) -> None:
    def visit(value: Any, *, key: str = "") -> None:
        lowered_key = key.lower()
        if any(marker in lowered_key for marker in LIFECYCLE_SENSITIVE_KEYS):
            _reject(
                status_code=422,
                error_code="sensitive_drift_field",
                message="Drift event contains a sensitive or non-observable field.",
                reason=f"Drift field {key!r} is not accepted by the metrics API.",
            )
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                if key == "metrics" and child_key not in DRIFT_ALLOWED_METRICS:
                    _reject(
                        status_code=422,
                        error_code="unsupported_drift_metric",
                        message="Drift event contains an unsupported metric.",
                        reason=f"Metric {child_key!r} is not accepted by the metrics API.",
                    )
                visit(child_value, key="" if key == "metrics" else str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                visit(item, key=key)
            return
        if isinstance(value, str) and _contains_sensitive_value(value):
            _reject(
                status_code=422,
                error_code="sensitive_drift_value",
                message="Drift event contains a sensitive value.",
                reason=f"Drift value for {key!r} looks like a path or URI.",
            )

    visit(payload)


def _new_lifecycle_state() -> dict[str, Any]:
    return {
        "current": {},
        "epoch_metrics": {},
        "epoch_updated_at": 0.0,
        "gate_metrics": {},
        "gate_values": {},
        "gate_deltas": {},
        "active_models": {},
        "final_models": {},
        "summary_metrics": {},
        "promotion_decisions": {},
        "promotion_seen": set(),
        "promotion_counters": {},
    }


def _new_drift_state() -> dict[str, Any]:
    return {"current": {}}


class ObservabilityExposition:
    """Single owner of the lifecycle/drift exposition state.

    Holds ``LIFECYCLE_STATE`` and ``DRIFT_STATE`` behind a narrow interface. The
    API keeps one instance; tests reset it instead of clearing parallel globals.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._lifecycle: dict[str, Any] = _new_lifecycle_state()
        self._drift: dict[str, Any] = _new_drift_state()

    # -- ingestion ------------------------------------------------------------

    def record_lifecycle_event(self, event: LifecycleEventRequest) -> None:
        payload = event.model_dump(mode="json", exclude_none=True)
        _reject_sensitive_lifecycle_payload(payload)
        current = self._lifecycle["current"]
        now = time.time()
        current["_updated_at"] = now
        for key in (
            "scenario_id",
            "lifecycle_run_id",
            "cycle_id",
            "candidate_version",
            "candidate_init_policy",
        ):
            if payload.get(key) is not None:
                current[key] = payload[key]
        if event.epoch is not None:
            current["epoch"] = event.epoch
        metrics_payload = {
            key: value
            for key, value in (payload.get("metrics") or {}).items()
            if _is_allowed_lifecycle_metric_name(str(key)) and _finite_metric(value) is not None
        }
        if event.event_type == "epoch_completed":
            epoch_metrics = self._lifecycle["epoch_metrics"]
            epoch_metrics.clear()
            self._lifecycle["epoch_updated_at"] = now
            for source_name, value in metrics_payload.items():
                normalized_name = LIFECYCLE_EPOCH_METRIC_ALIASES.get(source_name)
                if normalized_name is not None:
                    epoch_metrics[normalized_name] = value
        if event.event_type in {"gate_decision", "promotion_decision"}:
            gate_metrics = self._lifecycle["gate_metrics"]
            if "localization_metric_delta" in metrics_payload:
                gate_metrics.setdefault("localization", {})["metric_delta"] = metrics_payload["localization_metric_delta"]
            if "classification_metric_delta" in metrics_payload:
                gate_metrics.setdefault("classification", {})["metric_delta"] = metrics_payload["classification_metric_delta"]
            if "classification_fn_delta" in metrics_payload:
                gate_metrics.setdefault("classification", {})["fn_delta"] = metrics_payload["classification_fn_delta"]
            for name, value in metrics_payload.items():
                parsed_gate_value = _parse_gate_value_metric_name(str(name))
                if parsed_gate_value is not None:
                    role, model, metric_name = parsed_gate_value
                    labels = _lifecycle_gate_labels(event, role=role, model=model, metric_name=metric_name)
                    self._lifecycle["gate_values"][labels] = value
                    continue
                parsed_gate_delta = _parse_gate_delta_metric_name(str(name))
                if parsed_gate_delta is not None:
                    role, metric_name = parsed_gate_delta
                    labels = _lifecycle_gate_delta_labels(event, role=role, metric_name=metric_name)
                    self._lifecycle["gate_deltas"][labels] = value
            self._record_lifecycle_promotion_status(event)
        summary_metrics = self._lifecycle["summary_metrics"]
        for name in ("events_processed", "cycles_completed"):
            if name in metrics_payload:
                summary_metrics[name] = metrics_payload[name]
        active_models = self._lifecycle["active_models"]
        if event.active_classification_model_version:
            active_models["classification"] = event.active_classification_model_version
        if event.active_localization_model_version:
            active_models["localization"] = event.active_localization_model_version
        if event.candidate_initial_model_version and not active_models:
            active_models.setdefault("classification", event.candidate_initial_model_version)
            active_models.setdefault("localization", event.candidate_initial_model_version)
        self._record_lifecycle_final_models(event)

    def _record_lifecycle_final_models(self, event: LifecycleEventRequest) -> None:
        final_models = self._lifecycle["final_models"]
        specs = (
            (
                "classification",
                event.active_classification_model_version,
                event.active_classification_registered_model_name,
                event.active_classification_registered_model_version,
            ),
            (
                "localization",
                event.active_localization_model_version,
                event.active_localization_registered_model_name,
                event.active_localization_registered_model_version,
            ),
        )
        for role, version, model_name, model_version in specs:
            if not version:
                continue
            existing = final_models.get(role, {})
            final_models[role] = {
                "version": version,
                "registered_model_name": model_name or existing.get("registered_model_name", ""),
                "registered_model_version": model_version or existing.get("registered_model_version", ""),
            }

    def _record_lifecycle_promotion_status(self, event: LifecycleEventRequest) -> None:
        for role, status in (
            ("localization", event.localization_promotion_status),
            ("classification", event.classification_promotion_status),
        ):
            if not status:
                continue
            key = (event.lifecycle_run_id, event.cycle_id or "unknown", role, status)
            seen: set[tuple[str, str, str, str]] = self._lifecycle["promotion_seen"]
            if key not in seen:
                seen.add(key)
                counters = self._lifecycle["promotion_counters"]
                labels = (
                    ("scenario_id", event.scenario_id),
                    ("lifecycle_run_id", event.lifecycle_run_id),
                    ("role", role),
                    ("status", status),
                )
                counters[labels] = counters.get(labels, 0) + 1
            if status == "promoted" and event.candidate_version:
                self._lifecycle["active_models"][role] = event.candidate_version
            decision_labels = (
                ("scenario_id", event.scenario_id),
                ("lifecycle_run_id", event.lifecycle_run_id),
                ("cycle_id", event.cycle_id or "unknown"),
                ("role", role),
                ("status", status),
                ("candidate_version", event.candidate_version or ""),
            )
            self._lifecycle["promotion_decisions"][decision_labels] = 1

    def record_drift_event(self, event: DriftEventRequest) -> None:
        payload = event.model_dump(mode="json", exclude_none=True)
        _reject_sensitive_drift_payload(payload)
        status = event.status if event.status in DRIFT_ALLOWED_STATUSES else "clear"
        metrics_payload = {
            key: float(value)
            for key, value in (payload.get("metrics") or {}).items()
            if key in DRIFT_ALLOWED_METRICS and _finite_metric(value) is not None
        }
        if event.window_events is not None:
            metrics_payload["window_events"] = float(event.window_events)
        if event.window_index is not None:
            metrics_payload["window_index"] = float(event.window_index)
        if event.first_confirmed_window_index is not None:
            metrics_payload["first_confirmed_window_index"] = float(event.first_confirmed_window_index)
        active_models: dict[str, dict[str, str]] = {}
        for role, model_payload in (event.active_models or {}).items():
            if role not in DRIFT_ALLOWED_MODEL_ROLES:
                continue
            active_models[role] = {
                key: str(value)
                for key, value in model_payload.items()
                if key in DRIFT_ACTIVE_MODEL_ALLOWED_FIELDS and value not in {None, ""}
            }
        self._drift["current"] = {
            "event_type": event.event_type,
            "scenario_id": event.scenario_id,
            "status": status,
            "source_domain": event.source_domain,
            "lifecycle_run_id": event.lifecycle_run_id or "",
            "cycle_id": event.cycle_id or "",
            "updated_at": time.time(),
            "trigger_lifecycle": bool(event.trigger_lifecycle),
            "active_models": active_models,
            "metrics": metrics_payload,
        }

    # -- rendering ------------------------------------------------------------

    def render_prometheus_lines(self) -> list[str]:
        return self._drift_metrics_lines() + self._lifecycle_metrics_lines()

    def _lifecycle_metrics_lines(self) -> list[str]:
        current = dict(self._lifecycle["current"])
        lines = [
            "# HELP iqa_lifecycle_cycle_current Current IQA lifecycle cycle observed by the API",
            "# TYPE iqa_lifecycle_cycle_current gauge",
            "# HELP iqa_lifecycle_epoch_current Current IQA lifecycle training epoch observed by the API",
            "# TYPE iqa_lifecycle_epoch_current gauge",
            "# HELP iqa_lifecycle_epoch_pixel_aupimo Latest lifecycle epoch pixel AUPIMO",
            "# TYPE iqa_lifecycle_epoch_pixel_aupimo gauge",
            "# HELP iqa_lifecycle_epoch_pixel_ap Latest lifecycle epoch pixel AP",
            "# TYPE iqa_lifecycle_epoch_pixel_ap gauge",
            "# HELP iqa_lifecycle_epoch_image_ap Latest lifecycle epoch image AP",
            "# TYPE iqa_lifecycle_epoch_image_ap gauge",
            "# HELP iqa_lifecycle_epoch_metric Latest lifecycle epoch metric by metric label",
            "# TYPE iqa_lifecycle_epoch_metric gauge",
            "# HELP iqa_lifecycle_gate_metric_delta Latest lifecycle gate metric delta by role",
            "# TYPE iqa_lifecycle_gate_metric_delta gauge",
            "# HELP iqa_lifecycle_gate_fn_delta Latest lifecycle classification false negative delta",
            "# TYPE iqa_lifecycle_gate_fn_delta gauge",
            "# HELP iqa_lifecycle_gate_value Lifecycle gate active/candidate metric value",
            "# TYPE iqa_lifecycle_gate_value gauge",
            "# HELP iqa_lifecycle_gate_delta Lifecycle gate metric delta by role and metric",
            "# TYPE iqa_lifecycle_gate_delta gauge",
            "# HELP iqa_lifecycle_promotion_total IQA lifecycle promotion decisions by role and status",
            "# TYPE iqa_lifecycle_promotion_total counter",
            "# HELP iqa_lifecycle_promotion_decision_info Latest lifecycle promotion decisions by role",
            "# TYPE iqa_lifecycle_promotion_decision_info gauge",
            "# HELP iqa_lifecycle_active_model_info Active lifecycle model versions observed by the API",
            "# TYPE iqa_lifecycle_active_model_info gauge",
            "# HELP iqa_lifecycle_final_model_info Final promoted model versions for the lifecycle run",
            "# TYPE iqa_lifecycle_final_model_info gauge",
            "# HELP iqa_lifecycle_run_events_processed Lifecycle run events processed",
            "# TYPE iqa_lifecycle_run_events_processed gauge",
            "# HELP iqa_lifecycle_run_cycles_completed Lifecycle run cycles completed",
            "# TYPE iqa_lifecycle_run_cycles_completed gauge",
        ]
        if current:
            base_labels = _lifecycle_base_labels(current)
            lines.append(f"iqa_lifecycle_cycle_current{{{metric_labels(base_labels)}}} {_cycle_number(current.get('cycle_id'))}")
            epoch_recent = _observability_is_recent(self._lifecycle.get("epoch_updated_at"))
            if current.get("epoch") is not None and epoch_recent:
                lines.append(f"iqa_lifecycle_epoch_current{{{metric_labels(base_labels)}}} {int(current['epoch'])}")
            epoch_metrics = self._lifecycle["epoch_metrics"] if epoch_recent else {}
            for metric_name, value in sorted(epoch_metrics.items()):
                finite_value = _finite_metric(value)
                if finite_value is not None:
                    labels = base_labels + (("metric", metric_name),)
                    lines.append(f"iqa_lifecycle_epoch_metric{{{metric_labels(labels)}}} {finite_value}")
            epoch_specs = (
                ("pixel_aupimo", "iqa_lifecycle_epoch_pixel_aupimo"),
                ("pixel_ap", "iqa_lifecycle_epoch_pixel_ap"),
                ("image_ap", "iqa_lifecycle_epoch_image_ap"),
            )
            for source_name, metric_name in epoch_specs:
                value = _finite_metric(epoch_metrics.get(source_name))
                if value is not None:
                    lines.append(f"{metric_name}{{{metric_labels(base_labels)}}} {value}")
            for role, role_metrics in sorted(self._lifecycle["gate_metrics"].items()):
                labels = base_labels + (("role", role),)
                metric_delta = _finite_metric(role_metrics.get("metric_delta"))
                if metric_delta is not None:
                    lines.append(f"iqa_lifecycle_gate_metric_delta{{{metric_labels(labels)}}} {metric_delta}")
                fn_delta = _finite_metric(role_metrics.get("fn_delta"))
                if fn_delta is not None:
                    lines.append(f"iqa_lifecycle_gate_fn_delta{{{metric_labels(labels)}}} {fn_delta}")
            for labels, value in sorted(self._lifecycle["gate_values"].items(), key=lambda item: str(item[0])):
                finite_value = _finite_metric(value)
                if finite_value is not None:
                    lines.append(f"iqa_lifecycle_gate_value{{{metric_labels(labels)}}} {finite_value}")
            for labels, value in sorted(self._lifecycle["gate_deltas"].items(), key=lambda item: str(item[0])):
                finite_value = _finite_metric(value)
                if finite_value is not None:
                    lines.append(f"iqa_lifecycle_gate_delta{{{metric_labels(labels)}}} {finite_value}")
            for role, version in sorted(self._lifecycle["active_models"].items()):
                labels = (
                    ("scenario_id", current.get("scenario_id")),
                    ("role", role),
                    ("version", version),
                    ("run_id", current.get("lifecycle_run_id")),
                    ("cycle_id", current.get("cycle_id")),
                )
                lines.append(f"iqa_lifecycle_active_model_info{{{metric_labels(labels)}}} 1")
            for role, model_payload in sorted(self._lifecycle["final_models"].items()):
                labels = (
                    ("scenario_id", current.get("scenario_id")),
                    ("lifecycle_run_id", current.get("lifecycle_run_id")),
                    ("role", role),
                    ("version", model_payload.get("version", "")),
                    ("registered_model_version", model_payload.get("registered_model_version", "")),
                    ("registered_model_name", model_payload.get("registered_model_name", "")),
                )
                lines.append(f"iqa_lifecycle_final_model_info{{{metric_labels(labels)}}} 1")
            summary_metrics = self._lifecycle["summary_metrics"]
            events_processed = _finite_metric(summary_metrics.get("events_processed"))
            if events_processed is not None:
                lines.append(f"iqa_lifecycle_run_events_processed{{{metric_labels(base_labels)}}} {events_processed}")
            cycles_completed = _finite_metric(summary_metrics.get("cycles_completed"))
            if cycles_completed is not None:
                lines.append(f"iqa_lifecycle_run_cycles_completed{{{metric_labels(base_labels)}}} {cycles_completed}")
        for labels, value in sorted(self._lifecycle["promotion_counters"].items(), key=lambda item: str(item[0])):
            lines.append(f"iqa_lifecycle_promotion_total{{{metric_labels(labels)}}} {value}")
        for labels, value in sorted(self._lifecycle["promotion_decisions"].items(), key=lambda item: str(item[0])):
            lines.append(f"iqa_lifecycle_promotion_decision_info{{{metric_labels(labels)}}} {value}")
        return lines

    def _drift_metrics_lines(self) -> list[str]:
        current = dict(self._drift["current"])
        lines = [
            "# HELP iqa_drift_score Current IQA drift score received by the API",
            "# TYPE iqa_drift_score gauge",
            "# HELP iqa_drift_status Current IQA drift status as a one-hot gauge",
            "# TYPE iqa_drift_status gauge",
            "# HELP iqa_drift_window_events Number of events in the current drift window",
            "# TYPE iqa_drift_window_events gauge",
            "# HELP iqa_drift_window_index Current drift observation window index",
            "# TYPE iqa_drift_window_index gauge",
            "# HELP iqa_drift_first_confirmed_window First window index where drift was confirmed",
            "# TYPE iqa_drift_first_confirmed_window gauge",
            "# HELP iqa_drift_alert_rate Alert decision rate in the current drift window",
            "# TYPE iqa_drift_alert_rate gauge",
            "# HELP iqa_drift_red_rate Red decision rate in the current drift window",
            "# TYPE iqa_drift_red_rate gauge",
            "# HELP iqa_drift_unexpected_red_rate Red decision rate on conforming pieces in the current drift window",
            "# TYPE iqa_drift_unexpected_red_rate gauge",
            "# HELP iqa_drift_roi_fail_rate ROI failure rate in the current drift window",
            "# TYPE iqa_drift_roi_fail_rate gauge",
            "# HELP iqa_drift_oracle_fn_rate Oracle false-negative rate in the current drift window",
            "# TYPE iqa_drift_oracle_fn_rate gauge",
            "# HELP iqa_drift_domain_ratio Source-domain ratio in the current drift window",
            "# TYPE iqa_drift_domain_ratio gauge",
            "# HELP iqa_drift_degradation_score Model degradation score independent of source-domain ratio",
            "# TYPE iqa_drift_degradation_score gauge",
            "# HELP iqa_drift_domain_score Source-domain drift score component",
            "# TYPE iqa_drift_domain_score gauge",
            "# HELP iqa_drift_trigger_lifecycle Whether the observed drift window should trigger lifecycle correction",
            "# TYPE iqa_drift_trigger_lifecycle gauge",
            "# HELP iqa_drift_active_model_info Active models used by the drift observation replay",
            "# TYPE iqa_drift_active_model_info gauge",
        ]
        if not current:
            return lines
        if not _observability_is_recent(current.get("updated_at")):
            return lines

        scenario_id = current.get("scenario_id")
        source_domain = current.get("source_domain") or "piece_a_p4"
        status = current.get("status") if current.get("status") in DRIFT_ALLOWED_STATUSES else "clear"
        metrics_payload = current.get("metrics") or {}
        base_labels = (("scenario_id", scenario_id), ("source_domain", source_domain))
        for status_name in ("clear", "suspected", "confirmed"):
            labels = base_labels + (("status", status_name),)
            lines.append(f"iqa_drift_status{{{metric_labels(labels)}}} {1 if status == status_name else 0}")
        lines.append(
            f"iqa_drift_trigger_lifecycle{{{metric_labels(base_labels)}}} "
            f"{1 if current.get('trigger_lifecycle') else 0}"
        )
        metric_specs = (
            ("drift_score", "iqa_drift_score"),
            ("window_events", "iqa_drift_window_events"),
            ("window_index", "iqa_drift_window_index"),
            ("first_confirmed_window_index", "iqa_drift_first_confirmed_window"),
            ("alert_rate", "iqa_drift_alert_rate"),
            ("red_rate", "iqa_drift_red_rate"),
            ("unexpected_red_rate", "iqa_drift_unexpected_red_rate"),
            ("roi_fail_rate", "iqa_drift_roi_fail_rate"),
            ("oracle_fn_rate", "iqa_drift_oracle_fn_rate"),
            ("domain_ratio", "iqa_drift_domain_ratio"),
            ("domain_score", "iqa_drift_domain_score"),
            ("degradation_score", "iqa_drift_degradation_score"),
        )
        for source_name, metric_name in metric_specs:
            value = _finite_metric(metrics_payload.get(source_name))
            if value is not None:
                lines.append(f"{metric_name}{{{metric_labels(base_labels)}}} {value}")
        for role, model_payload in sorted((current.get("active_models") or {}).items()):
            labels = base_labels + (
                ("role", role),
                ("version", model_payload.get("version", "")),
                ("registry_model_name", model_payload.get("registry_model_name", "")),
                ("registered_model_version", model_payload.get("registered_model_version", "")),
                ("registry_stage", model_payload.get("registry_stage", "")),
                ("runtime_contract_status", model_payload.get("runtime_contract_status", "")),
            )
            lines.append(f"iqa_drift_active_model_info{{{metric_labels(labels)}}} 1")
        return lines
