"""Tests for the static retrain resolver A (Issue 9)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from iqa.datasets.retrain_resolver import (
    RetrainSample,
    RetrainTrigger,
    resolve_retrain_samples,
)

pytestmark = pytest.mark.unit

PLAN_HEADER = [
    "piece_event_id",
    "scenario_id",
    "scenario_phase",
    "source_class",
    "label",
    "relative_paths",
]


def _write_plan(tmp_path: Path, rows: list[dict]) -> Path:
    plan = tmp_path / "plan.csv"
    with plan.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PLAN_HEADER)
        writer.writeheader()
        writer.writerows(rows)
    return plan


def _rows() -> list[dict]:
    return [
        {"piece_event_id": "pe1", "scenario_id": "drift", "scenario_phase": "baseline_domain_class1", "source_class": "Casting_class1", "label": "good", "relative_paths": "c1/img1.jpg"},
        {"piece_event_id": "pe2", "scenario_id": "drift", "scenario_phase": "baseline_domain_class1", "source_class": "Casting_class1", "label": "defective", "relative_paths": "c1/img2.jpg"},
        {"piece_event_id": "pe3", "scenario_id": "drift", "scenario_phase": "baseline_domain_class1", "source_class": "Casting_class1", "label": "good", "relative_paths": "c1/img3a.jpg|c1/img3b.jpg"},
        {"piece_event_id": "pe4", "scenario_id": "drift", "scenario_phase": "domain_extension_class2", "source_class": "Casting_class2", "label": "good", "relative_paths": "c2/img4.jpg"},
        {"piece_event_id": "pe5", "scenario_id": "drift", "scenario_phase": "domain_extension_class2", "source_class": "Casting_class2", "label": "defective", "relative_paths": "c2/img5.jpg"},
        {"piece_event_id": "pe6", "scenario_id": "drift", "scenario_phase": "domain_extension_class3", "source_class": "Casting_class3", "label": "good", "relative_paths": "c3/img6.jpg"},
    ]


class TestRetrainTrigger:
    def test_round_trips_through_dict(self) -> None:
        trigger = RetrainTrigger(scenario_id="drift", triggering_class="Casting_class2", triggered_at="2026-06-26T12:00:00")
        assert RetrainTrigger.from_dict(trigger.to_dict()) == trigger

    def test_from_dict_minimal(self) -> None:
        trigger = RetrainTrigger.from_dict({"scenario_id": "drift", "triggering_class": "Casting_class2"})
        assert trigger.triggered_at is None


class TestResolveRetrainSamples:
    def test_class2_trigger_returns_class1_and_class2_good_only(self, tmp_path) -> None:
        plan = _write_plan(tmp_path, _rows())
        trigger = RetrainTrigger(scenario_id="drift", triggering_class="Casting_class2")
        samples = resolve_retrain_samples(trigger, plan_path=plan)
        classes = {s.source_class for s in samples}
        assert classes == {"Casting_class1", "Casting_class2"}
        assert all(s.label == "good" for s in samples)

    def test_class3_trigger_returns_all_three_classes(self, tmp_path) -> None:
        plan = _write_plan(tmp_path, _rows())
        trigger = RetrainTrigger(scenario_id="drift", triggering_class="Casting_class3")
        samples = resolve_retrain_samples(trigger, plan_path=plan)
        classes = {s.source_class for s in samples}
        assert classes == {"Casting_class1", "Casting_class2", "Casting_class3"}

    def test_class1_trigger_returns_class1_only(self, tmp_path) -> None:
        plan = _write_plan(tmp_path, _rows())
        trigger = RetrainTrigger(scenario_id="drift", triggering_class="Casting_class1")
        samples = resolve_retrain_samples(trigger, plan_path=plan)
        classes = {s.source_class for s in samples}
        assert classes == {"Casting_class1"}

    def test_excludes_defective_samples(self, tmp_path) -> None:
        plan = _write_plan(tmp_path, _rows())
        trigger = RetrainTrigger(scenario_id="drift", triggering_class="Casting_class3")
        samples = resolve_retrain_samples(trigger, plan_path=plan)
        assert all(s.label == "good" for s in samples)
        assert len(samples) == 5  # 3 from pe1+pe3(2imgs) class1, 1 class2, 1 class3

    def test_handles_multi_image_events(self, tmp_path) -> None:
        plan = _write_plan(tmp_path, _rows())
        trigger = RetrainTrigger(scenario_id="drift", triggering_class="Casting_class1")
        samples = resolve_retrain_samples(trigger, plan_path=plan)
        uris = [s.image_uri for s in samples]
        assert "c1/img3a.jpg" in uris
        assert "c1/img3b.jpg" in uris

    def test_image_root_prepends_path(self, tmp_path) -> None:
        plan = _write_plan(tmp_path, _rows())
        trigger = RetrainTrigger(scenario_id="drift", triggering_class="Casting_class1")
        root = Path("/data/raw")
        samples = resolve_retrain_samples(trigger, plan_path=plan, image_root=root)
        assert all(s.image_uri.startswith(str(root)) for s in samples)

    def test_unknown_class_falls_back_to_all_phases(self, tmp_path) -> None:
        plan = _write_plan(tmp_path, _rows())
        trigger = RetrainTrigger(scenario_id="drift", triggering_class="Unknown_class")
        samples = resolve_retrain_samples(trigger, plan_path=plan)
        classes = {s.source_class for s in samples}
        assert classes == {"Casting_class1", "Casting_class2", "Casting_class3"}

    def test_returns_retrain_sample_instances(self, tmp_path) -> None:
        plan = _write_plan(tmp_path, _rows())
        trigger = RetrainTrigger(scenario_id="drift", triggering_class="Casting_class2")
        samples = resolve_retrain_samples(trigger, plan_path=plan)
        assert all(isinstance(s, RetrainSample) for s in samples)
        assert all(s.scenario_phase in ("baseline_domain_class1", "domain_extension_class2") for s in samples)
