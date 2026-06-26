"""Tests for the real drift driver's pure seam (Issue 7).

The ``relative_path -> file://`` resolution and the per-phase grouping/ordering are
pure and unit-testable; the rest (ratio ~ 0 on class1, the proxy firing on class2/3,
no state.json write) is verified live against the stack.
"""

from __future__ import annotations

import pytest

from scripts.run_real_drift_demo import (
    BASELINE_PHASE,
    Sample,
    load_phase_samples,
    ordered_phases,
    to_file_uri,
)

pytestmark = pytest.mark.unit


def test_to_file_uri_builds_container_path() -> None:
    uri = to_file_uri("Casting_class1/train/good/x_2_3.jpg", "/opt/iqa/iqa-mlops/data/raw/hss-iad")
    assert uri == "file:///opt/iqa/iqa-mlops/data/raw/hss-iad/Casting_class1/train/good/x_2_3.jpg"


def test_to_file_uri_normalizes_separators_and_slashes() -> None:
    # Trailing root slash, leading rel slash, and backslashes all normalize.
    uri = to_file_uri("\\sub\\img.jpg", "/opt/data/")
    assert uri == "file:///opt/data/sub/img.jpg"


def test_load_phase_samples_groups_first_view_per_row(tmp_path) -> None:
    plan = tmp_path / "plan.csv"
    plan.write_text(
        "scenario_phase,piece_event_id,relative_paths\n"
        "baseline_domain_class1,pe1,Casting_class1/a_1_2.jpg|Casting_class1/a_1_3.jpg\n"
        "baseline_domain_class1,pe2,Casting_class1/b_2_3.jpg\n"
        "domain_extension_class2,pe3,Casting_class2/c_2_3.jpg\n",
        encoding="utf-8",
    )
    by_phase = load_phase_samples(plan, "/opt/root")

    assert by_phase[BASELINE_PHASE] == [
        Sample("pe1", "file:///opt/root/Casting_class1/a_1_2.jpg"),  # first view only
        Sample("pe2", "file:///opt/root/Casting_class1/b_2_3.jpg"),
    ]
    assert by_phase["domain_extension_class2"] == [
        Sample("pe3", "file:///opt/root/Casting_class2/c_2_3.jpg"),
    ]


def test_load_phase_samples_skips_rows_without_phase_or_paths(tmp_path) -> None:
    plan = tmp_path / "plan.csv"
    plan.write_text(
        "scenario_phase,piece_event_id,relative_paths\n"
        "baseline_domain_class1,pe1,Casting_class1/a.jpg\n"
        ",pe2,Casting_class1/b.jpg\n"
        "domain_extension_class2,pe3,\n",
        encoding="utf-8",
    )
    by_phase = load_phase_samples(plan, "/opt/root")
    assert list(by_phase) == [BASELINE_PHASE]
    assert len(by_phase[BASELINE_PHASE]) == 1


def test_ordered_phases_canonical_baseline_first() -> None:
    present = {
        "domain_extension_class3": [Sample("a", "file:///r/3.jpg")],
        "domain_extension_class2": [Sample("b", "file:///r/2.jpg")],
        "baseline_domain_class1": [Sample("c", "file:///r/1.jpg")],
    }
    assert ordered_phases(present) == [
        "baseline_domain_class1",
        "domain_extension_class2",
        "domain_extension_class3",
    ]


def test_ordered_phases_appends_unknown_phases_last() -> None:
    present = {
        "domain_extension_class2": [Sample("b", "file:///r/2.jpg")],
        "mystery_phase": [Sample("z", "file:///r/z.jpg")],
        "baseline_domain_class1": [Sample("c", "file:///r/1.jpg")],
    }
    assert ordered_phases(present) == [
        "baseline_domain_class1",
        "domain_extension_class2",
        "mystery_phase",
    ]
