from __future__ import annotations

from scripts.analyze_replay_classification import confusion, threshold_at_allowed_good_alerts


def classification_quality_rows(events: list[dict[str, object]], *, group_key: str = "active_model_version") -> list[dict[str, object]]:
    from scripts.analyze_replay_classification import group_by

    rows = []
    for group, subset in group_by(events, group_key).items():
        row = {group_key: group}
        row.update(confusion(subset))
        rows.append(row)
    return rows


def test_confusion_counts_model_against_oracle() -> None:
    events = [
        {"oracle_verdict": "conforme", "decision": "green"},
        {"oracle_verdict": "conforme", "decision": "orange"},
        {"oracle_verdict": "defective", "decision": "green"},
        {"oracle_verdict": "defective", "decision": "red"},
    ]

    metrics = confusion(events)

    assert metrics["tp_defect_detected"] == 1
    assert metrics["fn_missed_defect"] == 1
    assert metrics["fp_good_alerted"] == 1
    assert metrics["tn_good_accepted"] == 1
    assert metrics["defect_recall"] == 0.5
    assert metrics["alert_precision"] == 0.5


def test_threshold_at_allowed_good_alerts_reveals_required_tradeoff() -> None:
    events = [
        {"oracle_verdict": "conforme", "score": 0.9},
        {"oracle_verdict": "conforme", "score": 0.3},
        {"oracle_verdict": "defective", "score": 0.8},
        {"oracle_verdict": "defective", "score": 0.2},
    ]

    strict = threshold_at_allowed_good_alerts(events, 0)
    two_good_alerts = threshold_at_allowed_good_alerts(events, 2)

    assert strict["detected"] == 0
    assert strict["fn"] == 2
    assert two_good_alerts["detected"] == 1
    assert two_good_alerts["fn"] == 1


def test_grouped_classification_quality_surfaces_model_regression() -> None:
    events = [
        {"active_model_version": "m1", "oracle_verdict": "defective", "decision": "red"},
        {"active_model_version": "m2", "oracle_verdict": "defective", "decision": "green"},
    ]

    rows = classification_quality_rows(events)

    assert rows[0]["active_model_version"] == "m1"
    assert rows[0]["defect_recall"] == 1.0
    assert rows[1]["active_model_version"] == "m2"
    assert rows[1]["defect_recall"] == 0.0
