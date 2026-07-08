from __future__ import annotations

from scripts.audit_gate_panel import DatasetSummary, warning_lines


def test_gate_audit_flags_small_image_level_panel() -> None:
    gate = DatasetSummary(
        "gate",
        [
            {
                "group_key": "good_a",
                "source_class": "Casting_class1",
                "label": "good",
                "is_defective": "False",
                "n_images": "1",
            },
            {
                "group_key": "defect_a",
                "source_class": "Casting_class1",
                "label": "defective",
                "is_defective": "True",
                "n_images": "1",
            },
        ],
        "group_key",
    )
    replay = DatasetSummary(
        "replay",
        [
            {
                "source_group_key": "good_replay",
                "source_class": "Casting_class1",
                "label": "good",
                "is_defective": "False",
                "n_images": "3",
            },
            {
                "source_group_key": "defect_replay",
                "source_class": "Casting_class2",
                "label": "defective",
                "is_defective": "True",
                "n_images": "3",
            },
        ],
        "source_group_key",
    )

    warnings = warning_lines(gate, replay, None)

    assert any("only 1 good rows" in warning for warning in warnings)
    assert any("image-level only" in warning for warning in warnings)
    assert any("misses 1/1 replay defect" in warning for warning in warnings)


def test_gate_audit_allows_deliberate_disjoint_holdout() -> None:
    gate = DatasetSummary(
        "gate",
        [
            {
                "group_key": "defect_holdout",
                "source_class": "Casting_class1",
                "label": "defective",
                "is_defective": "True",
                "n_images": "3",
            }
        ],
        "group_key",
    )
    replay = DatasetSummary(
        "replay",
        [
            {
                "source_group_key": "defect_train",
                "source_class": "Casting_class1",
                "label": "defective",
                "is_defective": "True",
                "n_images": "3",
            }
        ],
        "source_group_key",
    )

    warnings = warning_lines(gate, replay, None, expect_disjoint_holdout=True)

    assert not any("misses" in warning for warning in warnings)
    assert not any("Low group overlap" in warning for warning in warnings)
