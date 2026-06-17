from __future__ import annotations

from iqa.feedback import OracleFeedbackRequest, oracle_gt_verdict


def test_oracle_gt_maps_empty_mask_to_conforme_and_train_eligible() -> None:
    verdict = oracle_gt_verdict(
        OracleFeedbackRequest(
            piece_event_id="piece_oracle_conforme",
            scenario_id="validation_set_v001",
            gt_mask_has_defect=False,
        )
    ).to_dict()

    assert verdict["feedback_source"] == "oracle_gt"
    assert verdict["verdict"] == "conforme"
    assert verdict["train_eligible"] is True


def test_oracle_gt_maps_non_empty_mask_to_defective_and_blocks_train() -> None:
    verdict = oracle_gt_verdict(
        OracleFeedbackRequest(
            piece_event_id="piece_oracle_defective",
            scenario_id="validation_set_v001",
            gt_mask_uri="s3://iqa/gt/piece_oracle_defective.png",
            gt_mask_has_defect=True,
        )
    ).to_dict()

    assert verdict["feedback_source"] == "oracle_gt"
    assert verdict["verdict"] == "defective"
    assert verdict["train_eligible"] is False
