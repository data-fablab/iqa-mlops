# IQA Phase 2 Feedback Rules

## Purpose

This document defines the Phase 2 feedback rules for IQA.

The main goal is to separate display feedback from training eligibility and to prevent feedback poisoning.

## Core rule

oracle_gt is the sovereign source for training eligibility.

human_sophie is a display feedback source only.

## Feedback sources

| Source | Accepted | Closes feedback | Can make train eligible | Purpose |
| --- | --- | --- | --- | --- |
| oracle_gt | Yes | Yes | Yes, if safe | Automated MVP ground truth |
| human_sophie | Yes | No | No | Display and review workflow |

## Human Sophie rule

human_sophie feedback is accepted only for display.

Expected state:

| Field | Value |
| --- | --- |
| feedback_closed | false |
| display_decision_source | human_sophie |
| train_eligibility_source | oracle_gt |
| eligible_for_train | false |
| train_block_reason | human_sophie_display_only |

This prevents a display action from poisoning the training candidate set.

## Oracle GT rule

oracle_gt feedback closes the feedback loop.

Expected state:

| Field | Value |
| --- | --- |
| feedback_closed | true |
| train_eligibility_source | oracle_gt |
| eligible_for_train | Computed from oracle and feedback status |

Oracle verdict is derived from gt_mask_has_defect.

| gt_mask_has_defect | Oracle verdict |
| --- | --- |
| true | defective |
| false | conforme |

## Replay protection

A prediction can be closed only once by oracle feedback.

A second oracle feedback on the same closed prediction is rejected.

Expected result:

| Field | Value |
| --- | --- |
| HTTP status | 409 |
| error_code | feedback_already_closed |
| incident_type | invalid_prediction_request |

## Prediction id protection

Feedback must reference an existing prediction.

Unknown prediction feedback is rejected before mutating feedback state.

Expected result:

| Field | Value |
| --- | --- |
| HTTP status | 404 |
| error_code | prediction_not_found |
| incident_type | invalid_prediction_request |

## Piece event mismatch protection

The feedback piece_event_id must match the prediction piece_event_id.

Expected result:

| Field | Value |
| --- | --- |
| HTTP status | 409 |
| error_code | feedback_piece_event_mismatch |
| incident_type | feedback_conflict |

A structured feedback_conflict incident is created.

## Scenario mismatch protection

The feedback scenario_id must match the prediction scenario_id.

Expected result:

| Field | Value |
| --- | --- |
| HTTP status | 409 |
| error_code | feedback_scenario_mismatch |
| incident_type | feedback_conflict |

A structured feedback_conflict incident is created.

## Unknown feedback source

Unsupported feedback sources are rejected.

Expected result:

| Field | Value |
| --- | --- |
| HTTP status | 400 |
| error_code | unknown_feedback_source |
| incident_type | invalid_prediction_request |

## Training eligibility rules

Training eligibility is blocked when the oracle detects a defect or when the feedback status indicates unsafe training data.

| Condition | Eligible for train | Block reason |
| --- | --- | --- |
| gt_mask_has_defect true | No | oracle_gt_defective |
| feedback_status defaut_confirme | No | feedback_status_defaut_confirme |
| feedback_status faux_negatif | No | feedback_status_faux_negatif |
| feedback_status roi_warning | No | roi_warning |
| feedback_status roi_fail | No | roi_fail |
| safe oracle conforming feedback | Yes | Empty |

## False negative rule

A false negative is detected when the model decision is Vert and the oracle verdict is defective.

Governance outcome:

| Field | Value |
| --- | --- |
| divergence | faux_negatif |
| eligible_for_train | false |
| incident_type | false_negative |
| severity | high |

## ROI warning rule

roi_warning blocks training eligibility.

It does not create a roi_fail incident.

## ROI fail rule

roi_fail blocks training eligibility.

It creates a high severity roi_fail incident.

## Metrics impacted by feedback

Feedback rules affect:

| Metric |
| --- |
| iqa_feedback_conflict_total |
| iqa_ai_security_incident_total |
| iqa_unsafe_train_blocked_total |
| iqa_invalid_feedback_total |
| iqa_divergence_filtered_total |
