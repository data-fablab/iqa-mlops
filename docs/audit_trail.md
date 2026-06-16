# IQA Phase 2 Audit Trail

## Purpose

This document defines the Phase 2 audit trail from input image to prediction, feedback, incident, model version and metrics.

The goal is to make every decision explainable and traceable.

## Traceability chain

The Phase 2 traceability chain is:

    sha256
    piece_event_id
    scenario_id
    lot_id
    source_class
    dataset_version
    model_version
    roi_model_version
    prediction_id
    feedback
    divergence
    incident
    metrics

## Prediction trace

A prediction is created through /predict or /piece-events/{event_id}/predict.

The API stores:

| Field |
| --- |
| prediction_id |
| piece_event_id |
| scenario_id |
| image_uri |
| sha256 |
| lot_id |
| source_class |
| dataset_version |
| decision |
| model_version |
| roi_model_version |
| created_at |
| feedback_closed |

The response includes an audit object with the same core identifiers.

## Model trace

The active Feature AE and ROI model versions are traced through:

| Field |
| --- |
| model_version |
| roi_model_version |
| registered_model_name |
| source_of_truth |

MLflow Registry is the target source of truth for model promotion and reload.

Registered model naming rule:

    feature_ae__{scenario_id}

## Display feedback trace

human_sophie feedback is stored as display feedback.

Important fields:

| Field | Value |
| --- | --- |
| feedback_source | human_sophie |
| display_decision_source | human_sophie |
| train_eligibility_source | oracle_gt |
| eligible_for_train | false |
| train_block_reason | human_sophie_display_only |
| feedback_closed | false |

## Oracle feedback trace

oracle_gt feedback closes the prediction feedback loop.

Important fields:

| Field |
| --- |
| feedback_source |
| feedback_closed |
| verdict |
| display_decision_source |
| train_eligibility_source |
| eligible_for_train |
| train_block_reason |
| conflict_logged |
| closed_at |

## Prediction history route

/predictions returns rows that merge prediction state, display feedback, oracle feedback and audit trail.

The nested audit_trail contains prediction context and feedback context.

Prediction context:

| Field |
| --- |
| prediction_id |
| piece_event_id |
| scenario_id |
| lot_id |
| source_class |
| sha256 |
| dataset_version |
| model_version |
| roi_model_version |
| decision |

Feedback context:

| Field |
| --- |
| feedback_source |
| display_feedback_source |
| display_feedback_status |
| oracle_verdict |
| divergence |
| train_eligibility_source |
| eligible_for_train |
| train_block_reason |
| feedback_closed |
| conflict_logged |

## Divergence trace

| Model decision | Oracle verdict | Divergence |
| --- | --- | --- |
| Vert | defective | faux_negatif |
| Vert | conforme | concordant |
| Rouge | conforme | faux_positif |
| Rouge | defective | concordant |
| Orange | any closed oracle verdict | orange_a_revoir |

## Incident trace

Structured incidents connect security events to prediction and scenario context.

Incident fields:

| Field |
| --- |
| incident_id |
| incident_type |
| severity |
| piece_event_id |
| prediction_id |
| scenario_id |
| model_version |
| message |
| created_at |
| metadata |

Examples:

| Incident type | Trace link |
| --- | --- |
| feedback_conflict | Prediction id plus expected and received identifiers |
| false_negative | Prediction id plus oracle divergence |
| roi_fail | Prediction id plus train block reason |
| reload_refused | Scenario id plus reload event id |
| unsafe_train_candidate_blocked | Scenario id plus dataset version |

## Reload audit trace

Admin reload events are stored in ADMIN_RELOAD_LOG.

Fields:

| Field |
| --- |
| reload_event_id |
| prediction_id |
| scenario_id |
| stage |
| reload_status |
| accepted |
| reason |
| registered_model_name |
| source_of_truth |
| created_at |

Reload refusal also creates a structured reload_refused incident.

## Metrics trace

Global metrics include:

| Metric |
| --- |
| iqa_api_up |
| iqa_feedback_conflict_total |
| iqa_ai_security_incident_total |
| iqa_unsafe_train_blocked_total |
| iqa_invalid_feedback_total |
| iqa_reload_refused_total |
| iqa_prediction_total |
| iqa_roi_fail_total |
| iqa_predict_latency_seconds |
| iqa_active_model_info |

Filtered metrics include:

| Metric |
| --- |
| iqa_prediction_filtered_total |
| iqa_feedback_closed_filtered_total |
| iqa_train_eligible_filtered_total |
| iqa_divergence_filtered_total |

Filtered labels include scenario_id, lot_id, source_class, model_version, dataset_version, decision and divergence.

## Persistence note

Current Phase 2 implementation uses in memory stores for API governance testing.

The target production persistence remains PostgreSQL for facts, statuses, timestamps and URIs.

Heavy artifacts remain in MinIO.
