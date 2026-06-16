# IQA Phase 2 API Contracts

## Purpose

This document defines the Phase 2 API contracts for the Industrial Quality Assistant project.

The Phase 2 API exposes prediction, feedback, traceability, incidents, model version, reload and monitoring contracts. These contracts support Streamlit review views, Airflow lifecycle tasks, MLflow model governance, Prometheus metrics and AI security auditability.

## Validation baseline

The contracts documented here are aligned with the Phase 2 validation suite after NAT16.

Validation evidence:

    95 passed
    SERVER_NAT16_AI_SECURITY_GOVERNANCE_OK

## Main routes

| Route | Method | Purpose |
| --- | --- | --- |
| /health | GET | API health check |
| /model/version | GET | Active model metadata scoped by scenario |
| /predict | POST | Create a prediction and audit trace |
| /piece-events/{event_id}/predict | POST | Predict using the path event id as piece event id |
| /feedback | POST | Submit oracle or display feedback |
| /predictions | GET | Read prediction history with feedback and audit trail |
| /lots/summary | GET | Read lot level quality summary |
| /incidents | GET | Read structured AI security incidents |
| /metrics | GET | Export Prometheus compatible metrics |
| /admin/reload-model | POST | Request model reload from MLflow Registry |

## Prediction contract

The /predict route requires:

| Field | Rule |
| --- | --- |
| piece_event_id | Required |
| scenario_id | Required |
| image_uri | Required |

Optional traceability fields are:

| Field | Purpose |
| --- | --- |
| sha256 | Image integrity and audit linkage |
| lot_id | Production or replay lot grouping |
| source_class | Source class, for example Casting_class1 |
| dataset_version | Dataset or replay version |

The prediction response includes:

| Field | Purpose |
| --- | --- |
| prediction_id | Generated prediction identifier |
| piece_event_id | Piece event identifier |
| scenario_id | Scenario identifier |
| image_uri | Input image URI |
| sha256 | Optional image hash |
| lot_id | Optional lot identifier |
| source_class | Optional source class |
| dataset_version | Optional dataset version |
| decision | Vert, Orange or Rouge |
| model_version | Feature AE version |
| roi_model_version | ROI model version |
| created_at | API timestamp |

## Piece event prediction contract

The /piece-events/{event_id}/predict route uses the path event id as piece_event_id.

It accepts the same traceability fields as /predict, except that piece_event_id comes from the URL path.

## Model version contract

/model/version requires scenario_id.

The response exposes:

| Field | Purpose |
| --- | --- |
| scenario_id | Requested scenario |
| registered_model_name | MLflow registered model name derived from the scenario |
| source_of_truth | Must be mlflow_registry |
| roi_segmenter | ROI model manifest |
| feature_ae | Feature AE model manifest |

Registered model naming rule:

    feature_ae__{scenario_id}

## Feedback contract

/feedback requires:

| Field | Rule |
| --- | --- |
| prediction_id | Existing prediction id |
| piece_event_id | Must match the prediction |
| scenario_id | Must match the prediction |

Allowed feedback sources:

| Source | Purpose |
| --- | --- |
| oracle_gt | Sovereign source for training eligibility |
| human_sophie | Display only review feedback |

Optional feedback fields:

| Field | Purpose |
| --- | --- |
| feedback_status | Business and safety feedback status |
| human_override | Reserved for future workflow |
| gt_mask_uri | Oracle GT mask URI |
| gt_mask_has_defect | Oracle GT defect flag |
| comment | Review comment |

## Prediction history contract

/predictions returns rows containing prediction state, feedback state and audit trail.

Important fields are:

| Field | Purpose |
| --- | --- |
| prediction_id | Prediction identifier |
| piece_event_id | Piece event identifier |
| scenario_id | Scenario identifier |
| lot_id | Lot identifier |
| source_class | Source class |
| sha256 | Image hash |
| dataset_version | Dataset version |
| decision | Model decision |
| model_version | Feature AE version |
| roi_model_version | ROI model version |
| feedback_closed | Oracle feedback closure flag |
| display_decision_source | Display decision source |
| train_eligibility_source | Training eligibility source |
| eligible_for_train | Training eligibility flag |
| train_block_reason | Reason when training is blocked |
| conflict_logged | Feedback conflict flag |
| oracle_verdict | Oracle verdict |
| divergence | Model versus oracle divergence |
| audit_trail | Nested prediction and feedback audit context |

## Lot summary contract

/lots/summary returns lot level indicators.

Rows include:

| Field | Purpose |
| --- | --- |
| lot_id | Lot grouping key |
| scenario_id | Scenario identifier |
| total | Number of predictions |
| vert | Vert decisions |
| orange | Orange decisions |
| rouge | Rouge decisions |
| feedback_closed | Closed oracle feedback count |
| divergences | Divergence count |
| taux_orange | Orange rate |
| taux_rouge | Rouge rate |

When lot_id is missing, the route falls back to scenario_id.

## Incident contract

/incidents exposes structured API incidents.

Optional filters:

| Parameter | Purpose |
| --- | --- |
| incident_type | Filter by incident type |
| scenario_id | Filter by scenario |

Supported incident types:

| Incident type | Meaning |
| --- | --- |
| false_negative | Model said Vert while oracle GT found a defect |
| roi_warning | ROI warning context |
| roi_fail | ROI failure blocks training |
| feedback_conflict | Feedback identifier mismatch |
| reload_refused | Admin reload refused |
| invalid_prediction_request | Invalid prediction or feedback request |
| unsafe_train_candidate_blocked | Unsafe candidate dataset sample blocked |

## Admin reload contract

/admin/reload-model is protected by X-IQA-Admin-Token.

The expected secret is read from IQA_ADMIN_TOKEN.

Refusal cases:

| Case | HTTP status | Error code |
| --- | --- | --- |
| Token not configured | 503 | admin_token_not_configured |
| Invalid token | 401 | invalid_admin_token |

Both refusal cases are audited and create a reload_refused incident.

## Standard error contract

API errors follow ApiErrorResponse.

Fields:

| Field | Purpose |
| --- | --- |
| error_code | Stable machine readable code |
| message | Human readable message |
| status_code | HTTP status |
| reason | Optional detailed reason |
| incident_type | Optional incident category |
| audit_logged | Whether the refusal was audited |
| reload_event_id | Reload audit identifier when relevant |
| details | Extra structured details |

Covered statuses:

| Status | Meaning |
| --- | --- |
| 400 | Bad business input |
| 401 | Invalid token |
| 403 | Reserved forbidden contract |
| 404 | Unknown prediction id |
| 409 | Conflict, replay or mismatch |
| 422 | Pydantic validation error |
| 500 | Internal server error contract |
| 503 | Admin token not configured |
