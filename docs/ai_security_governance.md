# IQA Phase 2 AI Security Governance

## Purpose

This document describes the AI security governance controls implemented and validated during IQA Phase 2.

The controls protect prediction, feedback, reload, incident and candidate dataset lifecycle.

## Security objectives

| Objective | Control |
| --- | --- |
| Prevent feedback poisoning | Strict feedback identity checks |
| Preserve training data integrity | oracle_gt sovereignty |
| Prevent unsafe training samples | Good only and ROI safe candidate filtering |
| Protect model reload | Admin token and audit log |
| Make security events observable | Metrics and structured incidents |
| Preserve auditability | Prediction and feedback traceability |
| Standardize API errors | Structured error responses |

## Trust boundaries

| Boundary | Role |
| --- | --- |
| Streamlit | Review and display interface |
| FastAPI iqa-api | API contract and governance boundary |
| iqa-inference | Prediction boundary |
| MLflow Registry | Model source of truth |
| MinIO | Heavy artifact storage |
| Metadata repository target | Runtime facts, status and URI storage |
| Prometheus | Metrics collection |
| Airflow | Lifecycle orchestration |

## Scenario isolation

scenario_id is mandatory on Phase 2 critical routes.

It is required for prediction, model version and admin reload operations.

Scenario isolation prevents model, feedback and replay collisions between natural replay, drift replay and future production ingest scenarios.

## Feedback poisoning controls

The API blocks:

| Attack | Result |
| --- | --- |
| Unknown prediction id | HTTP 404 |
| Mismatched piece event id | HTTP 409 and feedback conflict incident |
| Mismatched scenario id | HTTP 409 and feedback conflict incident |
| Replay after closed feedback | HTTP 409 |
| Unknown feedback source | HTTP 400 |
| Human display feedback used for training | Blocked by design |

## Oracle GT sovereignty

oracle_gt is the only source that can close feedback and make a sample train eligible.

human_sophie remains display only.

This protects the training set from accidental or malicious display actions.

## Candidate dataset security

Candidate training datasets must remain safe for Feature AE training.

Required filtering rules:

| Rule |
| --- |
| Good label only |
| ROI status ok |
| No defective sample |
| Not in validation set |

The good only training validation rejects poisoned samples.

Rejected cases include:

| Case |
| --- |
| Defective sample |
| Non normal label |
| Validation set sample |

## ROI safety

ROI safety is part of AI security because the Feature AE model must not learn from images where the functional surface segmentation is unreliable.

roi_warning blocks training eligibility.

roi_fail blocks training eligibility and creates a high severity incident.

## False negative governance

A false negative is critical.

Definition:

| Model decision | Oracle verdict |
| --- | --- |
| Vert | defective |

Governance outcome:

| Field | Value |
| --- | --- |
| divergence | faux_negatif |
| eligible_for_train | false |
| incident_type | false_negative |
| severity | high |

## Admin reload governance

Model reload is protected by IQA_ADMIN_TOKEN.

Reload refusal cases:

| Case | Outcome |
| --- | --- |
| Token not configured | HTTP 503, audit log, reload refused incident |
| Invalid token | HTTP 401, audit log, reload refused incident |

Accepted reloads are audited but do not create reload refused incidents.

## API incident governance

The API stores structured incidents.

Current incident sources:

| Incident type | Source |
| --- | --- |
| feedback_conflict | Piece event or scenario mismatch |
| false_negative | Oracle GT detects defective while model said Vert |
| roi_fail | ROI fail blocks training |
| reload_refused | Admin reload refused |
| unsafe_train_candidate_blocked | Candidate dataset blocked |

The /incidents route supports filtering by incident_type and scenario_id.

## Metrics governance

Security and safety metrics include:

| Metric |
| --- |
| iqa_feedback_conflict_total |
| iqa_ai_security_incident_total |
| iqa_unsafe_train_blocked_total |
| iqa_invalid_feedback_total |
| iqa_reload_refused_total |
| iqa_roi_fail_total |
| iqa_divergence_filtered_total |

Filtered metric labels include scenario_id, lot_id, source_class, model_version, dataset_version, decision and divergence.

## Error governance

Errors are standardized through ApiErrorResponse.

Covered categories:

| Status | Category |
| --- | --- |
| 400 | Unknown feedback source |
| 401 | Invalid service or admin token |
| 403 | Reserved forbidden contract |
| 404 | Unknown prediction id |
| 409 | Conflict or replay |
| 422 | Pydantic validation |
| 500 | Internal server error contract |
| 503 | Admin token not configured |

## Validation evidence

Key suites:

| Test suite |
| --- |
| tests/api/test_feedback_poisoning_controls_contract.py |
| tests/api/test_train_eligibility_blocks_contract.py |
| tests/api/test_api_error_standardization_contract.py |
| tests/api/test_incidents_contract.py |
| tests/api/test_api_contract_hardening.py |
| tests/security/test_ai_security_governance_contracts.py |

Server validation after NAT16:

    95 passed
    SERVER_NAT16_AI_SECURITY_GOVERNANCE_OK

## Known limits

The current incident store is in memory.

PostgreSQL remains the target metadata persistence layer for production grade runtime storage.

The current API documents the stable contract before final database write through is enabled.
