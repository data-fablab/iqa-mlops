# IQA Phase 1 Governance Matrix

## Scope

This document maps the IQA Phase 1 API, feedback, schemas, metrics and AI security controls to practical governance principles.

This matrix is not a legal compliance assessment. It does not claim full GDPR or EU AI Act compliance. It documents how the MVP controls support traceability, accountability, human oversight, risk control and cybersecurity by design.

## Governance position

IQA Phase 1 focuses on a computer vision quality assistant for industrial Casting parts.

No personal data processing has been identified in the Phase 1 technical scope. GDPR is therefore treated as a preventive governance reference, mainly for minimisation, purpose limitation, integrity, confidentiality and accountability.

The system is not legally classified as a high risk AI system in this phase. The EU AI Act is used as a governance reference for risk management, data governance, technical documentation, record keeping, human oversight, accuracy, robustness and cybersecurity.

## Phase 1 governance controls

| Governance area | Phase 1 control | Technical evidence | Status |
| --- | --- | --- | --- |
| Risk management | MVP AI security threat model defines key threats T1 to T10 | docs/security/threat_model_mvp.md and Word threat model report | Implemented |
| API contracts | Pydantic schemas reject invalid or unexpected fields | src/iqa/api/schemas.py and pytest contract tests | Implemented |
| Prediction traceability | Prediction returns prediction_id, piece_event_id, scenario_id, model versions and audit block | src/iqa/api/main.py and tests/security/test_api_security_contracts.py | Implemented |
| Feedback governance | Feedback requires prediction_id and cannot be replayed after closure | feedback lifecycle tests | Implemented |
| Human oversight | human_sophie can influence display decision only | human feedback display governance tests | Implemented |
| Train eligibility | oracle_gt remains sovereign for training eligibility | feedback governance tests | Implemented |
| Reload governance | admin reload requires IQA_ADMIN_TOKEN | admin reload tests | Implemented |
| Reload record keeping | accepted and refused reload attempts are logged | ADMIN_RELOAD_LOG and pytest tests | Implemented |
| Security metrics | AI security counters are exposed through /metrics | metrics endpoint and pytest tests | Implemented |
| Source of truth | reload target is resolved from MLflow registry reference | reload_model target response | Implemented for MVP |
| Secrets governance | real secrets must stay outside Git | .gitignore, .env policy and project operating rule | Documented |
| Reproducibility | tests are executed on CubeAI and must be replayed on Z420 | pytest outputs and future validation report | In progress |

## EU AI Act alignment

| AI Act governance theme | IQA Phase 1 alignment | Evidence |
| --- | --- | --- |
| Risk management | Threat model and security controls identify feedback poisoning, invalid feedback, unsafe train candidates, reload abuse and traceability loss | Threat model and NAT01 to NAT10 commits |
| Data and data governance | Prediction, feedback and scenario contracts reduce unsafe or untraceable data flow | schemas.py and feedback tests |
| Technical documentation | Threat model report, governance matrix, Phase 1 report and recipe report document the controls | Word reports and docs folder |
| Record keeping | Prediction audit, feedback lifecycle and admin reload log provide MVP level records | API response audit and ADMIN_RELOAD_LOG |
| Human oversight | Sophie is display priority only and cannot override GT for train eligibility | human feedback display tests |
| Accuracy, robustness and cybersecurity | Invalid feedback, reload abuse and unsafe train candidates are blocked or counted | security tests and /metrics |

## GDPR alignment

| GDPR principle | IQA Phase 1 interpretation | Evidence |
| --- | --- | --- |
| Purpose limitation | Technical identifiers are used for quality decision traceability and security only | prediction_id, piece_event_id, scenario_id |
| Data minimisation | No personal data is required for Phase 1 tests or API contracts | test payloads and schemas |
| Accuracy | GT remains sovereign for train eligibility | oracle_gt feedback tests |
| Integrity and confidentiality | Admin reload is protected by token and refused attempts are logged | reload tests and ADMIN_RELOAD_LOG |
| Accountability | Controls are documented, tested, committed and reproducible | Git commits, pytest outputs and reports |

## Current limits

The Phase 1 logs are in memory and are sufficient for MVP demonstration and unit tests. On the Z420 environment, persistent logging should be implemented in PostgreSQL, JSONL logs or another agreed logging sink.

The governance matrix does not replace a DPIA, a legal GDPR assessment, an EU AI Act classification analysis or a full production compliance package.

Candidate dataset governance, ROI warning handling, ROI fail handling, validation set exclusion and false negative promotion blocking will need additional tests when the corresponding components are finalized.

## Recipe report mapping

The Phase 1 recipe report will include the following evidence for each implemented control:

| Evidence type | Expected content |
| --- | --- |
| Test id | pytest function name |
| Threat covered | Link to T1 to T10 when applicable |
| Precondition | API object, token or metric state |
| Action | Function call or endpoint behavior |
| Expected result | ValidationError, HTTPException, accepted response or metric increment |
| Obtained result | pytest passed result |
| Proof | screenshot and commit hash |
