# IQA MVP Threat Model

## Scope

This threat model covers the Phase 1 MVP API and model-governance loop:
prediction, feedback, admin model reload, metrics, replay scenarios and
traceability identifiers.

The detailed signed-off reports are stored as PDF evidence in
`reports/phase1/`. This Markdown file is the lightweight Git-readable index used
by the governance matrix.

## Key Threats

| ID | Threat | MVP control |
| --- | --- | --- |
| T1 | Feedback replay after a prediction is closed | `/feedback` requires `prediction_id` and rejects closed predictions |
| T2 | Feedback attached to the wrong `piece_event` or scenario | API checks `prediction_id`, `piece_event_id` and `scenario_id` consistency |
| T3 | Human feedback poisoning the training set | `human_sophie` is display-only; `oracle_gt` remains sovereign for train eligibility |
| T4 | Unsafe defect samples entering good-only training | defective GT feedback sets `eligible_for_train=false` |
| T5 | Admin reload without valid authorization | `/admin/reload-model` requires `IQA_ADMIN_TOKEN` |
| T6 | Silent reload refusal or reload abuse | accepted and refused reload attempts are written to `ADMIN_RELOAD_LOG` |
| T7 | Loss of prediction traceability | `/predict` returns `prediction_id`, model versions and audit fields |
| T8 | Invalid or unexpected API payloads | Pydantic contracts reject extra or invalid fields |
| T9 | Missing security observability | `/metrics` exposes AI security counters |
| T10 | Heavy model artifacts committed to Git | Git stores manifests and checksums; MinIO stores `.pt` artifacts |

## Evidence

- API contracts: `src/iqa/api/schemas.py`
- API implementation: `src/iqa/api/main.py`
- Security tests: `tests/security/test_api_security_contracts.py`
- Governance matrix: `docs/governance/iqa_phase1_governance_matrix.md`
- Reports: `reports/phase1/`
