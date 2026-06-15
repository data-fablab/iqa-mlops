# IQA Security Test Suite

This folder contains Phase 1 MVP security and governance contract tests.

Current coverage:

- prediction audit fields and `prediction_id` generation;
- feedback lifecycle checks against replay and mismatched identifiers;
- `human_sophie` display-only behavior;
- `oracle_gt` sovereignty for training eligibility;
- admin reload refusal/acceptance logging;
- AI security counters exposed by `/metrics`;
- Pydantic rejection of invalid or unexpected payload fields.

Run with:

```powershell
uv run --extra cpu pytest -q tests/security
```
