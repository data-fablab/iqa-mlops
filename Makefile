.PHONY: sync lint test contracts dags replay simulate validate api demo demo-scratch smoke

sync:
	uv sync --extra cpu

lint:
	uv run --extra cpu ruff check src scripts tests

test:
	uv run --extra cpu pytest -q

contracts:
	uv run --extra cpu pytest tests/api tests/contracts -q

dags:
	uv run --extra cpu --extra mlops pytest tests/airflow -q

demo:
	uv run --extra cpu iqa-demo-phase2

demo-scratch:
	bash deploy/demo-from-scratch.sh

smoke:
	bash deploy/smoke-test.sh

replay:
	uv run --extra cpu iqa-build-flux-plan

simulate:
	uv run --extra cpu iqa-simulate-lifecycle

validate:
	uv run --extra cpu iqa-validate-mvp

api:
	uv run --extra cpu uvicorn iqa.api.main:app --host 0.0.0.0 --port 8000
