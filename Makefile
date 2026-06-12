.PHONY: sync lint test replay simulate validate api

sync:
	uv sync --extra cpu

lint:
	uv run --extra cpu ruff check src scripts tests

test:
	uv run --extra cpu pytest -q

replay:
	uv run --extra cpu iqa-build-flux-plan

simulate:
	uv run --extra cpu iqa-simulate-lifecycle

validate:
	uv run --extra cpu iqa-validate-mvp

api:
	uv run --extra cpu uvicorn iqa.api.main:app --host 0.0.0.0 --port 8000
