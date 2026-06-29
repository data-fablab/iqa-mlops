.PHONY: sync lint test contracts dags replay simulate validate api demo demo-scratch demo-reset demo-runsheet smoke

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

# Idempotent reset to the clean class1-only start (runsheet Phase 0). Restores the
# active artifacts from the immutable baselines and restarts iqa-api/iqa-inference.
demo-reset:
	uv run --extra cpu python -m scripts.demo_reset

# One command for the whole runsheet (docs/runsheet_demo_20min.md): reset to the
# class1-only baseline, pre-warm, then drive class2 then class3 end-to-end —
# drift -> autonomous retrain (manual fallback if the sensor is silent) ->
# promotion/refresh/restart -> recovery, streaming live traffic throughout.
# Requires the stack up with the demo overrides (see the runsheet pre-reqs).
# Pass extra flags via ARGS, e.g.: make demo-runsheet ARGS="--classes Casting_class2"
demo-runsheet:
	uv run --extra cpu python -m scripts.run_demo_runsheet $(ARGS)

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
