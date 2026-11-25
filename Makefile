.PHONY: install test lint typecheck golden schema smoke
install:
	uv sync --extra dev
test:
	uv run pytest
lint:
	uv run ruff check strata tests
typecheck:
	uv run mypy strata
golden:
	uv run strata golden
schema:
	uv run strata schema --out schema
smoke:  # the documented no-key run: generate, evaluate the reference system, verify, report
	uv run strata generate && uv run strata evaluate --adapter reference --block perturbation --quiet && uv run strata verify && uv run strata report
