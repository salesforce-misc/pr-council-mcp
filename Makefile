# pr-council-mcp developer/CI tasks use the project's locked environment. The
# config generator needs only the standard library. `make` with no target prints help.

.DEFAULT_GOAL := help
.PHONY: help install format format-check lint lint-fix typecheck test coverage build clean ci

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Sync the venv from the lockfile (incl. dev deps)
	uv sync

format: ## Format code with ruff
	uv run ruff format .

format-check: ## Check formatting without modifying files
	uv run ruff format --check .

lint: ## Lint with ruff
	uv run ruff check .

lint-fix: ## Lint and apply safe autofixes
	uv run ruff check --fix .

typecheck: ## Type-check the package with mypy
	uv run mypy src

test: ## Run the test suite
	uv run pytest

coverage: ## Run tests with coverage and write coverage.xml
	uv run pytest --cov=pr_council --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=80

build: ## Build the sdist and wheel into dist/
	uv build

clean: ## Remove build artifacts and caches
	rm -rf dist build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml htmlcov

ci: format-check lint typecheck coverage ## Run all checks and emit coverage
