# pr-council-mcp developer/CI tasks use the project's locked environment. The
# config generator needs only the standard library. `make` with no target prints help.

.DEFAULT_GOAL := help
.PHONY: help install format format-check lint lint-fix typecheck test coverage build clean ci \
	eval eval-pr eval-full eval-review eval-judge eval-report

# ReviewBench offline evaluation (evals/reviewbench/README.md). These targets call the configured models and the judge,
# so they cost money and are never part of `ci`. `eval` reviews the EVAL_PRS indices of EVAL_SET, so repeated runs
# compare the same pull requests; override EVAL_PRS to probe others, or run `eval-full` for the whole set. `eval-pr`
# runs only EVAL_PR, the pull request the system currently handles worst, for quick iteration; change it when another
# becomes the weakest. Name a run with EVAL_RUN to judge or report it later, and compare against earlier runs with
# EVAL_BASELINE.
REVIEWBENCH ?= ../ReviewBench
EVAL_SET ?= test
EVAL_PRS ?= 4 9 11 18 19 20 21 22 23 24
EVAL_PR ?= 12
EVAL_RUN ?= $(shell date -u +%Y%m%d-%H%M%S)
EVAL_BASELINE ?=
EVAL_STAGES ?= findings
EVAL_JOBS ?= 1
EVAL_JUDGE_CONCURRENCY ?= 2
EVAL_ARGS ?=
EVAL_RUNS := evals/reviewbench/runs
# Resolve a default run name once, and pass it to sub-makes so every step uses the same run.
override EVAL_RUN := $(EVAL_RUN)
EVAL_DIR = $(EVAL_RUNS)/$(EVAL_RUN)

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

eval: ## Review, judge, and report the EVAL_PRS subset of ReviewBench (costs model calls)
	@test ! -e $(EVAL_DIR) || { echo "$(EVAL_DIR) already exists; choose another EVAL_RUN" >&2; exit 1; }
	@status=0; $(MAKE) --no-print-directory EVAL_RUN=$(EVAL_RUN) eval-review || status=$$?; \
	test -f $(EVAL_DIR)/run.json || exit $$status; \
	$(MAKE) --no-print-directory EVAL_RUN=$(EVAL_RUN) eval-judge eval-report && exit $$status

eval-pr: ## Review, judge, and report only EVAL_PR (costs model calls)
	@$(MAKE) --no-print-directory EVAL_RUN=$(EVAL_RUN) eval EVAL_PRS=$(EVAL_PR)

eval-full: ## Review, judge, and report the whole EVAL_SET (costs model calls)
	@$(MAKE) --no-print-directory EVAL_RUN=$(EVAL_RUN) eval EVAL_PRS=

eval-review: ## Review the EVAL_PRS subset into EVAL_RUN without judging
	uv run python evals/reviewbench/run.py --reviewbench $(REVIEWBENCH) --set $(EVAL_SET) $(addprefix --pr ,$(EVAL_PRS)) \
		--jobs $(EVAL_JOBS) --run-dir $(EVAL_DIR) $(EVAL_ARGS)

eval-judge: ## Judge EVAL_STAGES of an existing EVAL_RUN; rerun to resume
	@test -f $(EVAL_DIR)/run.json || { echo "$(EVAL_DIR) is not a run; set EVAL_RUN" >&2; exit 1; }
	uv run python evals/reviewbench/judge.py --reviewbench $(REVIEWBENCH) $(addprefix --stage ,$(EVAL_STAGES)) \
		--concurrency $(EVAL_JUDGE_CONCURRENCY) $(EVAL_DIR)

eval-report: ## Report a judged EVAL_RUN against any EVAL_BASELINE runs
	@test -f $(EVAL_DIR)/run.json || { echo "$(EVAL_DIR) is not a run; set EVAL_RUN" >&2; exit 1; }
	uv run python evals/reviewbench/report.py $(addprefix --stage ,$(EVAL_STAGES)) $(EVAL_DIR) \
		$(addprefix $(EVAL_RUNS)/,$(EVAL_BASELINE))
