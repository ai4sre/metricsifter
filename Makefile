# Standard entry points. Each target wraps the uv commands documented in
# AGENTS.md; `make help` lists them. Python comes from uv: set UV_PYTHON (for
# example `UV_PYTHON=3.10 make test`) to pick the interpreter.

# Tool versions. CI pins the same versions (.github/workflows); bump together.
ACTIONLINT_VERSION  ?= v1.7.12
ZIZMOR_VERSION      ?= 1.32.0
OSV_SCANNER_VERSION ?= v2.7.0
# Examples per property test for test-fuzz (the deterministic default is in tests/conftest.py).
HYPOTHESIS_MAX_EXAMPLES ?= 2000
PYTEST_ARGS ?= -s -vv

UV          ?= uv
RUN         := $(UV) run --locked --extra dev
# actionlint and osv-scanner run through `go run`, which verifies the module checksums.
ACTIONLINT  ?= go run github.com/rhysd/actionlint/cmd/actionlint@$(ACTIONLINT_VERSION)
OSV_SCANNER ?= go run github.com/google/osv-scanner/v2/cmd/osv-scanner@$(OSV_SCANNER_VERSION)
# uvx or pipx (preinstalled on GitHub-hosted runners) runs the pinned zizmor.
ZIZMOR      ?= $(if $(shell command -v uvx 2>/dev/null),uvx zizmor@$(ZIZMOR_VERSION),pipx run zizmor==$(ZIZMOR_VERSION))

.DEFAULT_GOAL := help
.PHONY: help check lint fmt test test-fuzz build lint-actions vuln

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-13s %s\n", $$1, $$2}'

check: lint test build lint-actions ## Local pre-PR gate: lint, tests, build, workflow audit

lint: ## Lock file, ruff, black --check and mypy (the CI quality job)
	$(UV) lock --check
	$(RUN) ruff check .
	$(RUN) black --check .
	$(RUN) mypy metricsifter tests

fmt: ## Apply the formatter (black)
	$(RUN) black .

test: ## Full test suite (property tests run deterministically)
	$(RUN) pytest $(PYTEST_ARGS) tests

test-fuzz: ## Property tests with fresh random examples (HYPOTHESIS_MAX_EXAMPLES per test)
	HYPOTHESIS_PROFILE=fuzz HYPOTHESIS_MAX_EXAMPLES=$(HYPOTHESIS_MAX_EXAMPLES) $(RUN) pytest -q tests/test_properties.py

build: ## Build the sdist and wheel into dist/
	$(UV) build

# zizmor runs its online audits only with a GitHub token (CI passes one); `make
# lint-actions GH_TOKEN=` forces offline mode where GitHub's API is unreachable.
lint-actions: ## Lint (actionlint) and audit (zizmor) the GitHub Actions workflows
	$(ACTIONLINT)
	$(if $(GH_TOKEN),$(ZIZMOR),env -u GH_TOKEN $(ZIZMOR) --offline) .

vuln: ## Known vulnerabilities in uv.lock (osv-scanner; exceptions in osv-scanner.toml)
	$(OSV_SCANNER) scan source --config osv-scanner.toml --lockfile uv.lock
