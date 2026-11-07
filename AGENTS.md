# AGENTS.md

Guide for coding agents (and humans) working in this repository. The
user-facing documentation is [README.md](README.md); this file covers how the
code is built, tested and changed.

## 1. Overview and architecture

MetricSifter is a Python library (plus a small CLI) that removes the metrics of
a multivariate time series that are unrelated to a failure: it detects change
points per metric, finds the time frame with the highest change-point density,
and keeps only the metrics with a change point in that frame. It accompanies
the IEEE Access 2024 paper linked from the README.

`Sifter` (`metricsifter/sifter.py`) runs the pipeline; `sift()` returns a
`SiftResult`, and `run()`, `run_with_selected_segment()` and `run_upto_cpd()`
are thin backward-compatible wrappers.

1. **Simple filter** (`Sifter._filter_no_changes`): drops all-NaN and constant
   series, and constant-slope series without NaN. Skipped by
   `without_simple_filter=True`.
2. **Change-point detection** (`metricsifter/algo/detection.py`): one
   univariate detection per metric with ruptures (`pelt` = `KernelCPD` with a
   linear kernel, `binseg`, `bottomup`). The penalty (`aic` / `bic` from a
   noise-scale estimate, `sigma_estimator`: `std` / `mad` / `diff_std`, or a
   number) is multiplied by `penalty_adjust`; `penalty_adjust="auto"` picks the
   multiplier by a penalty-plateau search. Leading/trailing NaN are trimmed, interior NaN are
   linearly interpolated, and the positions where a metric goes missing are
   change points in their own right.
3. **KDE segmentation** (`metricsifter/algo/segmentation.py`): a statsmodels
   KDE over all change points, split at the local minima of the density;
   `bandwidth` is a float, `scott` / `silverman`, or `auto` (bootstrap
   stability selection, seeded by `random_state`).
4. **Segment selection** (`Sifter.select_largest_segment_with_label`): `max`,
   `weighted_max` (default) or a `Callable[[SegmentCandidate], float]`.

Around the core: `types.py` (`SiftResult`, `SegmentInfo`, tuning reports, JSON
serialization), `transformer.py` (`SifterTransformer`, a duck-typed
scikit-learn transformer with no scikit-learn dependency),
`adapters/prometheus.py` (Prometheus `query_range` matrix -> DataFrame, no
HTTP), `cli.py` (`metricsifter run`, stdlib argparse), `plot.py` (optional
matplotlib extra `viz`), `evaluation.py` (precision/recall against a ground
truth) and `utils.py` (joblib column-parallel apply).

## 2. Directory layout

| Path | Contents |
|---|---|
| `metricsifter/` | The published package (`py.typed`: its type hints are part of the API). |
| `tests/` | pytest suite. `conftest.py` builds deterministic synthetic data and registers the Hypothesis profiles; `test_properties.py` holds the property-based tests; `test_workflows.py` guards the shape of the CI and release workflows. |
| `experiments/` | Research code that reproduces the paper. **Not** part of the package, with its own Python 3.10 environment (see "Experiments" below). |
| `docs/` | README images, design notes (`docs/design/`) and earlier remediation plans. |
| `examples/` | A notebook. |
| `.github/` | CI (`ci.yaml`), security scans (`security.yml`), complexity tracking (`complexity.yml`), PyPI release (`publish.yaml`), Dependabot. |
| `.claude/` | Claude Code settings (see CLAUDE.md). |

## 3. Build and environment

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.10-3.14 (uv can
download interpreters). Go is needed only for `make lint-actions` and
`make vuln`, which run actionlint and osv-scanner through `go run`.

```bash
uv sync --extra dev          # .venv with the package (editable) and the dev tools, from uv.lock
UV_PYTHON=3.10 make test     # any target on another interpreter
uv build                     # sdist and wheel in dist/
```

After changing dependencies in `pyproject.toml`, run `uv lock` and commit
`uv.lock`; every CI job installs with `--locked`.

**Experiments.** `experiments/` needs `scikit-learn<1.2` and `numpy<2` for the
pinned PyRCA revision, so the `dev` and `experiments` extras are declared as
conflicting (`[tool.uv] conflicts`) and must never share an environment. Use
`uv venv --python 3.10 .venv-experiments` and install `-e ".[experiments]"`
plus the pinned `sfr-pyrca` Git revision (not on PyPI) into it
(`experiments/README.md`). The standard test suite does not need PyRCA.

## 4. Tests

| Command | What | Needs |
|---|---|---|
| `make test` | The whole suite: `uv run pytest -s -vv tests` | nothing (no network, root or Docker) |
| `uv run pytest -s -v tests/test_sifter.py::test_sifter_run` | A single test | |
| `make test-fuzz` | Only the property tests, with fresh random examples (`HYPOTHESIS_MAX_EXAMPLES`, default 2000 per test, a few minutes) | |

- All test data is synthetic and generated at runtime with fixed seeds
  (`tests/conftest.py`); there are no fixture files to manage.
- The property tests (`tests/test_properties.py`, Hypothesis) run in every
  `make test` with the `deterministic` profile: derandomized and 40 examples
  per test, so CI never flakes on a random draw. `HYPOTHESIS_PROFILE=fuzz`
  (what `make test-fuzz` and the weekly CI `fuzz` job use) explores new inputs
  and stores failures in `.hypothesis/`. A failure prints a
  `@reproduce_failure` decorator; turn a real bug into an `@example(...)` on
  the property (or an example-based test) next to the fix.
- CI runs the suite on Python 3.10, 3.11, 3.12, 3.13 and 3.14.
- There are no integration, e2e or benchmark suites.

## 5. Static analysis

| Command | What |
|---|---|
| `make lint` | `uv lock --check`, `ruff check .`, `black --check .`, `mypy metricsifter tests` (the CI quality job) |
| `make fmt` | `black .` |
| `make lint-actions` | actionlint and zizmor on `.github/` (zizmor runs offline without `GH_TOKEN`; `make lint-actions GH_TOKEN=` forces it) |
| `make vuln` | osv-scanner on `uv.lock` (needs network). Reasoned, expiring exceptions live in `osv-scanner.toml`. |
| `cccc --table --top-cognitive 20 .` | Complexity ranking (cccc; CI tracks the scores with octocov, it never fails on them; `cccc.toml` excludes tests and experiments) |

CodeQL (Python) runs in `security.yml`. Tool versions are pinned in the
`Makefile` and the workflows; bump them together.

## 6. Coding conventions

- black and ruff with line length 120; ruff keeps the pre-0.17 default rules
  (`E4`, `E7`, `E9`, `F`).
- Target Python 3.10 syntax and semantics (`X | Y` unions, `match`), also for
  mypy (`mypy.ini`). Every function in `metricsifter/` is annotated
  (`disallow_untyped_defs`).
- Invalid input or parameters raise `ValueError` with a message naming the
  parameter and the accepted values; the library does not log. The CLI turns
  input errors into exit code 2 instead of a traceback.
- The defaults of `Sifter` reproduce the original algorithm: new knobs are
  opt-in.
- Docstrings in Google style; comments and docs in English; comments say why.
- Commits: short prefixed subjects (`fix:`, `feat:`, `docs:`, `ci:`,
  `chore:`, `build:`, `release:`).

## 7. Compatibility contracts

Changing any of these is a breaking change that must be called out:

- The public API in `metricsifter/__init__.py` (`__all__`), the `Sifter` and
  `SifterTransformer` constructor parameters and defaults, and the return
  types of `run()`, `run_with_selected_segment()`, `run_upto_cpd()` and
  `sift()`. `SifterTransformer` must keep the scikit-learn
  `get_params`/`set_params`/`clone` convention (arguments stored verbatim).
- The `SiftResult.to_dict()` / `to_json()` schema (keys and meaning), which
  LLM agents and the CLI `--report` consume, and its `from_dict` round trip.
- The four-way partition of the input columns into `selected_metrics` and the
  three exclusion reasons, and independence from the column order of the input
  (both asserted by `tests/test_properties.py`; the KDE is fitted on sorted
  change points for the latter).
- Prometheus adapter column names (`name{k="v",...}`, labels sorted, `#<n>`
  suffix on collisions) and `df.attrs["metric_labels"]`.
- CLI subcommand, flags and exit codes (0 success, 2 input error).
- Type hints (the package ships `py.typed`).
- Release: the tag without its `v` must equal `project.version`;
  `tests/test_workflows.py` checks the CI and release workflow structure, and
  `tests/test_experiment_sources.py` the experiments extra and its pins.

## 8. Concurrency and resources

- Parallelism is joblib (`n_jobs`, joblib convention). `Sifter` defaults to
  `n_jobs=1`; the lower-level helpers (`detect_multi_changepoints`,
  `_filter_no_changes`, `parallel_apply`) default to `-1` (all cores), so pass
  `n_jobs` explicitly. Tests use `n_jobs=1` except where they test parallelism.
- No global mutable state. The only randomness is the `bandwidth="auto"`
  bootstrap, seeded by `random_state`.
- The library does no I/O except the CLI (reads the input CSV, writes the
  output CSV and report) and `plot.py` (returns figures; saving is the
  caller's choice).

## 9. Performance

- Change-point detection dominates: one ruptures fit per metric (PELT on
  `KernelCPD` is implemented in C). `penalty_adjust="auto"` adds 13 cheap
  `predict` calls per metric; `bandwidth="auto"` reruns the KDE segmentation
  20 times for each of 12 candidate bandwidths.
- Keep per-metric work vectorized (NumPy) and avoid per-sample Python loops.

## 10. Before opening a PR

1. `make check` (`lint`, `test`, `build`, `lint-actions`) passes. Where the GitHub
   API is unreachable but `GH_TOKEN` is set, run `env -u GH_TOKEN make check`.
2. Touched detection, segmentation, the Prometheus adapter or the CLI input
   handling: also `make test-fuzz`.
3. Changed dependencies: `uv lock`, then `make vuln`.
4. Changed the public API, CLI or report schema: update README.md.
5. Changed a workflow: keep `tests/test_workflows.py` passing (it is part of
   `make test`) and `make lint-actions` clean.

## 11. Rules for AI agents

Never:

- disable, skip, delete or weaken tests, assertions, lint or type-check rules
  to get green (no blanket ignores, `|| true`, `continue-on-error`, or lowered
  Hypothesis budgets in the default profile);
- edit generated files by hand (`uv.lock` comes from `uv lock`);
- install the `dev` and `experiments` extras into one environment, or modify
  `experiments/` beyond what a task explicitly asks;
- add dependencies (runtime or dev) without a justification in the PR;
- change a contract from section 7 without calling it out;
- loosen workflow permissions, unpin an action from its commit SHA, or add
  caches to the release workflow;
- commit secrets, rewrite history or force-push, tag, or publish to PyPI.

## 12. Changes that need human review

Even with green CI, ask for a maintainer's review of:

- workflows, `publish.yaml`, Dependabot, permissions and secrets;
- the PyPI release process (Trusted Publishing environment `metricsifter_pypi`);
- `osv-scanner.toml` exceptions;
- parsers of external input (`adapters/prometheus.py`, CLI CSV handling) and
  changes to the change-point or segmentation numerics, whose effect on
  selection quality tests cannot fully capture;
- the contracts in section 7, dependency additions, and large refactors.
