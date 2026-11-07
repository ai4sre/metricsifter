# AGENTS.md

Rules for coding agents (and humans) that the code does not show: what the
code, `make help` and README.md already say is not repeated here.

## Rules the code does not show

- English only.
- `Sifter`'s defaults reproduce the algorithm of the paper: new knobs are
  opt-in, and a change to the default selection needs evidence beyond the
  unit tests.
- The library does not log and does no I/O outside the CLI; invalid input
  raises `ValueError` naming the parameter and the accepted values.
- No global mutable state; the only randomness goes through `random_state`.
- Pass `n_jobs` explicitly to the lower-level helpers (they default to all
  cores); tests use `n_jobs=1` unless they test parallelism.
- Keep per-metric work vectorized with NumPy, without per-sample Python loops.
- `make test` runs the property tests under the derandomized `deterministic`
  Hypothesis profile so that CI never flakes on a random draw: never lower
  its budget. Turn a real failure found by `make test-fuzz` into an
  `@example(...)` on the property (or an example-based test) with the fix.
- Never install the `dev` and `experiments` extras into one environment: the
  experiments need `scikit-learn<1.2` and `numpy<2` for the pinned PyRCA and
  run in their own Python 3.10 environment (`experiments/README.md`).

## Compatibility contracts

- The public API: `__all__` in `metricsifter/__init__.py`, the `Sifter` and
  `SifterTransformer` parameters and defaults, and the return types of
  `run()`, `run_with_selected_segment()`, `run_upto_cpd()` and `sift()`.
- `SifterTransformer` follows the scikit-learn
  `get_params`/`set_params`/`clone` convention (arguments stored verbatim).
- The `SiftResult.to_dict()`/`to_json()` schema and its `from_dict` round
  trip (`metricsifter/types.py`), consumed by LLM agents and `--report`.
- The four-way partition of the input columns (`selected_metrics` and three
  exclusion reasons) and independence from the input column order, asserted
  by `tests/test_properties.py`.
- Prometheus adapter column names and `df.attrs["metric_labels"]`
  (`metricsifter/adapters/prometheus.py`).
- CLI subcommand, flags and exit codes (`metricsifter/cli.py`).
- Type hints: the package ships `py.typed`.

## Before opening a pull request

`make check`. Also, when you touch:

- change-point detection, segmentation, the Prometheus adapter or CLI input
  handling: `make test-fuzz`;
- dependencies: `uv lock`, then `make vuln`;
- the public API, the CLI or the report schema: README.md.

## Git conventions

Branches: `<type>/<issue>-<summary>`, or `<type>/<summary>` without an issue
(`fix/166-connect-errors`, `feat/qp-counters`). `<type>` is a commit type
below; `<summary>` is 2–5 lowercase ASCII words joined by `-`. Branches named
by tools keep their names (`agent/*`, `claude/*`, `copilot/*`,
`dependabot/*`). Never push to the default branch.

Commits follow [Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

    <type>(<scope>)!: <subject>

    <body>

    <footer>

- `type`: `feat` (new user-visible behaviour), `fix` (bug fix), `perf`,
  `refactor` (no behaviour change), `test`, `docs`, `build` (build system,
  dependencies, packaging), `ci` (workflows), `chore` (anything else, e.g.
  `chore(release): prepare v1.2.3`), `revert`.
- `scope`: optional; the package or area changed: `sifter`, `detection`,
  `segmentation`, `report`, `cli`, `prometheus`, `transformer`,
  `experiments`.
- `!` and a `BREAKING CHANGE: <what changed and how to migrate>` footer for
  any change to a compatibility contract above.
- `subject`: imperative, lower case (except identifiers), no final period,
  at most 72 characters including the prefix.
- `body`: why the change is needed and what it affects, wrapped at 72
  columns; required unless the subject says everything.
- `footer`: `Fixes #N` / `Refs #N`, `BREAKING CHANGE:`, trailers such as
  `Co-Authored-By:`.
- One logical change per commit; regenerated files go in the commit that
  changes their source. Pull request titles use the same format (they become
  the squash commit).

## Forbidden

- Disabling, skipping, deleting or weakening tests, assertions, lint or
  type-check rules to get green (blanket ignores, `|| true`,
  `continue-on-error`, a lower Hypothesis budget in the default profile).
- Editing `uv.lock` by hand.
- Modifying `experiments/` beyond what a task explicitly asks.
- Adding runtime or dev dependencies without a justification in the PR.
- Changing a compatibility contract without calling it out.
- Loosening workflow permissions, unpinning an action from its commit SHA,
  adding caches to the release workflow.
- Committing secrets, rewriting history, force-pushing, tagging or
  publishing to PyPI.

## Needs human review

Even with green CI: workflows, `publish.yaml` and its Trusted Publishing
environment `metricsifter_pypi`, Dependabot, permissions and secrets;
exceptions in `osv-scanner.toml`; parsers of external input
(`metricsifter/adapters/prometheus.py`, CLI CSV handling) and the
change-point or segmentation numerics, whose effect on selection quality the
tests cannot fully capture; the compatibility contracts, dependency
additions and large refactors.
