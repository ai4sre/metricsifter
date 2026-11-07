# AGENTS.md

## Rules
- `Sifter`'s defaults reproduce the algorithm of the paper: new knobs are opt-in, and a change to the default selection needs evidence beyond the unit tests.
- The library never logs and does no I/O outside `metricsifter/cli.py`.
- `experiments/` reproduces the paper in its own Python 3.10 environment: never install its extra next to `dev`; change it only when a task asks.

## Compatibility contracts
- Public API: `__all__` in `metricsifter/__init__.py`, the `Sifter`/`SifterTransformer` parameters and defaults, the `py.typed` type hints.
- Return types of `run()`, `run_with_selected_segment()`, `run_upto_cpd()`, `sift()`; `SifterTransformer`'s `get_params`/`set_params`/`clone` convention.
- The `SiftResult.to_dict()`/`to_json()` schema and its `from_dict` round trip (`metricsifter/types.py`), consumed by LLM agents and `--report`.
- The four-way partition of the input columns and its independence from their order (`tests/test_properties.py`).
- Prometheus adapter column names and `df.attrs["metric_labels"]` (`metricsifter/adapters/prometheus.py`); CLI subcommand, flags, exit codes (`cli.py`).

## Before opening a pull request
- `make check`; also `make test-fuzz` for change-point detection, segmentation, the Prometheus adapter or CLI input handling.
- Dependencies: `uv lock`, then `make vuln`.

## Git conventions
- Branches: `<type>/<issue>-<summary>` (`<type>/<summary>` without an issue); tools' `agent/*`, `claude/*`, `copilot/*`, `dependabot/*` keep their names.
- Commits and PR titles: Conventional Commits 1.0.0, types `feat` `fix` `perf` `refactor` `test` `docs` `build` `ci` `chore` `revert`; scopes: `sifter`, `detection`, `segmentation`, `report`, `cli`, `prometheus`, `transformer`, `experiments`.
