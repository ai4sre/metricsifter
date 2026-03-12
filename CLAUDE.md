# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
uv sync --all-extras

# Test (all)
uv run pytest -s -vv tests

# Test (single)
uv run pytest -s -v tests/test_sifter.py::test_sifter_run

# Format
black .

# Lint
ruff check .
```

## Code Style

- line-length=120 (black and ruff)
- Target Python 3.10 — use `str | float` union syntax, `match/case`, etc.

## Architecture

Core package (`metricsifter/`) implements a 3-phase pipeline in `Sifter`:

1. **Simple Filtering** (`sifter.py:_filter_no_changes`) — removes constant/no-variation metrics
2. **Change Point Detection** (`algo/detection.py`) — per-metric CPD using ruptures library
3. **KDE Segmentation** (`algo/segmentation.py`) — clusters change points via KDE, selects densest segment

`experiments/` is a **separate** research evaluation framework for the paper — not part of the published package. It has its own dependencies and only works on Python 3.10-3.11.

## Gotchas

- CI runs full test suite on Python 3.10 and 3.11 only. Python 3.12-3.14 only verify that `import metricsifter` succeeds.
- `experiments/` requires `sfr-pyrca`, which is not on PyPI: `pip install git+https://github.com/salesforce/PyRCA@d85512b`
- Tests generate synthetic data at runtime via `tests/sample_gen/generator.py` — no fixture files to manage.
