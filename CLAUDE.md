# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
uv sync --extra dev

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

`experiments/` is a **separate** research evaluation framework for the paper — not part of the published package. It uses a separate Python 3.10 environment because its pinned PyRCA dependency requires scikit-learn below 1.2.

## Gotchas

- CI runs the full test suite on Python 3.10-3.14.
- `experiments/` requires the pinned `sfr-pyrca` revision, which is not on PyPI. The `dev` and `experiments` extras conflict by design and must use separate environments.
- Tests generate deterministic synthetic data at runtime via fixtures in `tests/conftest.py` — no fixture files to manage.
