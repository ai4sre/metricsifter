# Codebase Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 調査済みのコア、実験、依存管理、CI、リリース、文書の問題をすべて修正し、回帰テストと独立レビューで保証する。

**Architecture:** 既存の3段パイプラインを維持し、KDEの排他的区間、欠損を考慮したsimple filter、CPD入力の短絡と公開入口の検証を追加する。実験依存をコアテストから分離し、lock・静的検査・全対応Python・build済みwheelをCIと公開workflowのゲートにする。

**Tech Stack:** Python 3.10–3.14、pandas、NumPy、ruptures、statsmodels、pytest、Ruff、Black、uv、GitHub Actions

---

## File map

- `metricsifter/algo/segmentation.py`: KDE分割の排他性。
- `metricsifter/algo/detection.py`: 空・定数系列の短絡。
- `metricsifter/sifter.py`: DataFrame契約、simple filter、列順、パラメータ検証。
- `tests/test_segmentation.py`, `tests/test_detection.py`, `tests/test_sifter_extended.py`: コア回帰テスト。
- `experiments/localization/rcd.py`, `experiments/localization/pyrca.py`: 実験RCAの実行時エラー修正。
- `tests/test_experiment_sources.py`: 外部PyRCAなしで実験ソースの静的健全性を検査。
- `tests/test_sifter.py`: PyRCA依存のテストデータ生成を共通fixtureへ置換。
- `tests/test_transformer.py`: sklearn import failureの正しい扱い。
- `pyproject.toml`, `uv.lock`, `requirements-dev.txt`, `experiments/requirements.txt`: 依存の正規化。
- `.github/workflows/ci.yaml`, `.github/workflows/publish.yaml`: 品質・成果物・公開ゲート。
- `README.md`, `experiments/README.md`: 実装と一致する手順。
- `experiments/**/*.py`, `tests/**/*.py`: Black/Ruffで機械整形。

### Task 1: KDE segmentation partition invariant

**Files:**
- Modify: `metricsifter/algo/segmentation.py:74-107`
- Test: `tests/test_segmentation.py`

- [ ] **Step 1: Write the failing partition regression test**

```python
def test_boundary_change_point_belongs_to_exactly_one_cluster():
    change_points = [1, 4, 4, 6, 8, 8]
    labels, label_to_cps = segment_changepoints_with_kde(change_points, 21, 1.0)

    flattened = [int(cp) for cps in label_to_cps.values() for cp in cps]
    assert flattened.count(6) == 1
    for position, cp in enumerate(change_points):
        assert cp in label_to_cps[int(labels[position])]
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_segmentation.py::TestSegmentChangepointsWithKDE::test_boundary_change_point_belongs_to_exactly_one_cluster`

Expected: `flattened.count(6)` is `2` on the current implementation.

- [ ] **Step 3: Implement half-open cluster boundaries**

Use `<` for every upper bound and reserve `>= last_boundary` for the last cluster:

```python
clusters.append(np.where(x < s[mi][0])[0])
for i_cluster in range(len(mi) - 1):
    clusters.append(np.where((x >= s[mi][i_cluster]) & (x < s[mi][i_cluster + 1]))[0])
clusters.append(np.where(x >= s[mi][-1])[0])
```

- [ ] **Step 4: Verify GREEN and related tests**

Run: `.venv/bin/pytest -q tests/test_segmentation.py tests/test_autotune.py tests/test_regression.py`

Expected: all tests pass and each input position has one label.

- [ ] **Step 5: Self-review and commit**

Inspect `git diff --check` and the complete task diff. Commit: `fix: make KDE segments mutually exclusive`.

### Task 2: Core input handling and public API contracts

**Files:**
- Modify: `metricsifter/sifter.py:20-102,169-180,222-250`
- Modify: `metricsifter/algo/detection.py:68-150,185-212`
- Test: `tests/test_detection.py`
- Test: `tests/test_sifter_extended.py`

- [ ] **Step 1: Add failing tests for sparse NaN, constants, empty input, ordering, duplicate names, and numeric validation**

```python
def test_filter_keeps_changes_separated_by_nan():
    data = pd.DataFrame({"changing": [1.0, np.nan, 2.0, np.nan, 10.0]})
    assert list(Sifter._filter_no_changes(data, n_jobs=1).columns) == ["changing"]

@pytest.mark.parametrize("method", ["pelt", "binseg", "bottomup"])
def test_constant_series_has_no_change_points(method):
    assert detect_univariate_changepoints(np.ones(100), method, "l2", "bic", 2.0) == []

def test_empty_series_has_no_change_points():
    assert detect_univariate_changepoints(np.array([], dtype=float), "pelt", "l2", "bic", 2.0) == []

def test_run_upto_cpd_preserves_input_column_order():
    step = np.r_[np.zeros(50), np.ones(50) * 5]
    data = pd.DataFrame({name: step for name in ["z", "a", "m", "q", "b"]})
    assert list(Sifter(n_jobs=1).run_upto_cpd(data).columns) == list(data.columns)

def test_duplicate_columns_are_rejected():
    data = pd.DataFrame(np.ones((10, 2)), columns=["metric", "metric"])
    with pytest.raises(ValueError, match="unique"):
        Sifter(n_jobs=1).sift(data)

@pytest.mark.parametrize(
    ("kwargs", "name"),
    [({"penalty_adjust": 0}, "penalty_adjust"), ({"penalty_adjust": -1}, "penalty_adjust"),
     ({"bandwidth": 0}, "bandwidth"), ({"bandwidth": float("nan")}, "bandwidth"),
     ({"penalty": "unknown"}, "penalty")],
)
def test_invalid_numeric_parameters_are_rejected(kwargs, name):
    with pytest.raises(ValueError, match=name):
        Sifter(**kwargs)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_detection.py tests/test_sifter_extended.py`

Expected: the new tests fail for the reported current behavior.

- [ ] **Step 3: Implement minimal input handling**

In `Sifter.__init__`, validate recognized penalties and finite positive numeric values. Add one DataFrame validator called by `sift()` and `run_upto_cpd()`:

```python
if isinstance(penalty, str) and penalty not in {"aic", "bic"}:
    raise ValueError("penalty must be 'aic', 'bic', or a positive float")
for name, value in (("penalty_adjust", penalty_adjust), ("bandwidth", bandwidth)):
    if not isinstance(value, str) and (not np.isfinite(value) or value <= 0):
        raise ValueError(f"{name} must be a finite positive number")

@staticmethod
def _validate_data(data: pd.DataFrame) -> None:
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame")
    if not data.columns.is_unique:
        raise ValueError("data columns must be unique metric names")
```

Rewrite the filter around non-missing observations. Preserve the historical arithmetic-progression removal only when the full series has no missing gaps. In CPD, return `[]` before `_detect_changepoints_with_missing_values` for an empty array and after preparation when the finite core is constant. Replace the `set` selection in `run_upto_cpd()` with an input-order comprehension.

- [ ] **Step 4: Verify GREEN and full core tests**

Run: `JOBLIB_MULTIPROCESSING=0 .venv/bin/pytest -q tests/test_detection.py tests/test_sifter_extended.py tests/test_regression.py tests/test_algorithms.py`

Expected: all pass with no warning from invalid downstream penalties.

- [ ] **Step 5: Self-review and commit**

Check that existing flat-with-NaN and arithmetic-progression tests remain green. Commit: `fix: harden sifter input handling`.

### Task 3: Experiment code and test isolation

**Files:**
- Modify: `experiments/localization/rcd.py`
- Modify: `experiments/localization/pyrca.py`
- Modify: `tests/test_sifter.py`
- Modify: `tests/test_transformer.py`
- Create: `tests/test_experiment_sources.py`

- [ ] **Step 1: Add failing tests without importing optional PyRCA**

Replace `tests/test_sifter.py`'s PyRCA generator with `tests.conftest.make_synthetic`. Add AST/static checks that load `rcd.py` and assert required imports exist, and inspect `pyrca.py` to ensure the unsupported-building-step error references `building_step`. Narrow `_import_sklearn_or_skip` to `pytest.importorskip(module)` without catching arbitrary exceptions.

```python
def test_rcd_source_defines_every_runtime_dependency():
    source = Path("experiments/localization/rcd.py").read_text()
    tree = ast.parse(source)
    imported = imported_names(tree)
    assert {"zscore", "RCD", "RCDConfig", "gsq", "threadpool_limits", "joblib", "defaultdict"} <= imported

def test_unknown_building_step_reports_the_step_name():
    source = Path("experiments/localization/pyrca.py").read_text()
    assert 'f"Model {building_step} is not supported."' in source
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_sifter.py tests/test_transformer.py tests/test_experiment_sources.py`

Expected: current `test_sifter.py` fails collection without PyRCA and source checks fail.

- [ ] **Step 3: Add explicit experiment imports and correct the error**

Use imports matching the installed experiment dependencies:

```python
from collections import defaultdict
import joblib
from pyrca.analyzers.rcd import RCD, RCDConfig
from pyrca.thirdparty.causallearn.utils.cit import gsq
from scipy.stats import zscore
from threadpoolctl import threadpool_limits
```

Change the unsupported-model error to `f"Model {building_step} is not supported."`. Use Black formatting for edited experiment files. Rewrite Sifter tests to use deterministic local data and assert the expected selected/reduced schema rather than depending on PyRCA simulation.

- [ ] **Step 4: Verify GREEN**

Run: `JOBLIB_MULTIPROCESSING=0 .venv/bin/pytest -q tests`

Expected: all tests collect and pass without PyRCA installed; sklearn is skipped only when unavailable.

- [ ] **Step 5: Self-review and commit**

Run `.venv/bin/ruff check experiments/localization tests/test_sifter.py tests/test_transformer.py tests/test_experiment_sources.py`. Commit: `fix: isolate optional experiment dependencies`.

### Task 4: Dependency and lockfile consistency

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `requirements-dev.txt`
- Modify: `experiments/requirements.txt`

- [ ] **Step 1: Record failing dependency checks**

Run: `uv lock --check` and `python -m pip install --dry-run -r requirements-dev.txt`.

Expected: lock metadata reports project `0.1.0` while `pyproject.toml` is `0.2.0`; requirements include fails because root `requirements.txt` does not exist.

- [ ] **Step 2: Normalize dependency declarations**

Remove `networkx` from core dependencies and add it to `experiments`. Make `requirements-dev.txt` contain only `-e ".[dev]"` plus the documented optional PyRCA Git dependency if full experiments are requested separately. Remove `metricsifter @ ...@main` from `experiments/requirements.txt` and document `pip install -e ".[experiments]"` from repository root.

- [ ] **Step 3: Regenerate and verify the lock**

Run: `uv lock` followed by `uv lock --check`.

Expected: `uv.lock` contains `metricsifter` version `0.2.0`, reflects the moved `networkx`, and the check exits 0.

- [ ] **Step 4: Verify package dependency boundary**

Run: `uv build`; inspect wheel `METADATA` and assert core `Requires-Dist` does not contain `networkx`, while the experiments extra does.

- [ ] **Step 5: Self-review and commit**

Check `git diff --check` and commit: `build: synchronize project dependencies`.

### Task 5: CI and release gates

**Files:**
- Modify: `.github/workflows/ci.yaml`
- Modify: `.github/workflows/publish.yaml`

- [ ] **Step 1: Define workflow assertions before editing**

Add a lightweight YAML/source test in `tests/test_workflows.py` that checks:

```python
assert "pull_request:" in ci
assert "uv lock --check" in ci
assert "ruff check ." in ci
assert "black --check ." in ci
assert "uv build" in ci
assert "metricsifter --help" in ci
assert "pytest" in ci and "3.14" in ci
assert "pytest" in publish
assert "uv version --output-format json" in publish or "project.version" in publish
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_workflows.py`.

Expected: workflow gates are missing.

- [ ] **Step 3: Implement CI jobs**

Set explicit triggers for `push` and `pull_request`. Add a `quality` job for `uv lock --check`, Ruff, and Black. Run core pytest on every supported Python using `uv sync --locked --all-extras`. Add a `package` job that builds distributions, creates a fresh venv, installs the wheel without the source checkout on `PYTHONPATH`, and runs `python -c "import metricsifter"` plus `metricsifter --help`.

- [ ] **Step 4: Gate publication on tested artifacts**

In `publish.yaml`, before upload: install locked dev dependencies, run pytest/Ruff/Black, compare `${GITHUB_REF_NAME#v}` with `pyproject.toml` version, build once, install that wheel in a fresh venv, and run import/CLI smoke tests. Publish only the artifact produced by this verified job.

- [ ] **Step 5: Validate workflows and commit**

Run `.venv/bin/pytest -q tests/test_workflows.py`, parse both files with an available YAML parser when present, and inspect GitHub expression escaping. Commit: `ci: enforce test and release quality gates`.

### Task 6: Documentation, formatting, and repository-wide verification

**Files:**
- Modify: `README.md`
- Modify: `experiments/README.md`
- Modify: Python files reported by Black/Ruff
- Test: all tests and built artifacts

- [ ] **Step 1: Update user and contributor documentation**

Document `uv sync --all-extras`, the optional PyRCA Git install only for RCA experiments, full pytest behavior without PyRCA, supported Python test matrix, and actual single-PyPI Trusted Publisher environment `metricsifter_pypi`. Remove claims that TestPyPI is automatically used unless a TestPyPI job exists.

- [ ] **Step 2: Complete experiment execution instructions**

Document running from `experiments/` with repository root installed editable, the dataset download/unpack location, and concrete Python entry snippets using `dataset.loader`, `sweeper.sweeper`, or notebooks that exist in the tree. Do not invent a nonexistent CLI.

- [ ] **Step 3: Format the repository**

Run: `.venv/bin/black .` followed by `.venv/bin/ruff check . --fix` and inspect every non-mechanical change.

- [ ] **Step 4: Run fresh final verification**

Run all of:

```bash
JOBLIB_MULTIPROCESSING=0 .venv/bin/pytest -q tests
.venv/bin/ruff check .
.venv/bin/black --check .
uv lock --check
uv build
python -m venv /tmp/metricsifter-wheel-smoke
/tmp/metricsifter-wheel-smoke/bin/pip install dist/metricsifter-*.whl
(cd /tmp && /tmp/metricsifter-wheel-smoke/bin/python -c "import metricsifter; print(metricsifter.__version__)")
(cd /tmp && /tmp/metricsifter-wheel-smoke/bin/metricsifter --help)
```

Expected: every command exits 0; pytest reports no failures/errors; imported version is `0.2.0`.

- [ ] **Step 5: Self-review and commit**

Compare the completed diff line-by-line with the design scope and confirm no placeholder or unrelated refactor. Commit: `docs: align development and release guidance`.

### Task 7: Final independent review and delivery

**Files:** all changes since `892a887`

- [ ] **Step 1: Dispatch final spec review**

Provide the complete design requirements and commit range to a fresh reviewer. Any missing or extra behavior returns to the responsible implementer and is re-reviewed until accept.

- [ ] **Step 2: Dispatch final code-quality review**

Review error contracts, numerical edge cases, workflow security, dependency boundaries, test quality, and documentation consistency. Resolve every Critical/Important finding and re-review until accept.

- [ ] **Step 3: Re-run final verification after the last review fix**

Repeat the exact commands from Task 6 Step 4 and record outputs.

- [ ] **Step 4: Push and create one PR**

Push `codex/fix-codebase-issues`, create a PR against `main`, include topic summaries, reviewer accept status, and exact verification commands.

- [ ] **Step 5: Monitor CI and merge**

Wait for every required check. For failures, inspect logs, identify the root cause, add a regression test when behavior changed, fix, re-review, and push. When all checks succeed, squash-merge the PR and verify the merge commit is present on `main`.
