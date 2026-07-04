"""Tests for stability-selection auto-tuning of ``penalty_adjust`` and ``bandwidth``.

``penalty_adjust="auto"`` is fully deterministic (penalty-plateau detection);
``bandwidth="auto"`` uses a seeded bootstrap, so every test fixes
``random_state``. End-to-end assertions target "picks a sensible value with the
expected diagnostic report" rather than exact golden floats, plus unit tests
that pin the selection rules themselves on hand-built inputs.
"""

import json

import numpy as np
import pandas as pd
import pytest

from metricsifter import Sifter, SifterTransformer
from metricsifter.algo.detection import (
    PENALTY_ADJUST_FALLBACK,
    PENALTY_ADJUST_GRID,
    _univariate_penalty_path,
    detect_univariate_changepoints,
    select_penalty_adjust,
)
from metricsifter.algo.segmentation import BANDWIDTH_FALLBACK, select_bandwidth
from metricsifter.types import SiftResult
from tests.conftest import make_synthetic


def make_two_bursts(seed: int = 0) -> pd.DataFrame:
    """Two time-separated bursts: a dense one at t~60 and a sparse one at t=20."""
    rng = np.random.default_rng(seed)
    length = 100
    data: dict[str, np.ndarray] = {}
    for i in range(8):
        x = rng.normal(0, 0.1, length)
        x[60 + (i % 3) :] += 5.0
        data[f"dense_{i}"] = x
    for i in range(3):
        x = rng.normal(0, 0.1, length)
        x[20:] += 4.0
        data[f"sparse_{i}"] = x
    return pd.DataFrame(data)


class TestValidation:
    def test_auto_values_accepted(self):
        sifter = Sifter(penalty_adjust="auto", bandwidth="auto", random_state=0)
        assert sifter.penalty_adjust == "auto"
        assert sifter.bandwidth == "auto"

    def test_invalid_penalty_adjust_string_raises(self):
        with pytest.raises(ValueError, match="penalty_adjust"):
            Sifter(penalty_adjust="bogus")

    def test_invalid_bandwidth_string_still_raises(self):
        with pytest.raises(ValueError, match="bandwidth"):
            Sifter(bandwidth="not_a_rule")

    def test_defaults_do_not_tune(self):
        result = Sifter(n_jobs=1).sift(make_synthetic())
        assert result.penalty_tuning is None
        assert result.bandwidth_tuning is None


class TestPenaltyPlateauSelection:
    """Unit tests of select_penalty_adjust on hand-built penalty paths."""

    def test_clear_plateau_midpoint_is_selected(self):
        grid = PENALTY_ADJUST_GRID
        # 20 metrics: results churn for g<3, are constant for 3<=g<=9, then vanish.
        paths = []
        for m in range(20):
            path = []
            for g in range(len(grid)):
                if g < 3:
                    path.append([10 + 7 * g, 50, 70 - 5 * g])  # churning regime
                elif g <= 9:
                    path.append([50])  # the plateau
                else:
                    path.append([])  # missed-detection tail
            paths.append(path)

        resolved, diag = select_penalty_adjust(paths, series_length=100)

        assert diag["reason"] == "plateau"
        assert diag["plateau"] == (grid[3], grid[9])
        assert resolved == grid[6] == 2.0  # midpoint of the plateau
        assert len(diag["adjacent_jaccard"]) == len(grid) - 1

    def test_no_plateau_falls_back_to_default(self):
        grid = PENALTY_ADJUST_GRID
        # Every adjacent pair differs completely: no plateau anywhere.
        paths = [[[(g + 1) * 10 * (m + 1) % 97] for g in range(len(grid))] for m in range(5)]
        resolved, diag = select_penalty_adjust(paths, series_length=1000)
        assert resolved == PENALTY_ADJUST_FALLBACK
        assert diag["reason"] == "no_plateau"
        assert diag["plateau"] is None

    def test_empty_results_never_form_a_plateau(self):
        # "Nothing detected everywhere" is trivially stable but must not count.
        paths = [[[] for _ in PENALTY_ADJUST_GRID] for _ in range(5)]
        resolved, diag = select_penalty_adjust(paths, series_length=100)
        assert resolved == PENALTY_ADJUST_FALLBACK
        assert diag["reason"] == "no_plateau"

    def test_tolerance_absorbs_one_sample_jitter(self):
        # cps drift by 1 sample between adjacent grid points; with the 1% (>=1)
        # tolerance the whole grid is one plateau.
        grid = PENALTY_ADJUST_GRID
        paths = [[[50 + (g % 2)] for g in range(len(grid))] for _ in range(3)]
        resolved, diag = select_penalty_adjust(paths, series_length=100)
        assert diag["reason"] == "plateau"
        assert diag["plateau"] == (grid[0], grid[-1])
        assert all(j == 1.0 for j in diag["adjacent_jaccard"])


class TestPenaltyPath:
    def test_path_matches_direct_detection(self):
        """The fit-once/predict-many sweep must equal one-shot detection."""
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.1, 120)
        x[80:] += 3.0
        path, mv_cps = _univariate_penalty_path(x, "pelt", "l2", "bic", PENALTY_ADJUST_GRID, "std")

        assert mv_cps == []
        for g, adjust in enumerate(PENALTY_ADJUST_GRID):
            direct = detect_univariate_changepoints(x, "pelt", "l2", "bic", adjust)
            assert path[g] == direct, f"sweep and direct detection disagree at adjust={adjust}"

    def test_missing_value_boundaries_are_kept_separate(self):
        x = np.concatenate([np.full(10, np.nan), np.ones(40), np.ones(50) * 5.0])
        path, mv_cps = _univariate_penalty_path(x, "pelt", "l2", "bic", PENALTY_ADJUST_GRID, "std")
        # The leading-NaN boundary is penalty-invariant and reported separately,
        # never inside the per-grid paths.
        assert mv_cps == [0]
        assert all(0 not in p for p in path)
        # Detected positions are remapped past the leading-NaN trim (shift at t=50).
        assert any(45 <= cp <= 55 for cp in path[0])

    def test_all_nan_series_yields_empty_path(self):
        x = np.full(30, np.nan)
        path, mv_cps = _univariate_penalty_path(x, "pelt", "l2", "bic", PENALTY_ADJUST_GRID, "std")
        assert all(p == [] for p in path)
        assert mv_cps == [0]


class TestPenaltyAutoEndToEnd:
    def test_auto_selects_within_grid_and_keeps_failures(self):
        data = make_synthetic()
        result = Sifter(penalty_adjust="auto", n_jobs=1).sift(data)

        tuning = result.penalty_tuning
        assert tuning is not None
        assert tuning.requested == "auto"
        assert tuning.reason == "plateau"
        assert tuning.resolved in tuning.grid
        assert tuning.plateau is not None
        for i in range(3):
            assert f"failure_{i}" in result.selected_metrics

    def test_auto_equals_fixed_run_at_resolved_value(self):
        """The change points assembled from the sweep must equal a fixed run."""
        data = make_synthetic()
        auto = Sifter(penalty_adjust="auto", n_jobs=1).sift(data)
        fixed = Sifter(penalty_adjust=auto.penalty_tuning.resolved, n_jobs=1).sift(data)

        assert auto.selected_metrics == fixed.selected_metrics
        assert auto.metric_to_change_points == fixed.metric_to_change_points

    def test_auto_is_deterministic(self):
        data = make_synthetic()
        a = Sifter(penalty_adjust="auto", n_jobs=1).sift(data)
        b = Sifter(penalty_adjust="auto", n_jobs=1).sift(data)
        assert a.penalty_tuning.resolved == b.penalty_tuning.resolved
        assert a.selected_metrics == b.selected_metrics

    def test_run_upto_cpd_supports_auto(self):
        data = make_synthetic()
        out = Sifter(penalty_adjust="auto", n_jobs=1).run_upto_cpd(data)
        assert {"failure_0", "failure_1", "failure_2"} <= set(out.columns)


class TestBandwidthAutoEndToEnd:
    def test_auto_selects_dense_burst(self):
        data = make_two_bursts()
        result = Sifter(bandwidth="auto", random_state=0, n_jobs=1).sift(data)

        tuning = result.bandwidth_tuning
        assert tuning is not None
        assert tuning.requested == "auto"
        assert tuning.reason == "stability"
        assert tuning.resolved in tuning.grid
        assert len(tuning.stability) == len(tuning.grid) == len(tuning.n_segments)
        # The admissibility rule must have seen >= 2 segments at the chosen value.
        chosen = tuning.grid.index(tuning.resolved)
        assert tuning.n_segments[chosen] >= 2
        assert tuning.stability[chosen] is not None
        # The dense burst wins; the sparse burst is excluded.
        assert result.selected_metrics == frozenset(f"dense_{i}" for i in range(8))

    def test_auto_is_reproducible_with_fixed_seed(self):
        data = make_two_bursts()
        a = Sifter(bandwidth="auto", random_state=123, n_jobs=1).sift(data)
        b = Sifter(bandwidth="auto", random_state=123, n_jobs=1).sift(data)
        assert a.bandwidth_tuning.resolved == b.bandwidth_tuning.resolved
        assert a.bandwidth_tuning.to_dict() == b.bandwidth_tuning.to_dict()
        assert a.selected_metrics == b.selected_metrics

    def test_chosen_bandwidth_maximizes_stability_among_admissible(self):
        """Wiring check: resolved == argmax of the reported diagnostics."""
        data = make_two_bursts()
        tuning = Sifter(bandwidth="auto", random_state=0, n_jobs=1).sift(data).bandwidth_tuning
        admissible = [
            (score, -h) for h, score, n in zip(tuning.grid, tuning.stability, tuning.n_segments) if score is not None
        ]
        best_score, neg_h = max(admissible)
        assert tuning.resolved == -neg_h
        assert tuning.stability[tuning.grid.index(tuning.resolved)] == best_score

    def test_no_change_points_falls_back(self):
        # Constant columns are removed by the simple filter, so detection sees
        # nothing and sift() takes the early no-change-point return.
        data = pd.DataFrame({f"flat_{i}": np.full(80, float(i)) for i in range(3)})
        result = Sifter(bandwidth="auto", random_state=0, n_jobs=1).sift(data)
        tuning = result.bandwidth_tuning
        assert tuning is not None
        assert tuning.resolved == BANDWIDTH_FALLBACK
        assert tuning.reason == "no_change_points"


class TestBandwidthSelectionUnit:
    def test_too_few_change_points_falls_back(self):
        resolved, diag = select_bandwidth(
            flatten_change_points=[50, 50],
            cp_to_metrics={50: ["a", "b"]},
            metric_to_cps={"a": [50], "b": [50]},
            time_series_length=100,
            selector=lambda l2m, m2cps, l2cp: (0, set()),
            random_state=0,
        )
        assert resolved == BANDWIDTH_FALLBACK
        assert diag["reason"] == "too_few_change_points"

    def test_two_clusters_yield_admissible_candidates(self):
        metric_to_cps = {f"a_{i}": [20] for i in range(3)} | {f"b_{i}": [60 + i] for i in range(6)}
        flatten = [cp for cps in metric_to_cps.values() for cp in cps]
        cp_to_metrics: dict[int, list[str]] = {}
        for metric, cps in metric_to_cps.items():
            for cp in cps:
                cp_to_metrics.setdefault(cp, []).append(metric)

        selector = Sifter(n_jobs=1).select_largest_segment_with_label
        resolved, diag = select_bandwidth(
            flatten, cp_to_metrics, metric_to_cps, time_series_length=100, selector=selector, random_state=0
        )
        assert diag["reason"] == "stability"
        assert any(n >= 2 for n in diag["n_segments"])
        assert resolved in diag["grid"]


class TestSerializationAndIntegration:
    def test_tuning_reports_round_trip_through_json(self):
        data = make_two_bursts()
        result = Sifter(penalty_adjust="auto", bandwidth="auto", random_state=0, n_jobs=1).sift(data)
        restored = SiftResult.from_json(result.to_json())

        assert restored.penalty_tuning == result.penalty_tuning
        assert restored.bandwidth_tuning == result.bandwidth_tuning

    def test_untuned_result_serializes_null_reports(self):
        result = Sifter(n_jobs=1).sift(make_synthetic())
        payload = json.loads(result.to_json())
        assert payload["penalty_tuning"] is None
        assert payload["bandwidth_tuning"] is None

    def test_transformer_passes_auto_through(self):
        data = make_synthetic()
        tr = SifterTransformer(penalty_adjust="auto", bandwidth="auto", random_state=0, n_jobs=1).fit(data)
        assert tr.result_.penalty_tuning is not None
        assert tr.result_.bandwidth_tuning is not None
        assert "random_state" in tr.get_params()


class TestCLI:
    def test_cli_accepts_auto_and_reports_tuning(self, tmp_path):
        from metricsifter import cli

        csv_path = tmp_path / "input.csv"
        report_path = tmp_path / "report.json"
        make_two_bursts().to_csv(csv_path, index=False)

        code = cli.main(
            [
                "run",
                str(csv_path),
                "--penalty-adjust",
                "auto",
                "--bandwidth",
                "auto",
                "--random-state",
                "0",
                "--output",
                str(tmp_path / "out.csv"),
                "--report",
                str(report_path),
            ]
        )

        assert code == cli.EXIT_OK
        payload = json.loads(report_path.read_text())
        assert payload["penalty_tuning"]["resolved"] > 0
        assert payload["bandwidth_tuning"]["resolved"] > 0

    def test_cli_rejects_bogus_penalty_adjust(self, capsys):
        from metricsifter import cli

        with pytest.raises(SystemExit):
            cli.main(["run", "whatever.csv", "--penalty-adjust", "bogus"])
