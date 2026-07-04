"""Tests for the Phase-5 algorithm improvements.

Covers the three configurable knobs added to the pipeline:

* ``sigma_estimator`` (robust penalty) in change-point detection,
* ``bandwidth`` rule-of-thumb auto-estimation in KDE segmentation, and
* a pluggable ``segment_selection_method`` callable.

All random data uses a fixed seed so the assertions are deterministic.
"""

import numpy as np
import pytest

from metricsifter import Sifter
from metricsifter.algo.detection import _estimate_sigma, detect_univariate_changepoints
from metricsifter.types import SegmentCandidate
from tests.conftest import make_synthetic


def _near(cps, target, tol=3) -> bool:
    return any(abs(cp - target) <= tol for cp in cps)


class TestSigmaEstimator:
    """Robust noise-scale estimators behind the AIC/BIC penalty."""

    def test_all_estimators_detect_clear_level_shift(self):
        """The basic case (a clean level shift) must be caught by every estimator."""
        x = np.concatenate([np.ones(50), np.ones(50) * 5.0])
        for est in ("std", "mad", "diff_std"):
            cps = detect_univariate_changepoints(x, "pelt", "l2", "bic", 2.0, sigma_estimator=est)
            assert _near(cps, 50), f"{est} should detect the level shift at t=50"

    def test_mad_recovers_shift_that_outliers_hide_from_std(self):
        """Transient outliers inflate the global std and hide a genuine shift.

        ``std`` over-penalizes (the spikes enlarge sigma) and misses the real
        level shift at t=75, whereas the median-based ``mad`` is unaffected by a
        minority of outliers and recovers it.
        """
        rng = np.random.default_rng(3)
        x = rng.normal(0, 0.1, 150)
        x[75:] += 1.0  # genuine, modest level shift (the target change point)
        for idx in (12, 33, 58, 132):
            x[idx] += 15.0  # sparse transient outliers

        std_cps = detect_univariate_changepoints(x, "pelt", "l2", "bic", 2.0, sigma_estimator="std")
        mad_cps = detect_univariate_changepoints(x, "pelt", "l2", "bic", 2.0, sigma_estimator="mad")

        assert not _near(std_cps, 75), "std should miss the shift hidden by outlier-inflated sigma"
        assert _near(mad_cps, 75), "mad should recover the genuine level shift at t=75"

    def test_diff_std_recovers_shift_that_a_larger_shift_hides_from_std(self):
        """A huge level shift inflates the global std and hides a second, moderate shift.

        ``std`` catches only the dominant shift at t=50; the ``+1`` shift at t=120
        is swamped by the inflated penalty. First differencing makes ``diff_std``
        depend only on the noise floor, so it recovers both.
        """
        rng = np.random.default_rng(5)
        x = rng.normal(0, 0.1, 200)
        x[50:] += 10.0  # dominant shift -> inflates std
        x[120:] += 1.0  # moderate shift -> the target change point

        std_cps = detect_univariate_changepoints(x, "pelt", "l2", "bic", 2.0, sigma_estimator="std")
        diff_cps = detect_univariate_changepoints(x, "pelt", "l2", "bic", 2.0, sigma_estimator="diff_std")

        assert _near(std_cps, 50), "std should still catch the dominant shift"
        assert not _near(std_cps, 120), "std should miss the moderate shift hidden by the dominant one"
        assert _near(diff_cps, 50) and _near(diff_cps, 120), "diff_std should recover both shifts"

    def test_estimate_sigma_values(self):
        """Sanity-check the closed-form scaling of each estimator on clean Gaussian noise."""
        rng = np.random.default_rng(0)
        core = rng.normal(0.0, 1.0, 5000)
        assert _estimate_sigma(core, "std") == pytest.approx(1.0, abs=0.05)
        assert _estimate_sigma(core, "mad") == pytest.approx(1.0, abs=0.05)
        assert _estimate_sigma(core, "diff_std") == pytest.approx(1.0, abs=0.05)

    def test_degenerate_mad_falls_back_to_std(self):
        """MAD is exactly 0 when >50% of samples share one value; the penalty must not become 0."""
        # A constant baseline with sparse spikes: median-centered MAD is exactly 0.
        x = np.full(100, 1.0)
        x[[10, 40, 70]] = 8.0

        assert _estimate_sigma(x, "mad") == pytest.approx(float(np.nanstd(x)))
        # With the fallback the detector must not over-segment the baseline.
        cps = detect_univariate_changepoints(x, "pelt", "l2", "bic", 2.0, sigma_estimator="mad")
        assert len(cps) <= 6

    def test_degenerate_diff_std_falls_back_to_std(self):
        """An exact integer-step ramp has a constant diff (diff_std exactly 0); fall back to nanstd."""
        x = np.arange(100, dtype=float)
        assert _estimate_sigma(x, "diff_std") == pytest.approx(float(np.nanstd(x)))

    def test_invalid_estimator_raises_in_detect(self):
        x = np.concatenate([np.ones(50), np.ones(50) * 5.0])
        with pytest.raises(ValueError, match="sigma_estimator"):
            detect_univariate_changepoints(x, "pelt", "l2", "bic", 2.0, sigma_estimator="bogus")

    def test_invalid_estimator_raises_in_sifter(self):
        with pytest.raises(ValueError, match="sigma_estimator"):
            Sifter(sigma_estimator="bogus")

    def test_default_estimator_is_std(self):
        assert Sifter().sigma_estimator == "std"


class TestBandwidth:
    """KDE bandwidth accepts a float or a data-driven rule-of-thumb name."""

    @pytest.mark.parametrize("rule", ["scott", "silverman"])
    def test_rule_of_thumb_keeps_failure_metrics(self, rule):
        data = make_synthetic()
        result = Sifter(bandwidth=rule, n_jobs=1).sift(data)
        # The rule-of-thumb bandwidth must still produce a sensible reduction that
        # retains the failure-related metrics.
        assert 0 < len(result.selected_metrics) < data.shape[1]
        for i in range(3):
            assert f"failure_{i}" in result.selected_metrics

    def test_numeric_bandwidth_backward_compatible(self):
        data = make_synthetic()
        default = Sifter(n_jobs=1).sift(data)
        explicit = Sifter(bandwidth=2.5, n_jobs=1).sift(data)
        assert default.selected_metrics == explicit.selected_metrics

    def test_invalid_bandwidth_string_raises(self):
        with pytest.raises(ValueError, match="bandwidth"):
            Sifter(bandwidth="not_a_rule")


class TestCustomSegmentSelection:
    """A custom Callable[[SegmentCandidate], float] can override segment selection."""

    def test_callable_changes_selection(self):
        data = make_synthetic()

        # Default (weighted_max) selects the dense failure cluster at t~60.
        default = Sifter(n_jobs=1).sift(data)
        assert "failure_0" in default.selected_metrics
        assert "unrelated" not in default.selected_metrics

        # A callable preferring the segment with the *fewest* metrics flips the
        # choice to the single-metric "unrelated" cluster at t~20.
        fewest = Sifter(segment_selection_method=lambda c: -len(c.metrics), n_jobs=1).sift(data)
        assert "unrelated" in fewest.selected_metrics
        assert "failure_0" not in fewest.selected_metrics

    def test_callable_receives_well_formed_candidates(self):
        data = make_synthetic()
        seen: list[SegmentCandidate] = []

        def scorer(candidate: SegmentCandidate) -> float:
            seen.append(candidate)
            return float(len(candidate.metrics))  # equivalent to "max"

        Sifter(segment_selection_method=scorer, n_jobs=1).sift(data)

        assert seen, "the custom scorer should be invoked for each candidate segment"
        for candidate in seen:
            assert isinstance(candidate, SegmentCandidate)
            assert len(candidate.change_points) > 0
            # metric_to_cps is restricted to exactly this segment's metrics.
            assert set(candidate.metric_to_cps) == set(candidate.metrics)

    def test_max_equivalent_callable_matches_builtin(self):
        data = make_synthetic()
        builtin = Sifter(segment_selection_method="max", n_jobs=1).sift(data)
        callable_ = Sifter(segment_selection_method=lambda c: float(len(c.metrics)), n_jobs=1).sift(data)
        assert builtin.selected_metrics == callable_.selected_metrics
