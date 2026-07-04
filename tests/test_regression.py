"""
Regression tests verifying Sifter's actual filtering behavior

Unlike the smoke tests in test_sifter.py (which only check that the number of
columns shrinks), these tests assert *which* metrics are kept or dropped.
"""

import numpy as np
import pandas as pd

from metricsifter.sifter import Sifter
from metricsifter.types import Segment


def _build_regression_data(seed: int = 42, length: int = 100) -> pd.DataFrame:
    """Build a synthetic DataFrame with a known ground truth.

    - failure_related_{0,1,2}: change points clustered around t=60 (the failure)
    - unrelated: a single change point far away from the failure cluster (t=15)
    - flat_{0,1}: constant series with no variation at all
    """
    rng = np.random.default_rng(seed)
    data = {}

    # (a) failure-related metrics: change points clustered around t=60
    for i, cp in enumerate([58, 60, 62]):
        data[f"failure_related_{i}"] = np.concatenate([rng.normal(0.0, 0.1, cp), rng.normal(10.0, 0.1, length - cp)])

    # (b) unrelated metric: single change point far away from the failure cluster
    data["unrelated"] = np.concatenate([rng.normal(0.0, 0.1, 15), rng.normal(8.0, 0.1, length - 15)])

    # (c) flat metrics: no variation at all
    data["flat_0"] = np.zeros(length)
    data["flat_1"] = np.full(length, 3.0)

    return pd.DataFrame(data)


def test_sifter_run_keeps_failure_related_and_drops_unrelated_and_flat():
    """run() should keep exactly the failure-related metrics."""
    data = _build_regression_data()
    sifter = Sifter(n_jobs=1)

    result = sifter.run(data)

    assert set(result.columns) == {"failure_related_0", "failure_related_1", "failure_related_2"}


def test_sifter_run_with_selected_segment_covers_changepoint_times():
    """run_with_selected_segment() should keep the same metrics and return a
    Segment whose range covers the clustered changepoint times."""
    data = _build_regression_data()
    sifter = Sifter(n_jobs=1)

    filtered_data, segment = sifter.run_with_selected_segment(data)

    assert set(filtered_data.columns) == {"failure_related_0", "failure_related_1", "failure_related_2"}
    assert isinstance(segment, Segment)
    # The selected segment should cover the clustered changepoints around t=60
    assert 58 <= segment.start_time <= 62
    assert 58 <= segment.end_time <= 62
