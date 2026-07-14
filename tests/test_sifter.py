"""Public Sifter behavior on deterministic, locally generated data."""

import pandas as pd

from metricsifter.sifter import Sifter


def test_sifter_run(synthetic_df):
    data = synthetic_df
    sifter = Sifter(n_jobs=1)
    sifted_data = sifter.run(data)
    assert list(sifted_data.columns) == ["failure_0", "failure_1", "failure_2"]


def test_sifter_run_upto_cpd(synthetic_df):
    data = synthetic_df
    sifter = Sifter(n_jobs=1)
    sifted_data = sifter.run_upto_cpd(data)
    assert list(sifted_data.columns) == ["failure_0", "failure_1", "failure_2", "unrelated"]


def test_sifter_run_with_selected_segment(synthetic_df):
    """Verify that run_with_selected_segment() correctly returns the selected segment information"""
    data = synthetic_df
    sifter = Sifter(n_jobs=1)
    filtered_data, selected_segment = sifter.run_with_selected_segment(data)

    from metricsifter.types import Segment

    assert list(filtered_data.columns) == ["failure_0", "failure_1", "failure_2"]
    assert isinstance(selected_segment, Segment)
    assert selected_segment.start_time == 60
    assert selected_segment.end_time == 60


def test_sifter_run_with_selected_segment_no_change_points():
    """Verify behavior when no change points are detected"""
    constant_data = pd.DataFrame({"metric1": [1.0] * 100, "metric2": [2.0] * 100})

    sifter = Sifter(n_jobs=1)
    filtered_data, selected_segment = sifter.run_with_selected_segment(constant_data)

    assert filtered_data.empty, "Should return an empty DataFrame when no change points"
    assert selected_segment is None, "Should return None when no change points"
