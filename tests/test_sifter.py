"""
Test suites for sifter
"""

import pandas as pd
import pytest

from metricsifter.sifter import Sifter
from tests.sample_gen.generator import generate_synthetic_data


@pytest.fixture(scope="module")
def synthetic_data_20() -> pd.DataFrame:
    normal_data, abonormal_data, _, _, _ = generate_synthetic_data(num_node=20, num_edge=20, num_normal_samples=60, num_abnormal_samples=20, anomaly_type=0)
    return pd.concat([normal_data, abonormal_data], axis=0)

def test_sifter_run(synthetic_data_20):
    data = synthetic_data_20
    sifter = Sifter(n_jobs=1)
    siftered_data = sifter.run(data)
    assert siftered_data.shape[1] < data.shape[1], "The number of columns should be reduced."

def test_sifter_run_upto_cpd(synthetic_data_20):
    data = synthetic_data_20
    sifter = Sifter(n_jobs=1)
    siftered_data = sifter.run_upto_cpd(data)
    assert siftered_data.shape[1] < data.shape[1], "The number of columns should be reduced."

def test_sifter_run_with_selected_segment(synthetic_data_20):
    """Verify that run_with_selected_segment() correctly returns the selected segment information"""
    data = synthetic_data_20
    sifter = Sifter(n_jobs=1)
    filtered_data, selected_segment = sifter.run_with_selected_segment(data)

    # Basic validation
    assert filtered_data.shape[1] < data.shape[1], "The number of metrics should be reduced"

    # Segment information validation
    from metricsifter.types import Segment
    if selected_segment is not None:
        assert isinstance(selected_segment, Segment), "Segment should be a Segment instance"
        assert selected_segment.start_time <= selected_segment.end_time, "Start time should be less than or equal to end time"
        assert 0 <= selected_segment.start_time < len(data), "Start time should be within data range"
        assert 0 <= selected_segment.end_time < len(data), "End time should be within data range"
        assert selected_segment.label >= 0, "Segment label should be non-negative"

def test_sifter_run_with_selected_segment_no_change_points():
    """Verify behavior when no change points are detected"""
    # Completely constant data
    constant_data = pd.DataFrame({
        'metric1': [1.0] * 100,
        'metric2': [2.0] * 100,
    })

    sifter = Sifter(n_jobs=1)
    filtered_data, selected_segment = sifter.run_with_selected_segment(constant_data)

    assert filtered_data.empty, "Should return an empty DataFrame when no change points"
    assert selected_segment is None, "Should return None when no change points"
