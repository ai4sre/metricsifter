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
