"""Shared synthetic-data helpers for the ecosystem-integration test suites.

All data uses a fixed seed so that change-point classification is deterministic
and does not depend on the pyrca-based generator.
"""

import numpy as np
import pandas as pd
import pytest


def make_synthetic(as_datetime: bool = False) -> pd.DataFrame:
    """Build data exercising every exclusion reason (see test_sift_result).

    - failure_{0,1,2}: level shift at t=60 -> selected (densest segment)
    - unrelated:       level shift at t=20 -> excluded (outside densest segment)
    - noise:           stationary noise    -> excluded (no change points)
    - flat_{0..5}:     constant            -> excluded (no-change filter)
    """
    rng = np.random.default_rng(0)
    length = 80
    data: dict[str, np.ndarray] = {}
    for i in range(3):
        data[f"failure_{i}"] = np.concatenate([rng.normal(0, 0.1, 60), rng.normal(5, 0.1, 20)])
    data["unrelated"] = np.concatenate([rng.normal(0, 0.1, 20), rng.normal(3, 0.1, 60)])
    data["noise"] = rng.normal(0, 0.05, length)
    for i in range(6):
        data[f"flat_{i}"] = np.full(length, float(i))

    df = pd.DataFrame(data)
    if as_datetime:
        df.index = pd.date_range("2024-01-01", periods=length, freq="1min")
    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return make_synthetic()
