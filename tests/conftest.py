"""Shared synthetic-data helpers for the ecosystem-integration test suites.

All data uses a fixed seed so that change-point classification is deterministic
and does not depend on the pyrca-based generator.
"""

import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import settings

# Hypothesis profiles for the property-based tests (tests/test_properties.py).
# "deterministic" is what `make test` and every CI test job run: derandomized
# (the same examples on every run, so a red build is never a coin flip) and
# bounded. "fuzz" is for exploration (`make test-fuzz`, the scheduled CI job):
# fresh random examples, a larger budget (HYPOTHESIS_MAX_EXAMPLES), and the
# example database (.hypothesis/) so that a failure replays on the next run.
# No deadline in either profile: the machine load decides the wall time of a
# sift. print_blob prints a @reproduce_failure decorator for every failure.
settings.register_profile(
    "deterministic", max_examples=40, derandomize=True, database=None, deadline=None, print_blob=True
)
settings.register_profile(
    "fuzz", max_examples=int(os.environ.get("HYPOTHESIS_MAX_EXAMPLES", "1000")), deadline=None, print_blob=True
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "deterministic"))


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
