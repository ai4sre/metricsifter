"""Property-based tests (Hypothesis) for the sift pipeline and its input adapters.

The example-based suites pin the behaviour on hand-made fixtures; these tests
generate many small inputs instead and assert invariants that must hold for
*every* input: the exclusion reasons partition the input columns, change points
stay on the time axis, the result does not depend on column order or on
constant columns, the detector is unit-free, KDE clusters partition the change
points, and the Prometheus adapter and the CLI keep their contracts.

The Hypothesis profile comes from tests/conftest.py: derandomized and bounded
by default, exploratory under ``HYPOTHESIS_PROFILE=fuzz`` (``make test-fuzz``).

Input domain: finite floats and NaN (missing values are a documented input),
with nonzero magnitudes between 1e-100 and 1e100. The AIC/BIC penalty scales
with the variance of a metric, and the pipeline does not yet define what
happens when that variance is not a finite positive double: an infinite
sample, or magnitudes beyond about 1e154 (overflow) or below about 1e-154
(underflow to a zero penalty), make the default PELT path fail with an
AssertionError from ruptures while the other search methods return a result.
Until that behaviour is decided there is no property to assert for those inputs.
"""

from __future__ import annotations

import contextlib
import io
import json
import string
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import example, given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from metricsifter import cli
from metricsifter.adapters import prometheus
from metricsifter.algo import detection, segmentation
from metricsifter.algo.detection import SEARCH_METHODS, SIGMA_ESTIMATORS
from metricsifter.sifter import Sifter
from metricsifter.types import SegmentCandidate, SiftResult

MAX_ROWS = 48
MAX_COLUMNS = 6


def magnitudes(low: float, high: float) -> st.SearchStrategy[float]:
    """0.0, or a float whose magnitude lies in [low, high] (see the module docstring)."""
    return st.one_of(st.just(0.0), st.floats(low, high).flatmap(lambda m: st.sampled_from([m, -m])))


FINITE = magnitudes(1e-100, 1e100)
# Moderate magnitudes for structured signals, so that a level shift is not lost in rounding.
LEVEL = magnitudes(1e-100, 1e6)
# Jitter around a level; the product of two of them stays above 1e-100 in magnitude.
JITTER = magnitudes(1e-50, 1.0)


def _widest_change_points(candidate: SegmentCandidate) -> float:
    """A custom segment-selection strategy (the callable form of segment_selection_method)."""
    if not candidate.change_points:
        return 0.0
    return float(max(candidate.change_points) - min(candidate.change_points))


@st.composite
def metric_values(draw: st.DrawFn, n_rows: int) -> np.ndarray:
    """One metric: noise, a level shift, a constant, a ramp or nothing at all, optionally with gaps."""
    kind = draw(st.sampled_from(["noise", "step", "constant", "ramp", "all_nan"]))
    if kind == "all_nan":
        return np.full(n_rows, np.nan)
    if kind == "noise":
        values = draw(hnp.arrays(np.float64, n_rows, elements=FINITE))
    elif kind == "step":
        before, after = draw(LEVEL), draw(LEVEL)
        at = draw(st.integers(0, n_rows))
        jitter = draw(hnp.arrays(np.float64, n_rows, elements=JITTER))
        values = np.where(np.arange(n_rows) < at, before, after) + jitter * abs(draw(JITTER))
    elif kind == "constant":
        values = np.full(n_rows, draw(LEVEL))
    else:
        values = draw(LEVEL) + draw(LEVEL) * np.arange(n_rows, dtype=np.float64)
    if n_rows and draw(st.booleans()):
        values = np.where(draw(hnp.arrays(np.bool_, n_rows)), np.nan, values)
    return np.asarray(values, dtype=np.float64)


@st.composite
def frames(draw: st.DrawFn) -> pd.DataFrame:
    """A DataFrame of metrics with unique (sometimes awkward) names and an optional DatetimeIndex."""
    n_rows = draw(st.integers(0, MAX_ROWS))
    names = draw(
        st.lists(
            st.text(alphabet=string.ascii_letters + string.digits + '_{}=",#. ', min_size=1, max_size=8),
            max_size=MAX_COLUMNS,
            unique=True,
        )
    )
    df = pd.DataFrame({name: draw(metric_values(n_rows)) for name in names}, index=pd.RangeIndex(n_rows))
    if draw(st.booleans()):
        df.index = pd.date_range("2024-01-01", periods=n_rows, freq="1min")
    return df


# Sifter keyword arguments (drawn as a dict so that a failure report shows them).
sifter_kwargs = st.fixed_dictionaries(
    {
        "search_method": st.sampled_from(sorted(SEARCH_METHODS)),
        "penalty": st.one_of(st.sampled_from(["bic", "aic"]), st.floats(0.01, 1000.0)),
        "penalty_adjust": st.one_of(st.sampled_from([2.0, "auto"]), st.floats(0.1, 10.0)),
        "bandwidth": st.one_of(st.sampled_from([2.5, "scott", "silverman", "auto"]), st.floats(0.5, 20.0)),
        "segment_selection_method": st.sampled_from(["max", "weighted_max", _widest_change_points]),
        "sigma_estimator": st.sampled_from(sorted(SIGMA_ESTIMATORS)),
        "n_jobs": st.just(1),
        "random_state": st.just(0),
    }
)

# Sifters whose penalty is derived from the data scale (an explicit numeric
# penalty is in the data's squared units, so it is not scale-free).
data_scaled_detectors = st.tuples(
    st.sampled_from(sorted(SEARCH_METHODS)),
    st.sampled_from(["bic", "aic"]),
    st.sampled_from(sorted(SIGMA_ESTIMATORS)),
    st.floats(0.1, 10.0),
)


def _comparable(result: SiftResult) -> dict:
    """Everything a sift reports except the DataFrame and the per-metric order."""
    report = result.to_dict()
    report["metric_to_change_points"] = dict(sorted(report["metric_to_change_points"].items()))
    if report["metric_to_change_times"] is not None:
        report["metric_to_change_times"] = dict(sorted(report["metric_to_change_times"].items()))
    return report


@given(data=frames(), params=sifter_kwargs, without_simple_filter=st.booleans())
def test_sift_result_is_consistent(data: pd.DataFrame, params: dict, without_simple_filter: bool) -> None:
    result = Sifter(**params).sift(data, without_simple_filter=without_simple_filter)
    columns = set(data.columns)
    n_rows = len(data)

    # The selection and the three exclusion reasons partition the input columns.
    parts = [
        result.selected_metrics,
        result.filtered_no_change,
        result.filtered_no_change_points,
        result.filtered_out_of_segment,
    ]
    assert set().union(*parts) == columns
    assert sum(len(part) for part in parts) == len(columns)
    if without_simple_filter:
        assert result.filtered_no_change == frozenset()

    # data holds exactly the selected columns, unchanged and in input order.
    selected_in_order = [c for c in data.columns if c in result.selected_metrics]
    assert result.data is not None
    assert list(result.data.columns) == selected_in_order
    if selected_in_order:
        pd.testing.assert_frame_equal(result.data, data[selected_in_order])

    # Change points: one entry per detected metric, sorted, unique, on the time axis.
    assert set(result.metric_to_change_points) == columns - result.filtered_no_change
    for metric, cps in result.metric_to_change_points.items():
        assert cps == sorted(set(cps))
        assert all(isinstance(cp, int) and 0 <= cp < n_rows for cp in cps)
        assert (len(cps) == 0) == (metric in result.filtered_no_change_points)
    if isinstance(data.index, pd.DatetimeIndex):
        assert result.metric_to_change_times == {
            metric: [data.index[cp] for cp in cps] for metric, cps in result.metric_to_change_points.items()
        }
    else:
        assert result.metric_to_change_times is None

    # Segments are disjoint, ordered intervals; the selected one is the best-scoring one.
    with_change_points = columns - result.filtered_no_change - result.filtered_no_change_points
    for segment in result.segments:
        assert 0 <= segment.start_index <= segment.end_index < n_rows
        assert segment.metrics and segment.metrics <= with_change_points
    for left, right in zip(result.segments, result.segments[1:]):
        assert left.label < right.label
        assert left.end_index < right.start_index
    flagged = [segment for segment in result.segments if segment.selected]
    if result.selected_metrics:
        assert result.selected_segment is not None and flagged == [result.selected_segment]
        assert result.selected_segment.metrics == result.selected_metrics
        assert result.selected_segment.score == max(segment.score for segment in result.segments)
    else:
        assert not with_change_points
        assert flagged == [] and result.selected_segment is None and result.segments == []

    # The JSON report round-trips.
    assert SiftResult.from_json(result.to_json()).to_dict() == result.to_dict()


def _flat_valley() -> pd.DataFrame:
    """Change points at 18, 20 | 23, 25: the KDE valley between them is flat (21 and 22 tie)."""
    level_shifts = np.zeros(28), np.zeros(28)
    level_shifts[0][23] = 1.0
    level_shifts[1][18:20] = 1.0
    return pd.DataFrame({"3": level_shifts[0], "00": level_shifts[1]})


@given(data=frames(), params=sifter_kwargs, seed=st.integers(0, 2**32 - 1))
# Found by `make test-fuzz`: before the KDE was fitted on sorted change points,
# rounding in its binning split the flat valley for one column order only.
@example(
    data=_flat_valley(),
    params={
        "search_method": "binseg",
        "penalty": "bic",
        "penalty_adjust": "auto",
        "bandwidth": 0.5,
        "segment_selection_method": "max",
        "sigma_estimator": "diff_std",
        "n_jobs": 1,
        "random_state": 0,
    },
    seed=3,  # swaps the two columns
)
def test_sift_does_not_depend_on_column_order(data: pd.DataFrame, params: dict, seed: int) -> None:
    order = np.random.default_rng(seed).permutation(len(data.columns))
    shuffled = data.iloc[:, order]

    expected = Sifter(**params).sift(data)
    actual = Sifter(**params).sift(shuffled)

    assert _comparable(actual) == _comparable(expected)
    assert actual.data is not None and expected.data is not None
    assert list(actual.data.columns) == [c for c in shuffled.columns if c in expected.selected_metrics]


@given(
    data=frames(),
    params=sifter_kwargs,
    constants=st.lists(LEVEL, min_size=1, max_size=3),
    gap=st.booleans(),
    without_simple_filter=st.booleans(),
)
def test_constant_columns_do_not_change_the_selection(
    data: pd.DataFrame, params: dict, constants: list[float], gap: bool, without_simple_filter: bool
) -> None:
    # A missing value is a change point in its own right, so a gap only keeps a
    # constant column inert when the no-variation filter removes it first.
    gap = gap and not without_simple_filter and len(data) > 0
    extra = {}
    for i, value in enumerate(constants):
        column = np.full(len(data), value)
        if gap:
            column[0] = np.nan
        extra[f"__constant_{i}"] = column
    padded = data.assign(**extra)

    expected = Sifter(**params).sift(data, without_simple_filter=without_simple_filter)
    actual = Sifter(**params).sift(padded, without_simple_filter=without_simple_filter)

    assert actual.selected_metrics == expected.selected_metrics
    assert actual.filtered_out_of_segment == expected.filtered_out_of_segment
    assert actual.segments == expected.segments
    added = frozenset(extra)
    if without_simple_filter:
        assert actual.filtered_no_change_points == expected.filtered_no_change_points | added
        assert actual.metric_to_change_points == {**expected.metric_to_change_points, **{c: [] for c in added}}
    else:
        assert actual.filtered_no_change == expected.filtered_no_change | added
        assert actual.metric_to_change_points == expected.metric_to_change_points


@given(
    x=st.integers(0, MAX_ROWS).flatmap(lambda n: metric_values(n)),
    method=st.sampled_from(sorted(SEARCH_METHODS)),
    penalty=st.one_of(st.sampled_from(["bic", "aic"]), st.floats(0.01, 1000.0)),
    penalty_adjust=st.floats(0.1, 10.0),
    sigma_estimator=st.sampled_from(sorted(SIGMA_ESTIMATORS)),
)
def test_detector_output_is_well_formed(
    x: np.ndarray, method: str, penalty: str | float, penalty_adjust: float, sigma_estimator: str
) -> None:
    cps = detection.detect_univariate_changepoints(x, method, "l2", penalty, penalty_adjust, sigma_estimator)

    assert cps == sorted(set(cps))
    assert all(isinstance(cp, int) and 0 <= cp < len(x) for cp in cps)
    # Where a metric goes missing is always reported.
    missing_boundaries = {int(i) for i in detection._detect_changepoints_with_missing_values(x)}
    assert missing_boundaries <= set(cps)
    observed = x[~np.isnan(x)]
    if observed.size == 0 or (observed == observed[0]).all():
        assert set(cps) == missing_boundaries


@given(
    values=st.integers(0, MAX_ROWS).flatmap(
        lambda n: hnp.arrays(np.float64, n, elements=st.integers(-(10**6), 10**6).map(float))
    ),
    gaps=st.data(),
    detector=data_scaled_detectors,
    exponent=st.integers(-30, 30),
)
def test_detector_is_unit_free(
    values: np.ndarray, gaps: st.DataObject, detector: tuple[str, str, str, float], exponent: int
) -> None:
    """Rescaling a metric by a power of two (exact in binary floating point) keeps its change points."""
    method, penalty, sigma_estimator, penalty_adjust = detector
    if values.size and gaps.draw(st.booleans()):
        values = np.where(gaps.draw(hnp.arrays(np.bool_, values.size)), np.nan, values)

    def detect(x: np.ndarray) -> list[int]:
        return detection.detect_univariate_changepoints(x, method, "l2", penalty, penalty_adjust, sigma_estimator)

    assert detect(values * 2.0**exponent) == detect(values)


@given(
    change_points=st.lists(st.integers(0, 199), min_size=1, max_size=60),
    bandwidth=st.one_of(st.floats(0.5, 50.0), st.sampled_from(["scott", "silverman"])),
)
def test_kde_segmentation_partitions_change_points(change_points: list[int], bandwidth: float | str) -> None:
    labels, label_to_values = segmentation.segment_changepoints_with_kde(
        change_points, time_series_length=200, kde_bandwidth=bandwidth, unique_values=True
    )
    points = np.asarray(change_points)

    assert len(labels) == len(change_points)
    assert sorted(label_to_values) == list(range(len(label_to_values)))
    clusters = [label_to_values[label] for label in sorted(label_to_values)]
    # Every distinct change point lands in exactly one cluster ...
    assert sorted(np.concatenate(clusters).tolist()) == sorted(set(change_points))
    for label, cluster in enumerate(clusters):
        # ... the one its label names ...
        assert set(points[labels == label].tolist()) == set(cluster.tolist())
    # ... and the clusters are ordered, non-overlapping intervals of the time axis.
    non_empty = [cluster for cluster in clusters if len(cluster)]
    for left, right in zip(non_empty, non_empty[1:]):
        assert left.max() < right.min()


# Label values: random text, plus a dictionary of values that collide once
# canonicalized (a quoted separator, or a "#<n>" disambiguation suffix).
label_text = st.one_of(
    st.text(alphabet=string.ascii_letters + string.digits + '_"=,{}#\\ é', max_size=6),
    st.sampled_from(["", "a", "a#1", "#1", "1", '1",b="2', "2"]),
)
sample_value = st.one_of(
    st.floats(allow_nan=False, allow_infinity=False).map(repr),
    st.sampled_from(["NaN", "+Inf", "-Inf", "0", "1e3"]),
)
series = st.fixed_dictionaries(
    {
        "metric": st.dictionaries(st.sampled_from(["__name__", "job", "instance", "le", "a"]), label_text, max_size=4),
        "values": st.lists(st.tuples(st.integers(1_700_000_000, 1_700_000_100), sample_value), max_size=8),
    }
)


@given(result=st.lists(series, max_size=6), label_sep=st.sampled_from([",", ", ", ";"]))
# Two series that canonicalize to the same name get a "#1" suffix, which must
# not collide with a third series that is literally named that way.
@example(
    result=[{"metric": {"__name__": "a"}, "values": []}] * 2 + [{"metric": {"__name__": "a#1"}, "values": []}],
    label_sep=",",
)
@example(
    result=[
        {"metric": {"__name__": "m", "a": '1",b="2'}, "values": [[1_700_000_000, "1"]]},
        {"metric": {"__name__": "m", "a": "1", "b": "2"}, "values": [[1_700_000_000, "2"]]},
    ],
    label_sep=",",
)
def test_prometheus_matrix_round_trips(result: list[dict], label_sep: str) -> None:
    response = {"status": "success", "data": {"resultType": "matrix", "result": result}}

    df = prometheus.from_query_range(response, label_sep=label_sep)

    # One uniquely named column per series, and each maps back to its own labels.
    assert len(df.columns) == len(result)
    assert df.columns.is_unique
    for column, item in zip(df.columns, result):
        assert prometheus.to_metric_labels(df, column) == item["metric"]

    # The index is the sorted union of the timestamps (UTC); a cell holds the
    # last sample of its series at that timestamp, NaN where there is none.
    timestamps = sorted({ts for item in result for ts, _ in item["values"]})
    assert list(df.index) == list(pd.to_datetime(timestamps, unit="s", utc=True))
    for column, item in zip(df.columns, result):
        last = {ts: float(value) for ts, value in item["values"]}
        for ts in timestamps:
            cell = df.at[pd.Timestamp(ts, unit="s", tz="UTC"), column]
            if ts in last and not np.isnan(last[ts]):
                assert cell == last[ts]
            else:
                assert np.isnan(cell)


@given(labels=st.dictionaries(st.sampled_from(["__name__", "job", "instance", "a", "b"]), label_text, min_size=1))
def test_prometheus_column_name_ignores_label_order(labels: dict[str, str]) -> None:
    reordered = dict(reversed(list(labels.items())))
    as_response = [
        {"status": "success", "data": {"resultType": "matrix", "result": [{"metric": m, "values": [[1, "1"]]}]}}
        for m in (labels, reordered)
    ]
    names = [list(prometheus.from_query_range(r).columns) for r in as_response]
    assert names[0] == names[1]


csv_cell = st.one_of(
    st.integers(-1000, 1000).map(str),
    st.floats(-1e6, 1e6, allow_nan=False).map(repr),
    # No "inf": see the module docstring.
    st.sampled_from(["", "nan", "NaN", "True", "False", "x", "2024-01-01", '"a,b"']),
)


@given(
    header=st.lists(st.text(alphabet=string.ascii_letters, min_size=1, max_size=4), max_size=4, unique=True),
    rows=st.lists(st.lists(csv_cell, min_size=0, max_size=5), max_size=12),
    index_col=st.sampled_from([[], ["--index-col", "0"]]),
)
def test_cli_reports_input_errors_with_exit_code_2(
    header: list[str], rows: list[list[str]], index_col: list[str]
) -> None:
    """``metricsifter run`` exits 0 or 2 (input error) and never escapes with a traceback."""
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "input.csv"
        source.write_text("\n".join([",".join(header)] + [",".join(row) for row in rows]) + "\n", encoding="utf-8")
        report = Path(tmp) / "report.json"
        argv = ["run", str(source), "--output", str(Path(tmp) / "out.csv"), "--report", str(report), *index_col]
        with contextlib.redirect_stderr(io.StringIO()):
            code = cli.main([*argv, "--n-jobs", "1"])

        assert code in (cli.EXIT_OK, cli.EXIT_INPUT_ERROR)
        if code == cli.EXIT_OK:
            parsed = pd.read_csv(source, index_col=0 if index_col else None)
            assert set(json.loads(report.read_text())["selected_metrics"]) <= set(parsed.columns)
