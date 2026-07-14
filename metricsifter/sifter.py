from typing import Callable

import numpy as np
import pandas as pd

from metricsifter import utils
from metricsifter.algo import detection, segmentation
from metricsifter.algo.detection import SIGMA_ESTIMATORS
from metricsifter.types import BandwidthTuning, PenaltyTuning, Segment, SegmentCandidate, SegmentInfo, SiftResult

#: KDE bandwidth rule-of-thumb names accepted by ``bandwidth`` (in addition to a float).
BANDWIDTH_RULES: frozenset[str] = frozenset({"scott", "silverman"})

#: Information-criterion names accepted by ``penalty`` (in addition to a float).
PENALTY_RULES: frozenset[str] = frozenset({"aic", "bic"})

#: Sentinel that turns on stability-selection auto-tuning for a parameter.
AUTO: str = "auto"


def _validate_finite_positive(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite positive number; got {value!r}.")


class Sifter:
    def __init__(
        self,
        search_method: str = "pelt",
        cost_model: str = "l2",
        penalty: str | float = "bic",
        penalty_adjust: float | str = 2.0,
        bandwidth: float | str = 2.5,
        segment_selection_method: str | Callable[[SegmentCandidate], float] = "weighted_max",
        n_jobs: int = 1,
        sigma_estimator: str = "std",
        random_state: int | None = None,
    ) -> None:
        """Configure the feature-reduction pipeline.

        Args:
            search_method: Change-point search algorithm (``"pelt"`` / ``"binseg"``
                / ``"bottomup"``).
            cost_model: Cost model for ``binseg`` / ``bottomup`` (e.g. ``"l2"``).
            penalty: ``"bic"``, ``"aic"``, or a numeric penalty passed to ruptures.
            penalty_adjust: Multiplier applied to the derived penalty (default
                ``2.0``), or ``"auto"`` to choose it by penalty-plateau
                stability selection (see
                :func:`metricsifter.algo.detection.select_penalty_adjust`); the
                chosen value is reported in ``SiftResult.penalty_tuning``.
            bandwidth: KDE bandwidth for change-point segmentation. Either a
                ``float`` (fixed bandwidth, default ``2.5``), one of the
                data-driven rule-of-thumb names ``"scott"`` / ``"silverman"``
                (computed by statsmodels from the change-point distribution),
                or ``"auto"`` to choose it by bootstrap stability selection
                (see :func:`metricsifter.algo.segmentation.select_bandwidth`);
                the chosen value is reported in ``SiftResult.bandwidth_tuning``.
            segment_selection_method: How to pick the "densest" segment. Either a
                built-in name (``"max"`` = most metrics, ``"weighted_max"`` =
                sum of ``1 / len(change_points)`` per metric) or a custom
                ``Callable[[SegmentCandidate], float]`` whose highest-scoring
                segment is selected.
            n_jobs: Parallelism for detection/filtering (joblib convention).
            sigma_estimator: Noise-scale estimator behind the AIC/BIC penalty
                (``"std"`` / ``"mad"`` / ``"diff_std"``, default ``"std"``). Use
                ``"mad"`` for spiky/outlier-prone metrics and ``"diff_std"`` for
                trending or level-shifting metrics; see
                :func:`metricsifter.algo.detection._estimate_sigma`.
            random_state: Seed for the ``bandwidth="auto"`` bootstrap (``None``
                = OS entropy). Fix it for reproducible auto-tuning.

        Raises:
            ValueError: If a string option is unsupported, or a numeric
                ``penalty``, ``penalty_adjust`` or ``bandwidth`` is not finite
                and positive.
        """
        if sigma_estimator not in SIGMA_ESTIMATORS:
            raise ValueError(
                f"sigma_estimator={sigma_estimator!r} is not supported. " f"Choose one of {sorted(SIGMA_ESTIMATORS)}."
            )
        if isinstance(penalty, str):
            if penalty not in PENALTY_RULES:
                raise ValueError(
                    f"penalty={penalty!r} is not supported. Pass a finite positive number or one of {sorted(PENALTY_RULES)}."
                )
        else:
            _validate_finite_positive("penalty", penalty)
        if isinstance(penalty_adjust, str):
            if penalty_adjust != AUTO:
                raise ValueError(f"penalty_adjust={penalty_adjust!r} is not supported. Pass a float or {AUTO!r}.")
        else:
            _validate_finite_positive("penalty_adjust", penalty_adjust)
        if isinstance(bandwidth, str):
            if bandwidth not in BANDWIDTH_RULES | {AUTO}:
                raise ValueError(
                    f"bandwidth={bandwidth!r} is not supported. "
                    f"Pass a float or one of {sorted(BANDWIDTH_RULES | {AUTO})}."
                )
        else:
            _validate_finite_positive("bandwidth", bandwidth)
        self.search_method = search_method
        self.cost_model = cost_model
        self.bandwidth = bandwidth
        self.penalty = penalty
        self.penalty_adjust = penalty_adjust
        self.segment_selection_method = segment_selection_method
        self.n_jobs = n_jobs
        self.sigma_estimator = sigma_estimator
        self.random_state = random_state

    @staticmethod
    def _filter_no_changes(X: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
        def filter(x: pd.Series) -> bool:
            observed = x.dropna()
            if observed.empty or (observed == observed.iat[0]).all():
                return False
            if not x.hasnans:
                diff_x = np.diff(x)
                if (diff_x == diff_x[0]).all():
                    return False
            return True

        if n_jobs != 1:
            return X.loc[:, utils.parallel_apply(X, filter, n_jobs)]
        return X.loc[:, X.apply(filter)]

    @staticmethod
    def _validate_input(data: object) -> None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas DataFrame; got {type(data).__name__}.")
        if not data.columns.is_unique:
            raise ValueError("data column names must be unique.")

    def _detect_changepoints(
        self, X: pd.DataFrame
    ) -> tuple[list[int], dict[int, list[str]], dict[str, list[int]], PenaltyTuning | None]:
        """STEP1: detect change points, tuning ``penalty_adjust`` when requested."""
        if self.penalty_adjust == AUTO:
            flatten, cp_to_metrics, metric_to_cps, resolved, diag = (
                detection.detect_multi_changepoints_with_penalty_tuning(
                    X,
                    search_method=self.search_method,
                    cost_model=self.cost_model,
                    penalty=self.penalty,
                    sigma_estimator=self.sigma_estimator,
                    n_jobs=self.n_jobs,
                )
            )
            tuning = PenaltyTuning(
                requested=self.penalty_adjust,
                resolved=resolved,
                grid=diag["grid"],
                n_change_points=diag["n_change_points"],
                adjacent_jaccard=diag["adjacent_jaccard"],
                plateau=diag["plateau"],
                reason=diag["reason"],
            )
            return flatten, cp_to_metrics, metric_to_cps, tuning

        flatten, cp_to_metrics, metric_to_cps = detection.detect_multi_changepoints(
            X,
            search_method=self.search_method,
            cost_model=self.cost_model,
            penalty=self.penalty,
            penalty_adjust=float(self.penalty_adjust),
            sigma_estimator=self.sigma_estimator,
            n_jobs=self.n_jobs,
        )
        return flatten, cp_to_metrics, metric_to_cps, None

    def _resolve_bandwidth(
        self,
        flatten_change_points: list[int],
        cp_to_metrics: dict[int, list[str]],
        metric_to_cps: dict[str, list[int]],
        time_series_length: int,
    ) -> tuple[float | str, BandwidthTuning | None]:
        """Resolve the KDE bandwidth, tuning it when ``"auto"`` was requested."""
        if self.bandwidth != AUTO:
            return self.bandwidth, None
        resolved, diag = segmentation.select_bandwidth(
            flatten_change_points,
            cp_to_metrics,
            metric_to_cps,
            time_series_length=time_series_length,
            selector=self.select_largest_segment_with_label,
            random_state=self.random_state,
        )
        tuning = BandwidthTuning(
            requested=self.bandwidth,
            resolved=resolved,
            grid=diag["grid"],
            stability=diag["stability"],
            n_segments=diag["n_segments"],
            reason=diag["reason"],
        )
        return resolved, tuning

    def run_upto_cpd(self, data: pd.DataFrame, without_simple_filter: bool = False) -> pd.DataFrame:
        """Run up to change point detection.

        Raises:
            ValueError: If ``data`` is not a DataFrame or its column names are
                not unique.
        """
        self._validate_input(data)
        if without_simple_filter:
            X = data
        else:
            # STEP0: simple filter
            X = self._filter_no_changes(data, n_jobs=self.n_jobs)

        # STEP1: detect change points
        _, _, metric_to_cps, _ = self._detect_changepoints(X)
        remained_metrics = [metric for metric in X.columns if metric_to_cps.get(metric)]
        return X.loc[:, remained_metrics]

    def run(self, data: pd.DataFrame, without_simple_filter: bool = False) -> pd.DataFrame:
        """Run the feature reduction pipeline and return filtered metrics

        This method is a thin wrapper around sift() that only returns the
        filtered DataFrame for backward compatibility.

        Args:
            data: Input time series data
            without_simple_filter: If True, skip STEP0 simple filter

        Returns:
            pd.DataFrame: DataFrame containing only the selected metrics
        """
        result = self.sift(data, without_simple_filter)
        return result.data if result.data is not None else pd.DataFrame()

    def run_with_selected_segment(
        self, data: pd.DataFrame, without_simple_filter: bool = False
    ) -> tuple[pd.DataFrame, Segment | None]:
        """Extract anomalous metrics from time series data and return information about the selected segment

        This method is a thin wrapper around sift() that returns the legacy
        (positional) :class:`Segment` for backward compatibility.

        Args:
            data: Input time series data
            without_simple_filter: If True, skip STEP0 simple filter

        Returns:
            tuple[pd.DataFrame, Segment | None]:
                - DataFrame of extracted metrics
                - Information about the selected segment (None if not found)
        """
        result = self.sift(data, without_simple_filter)
        selected_segment = None
        if result.selected_segment is not None:
            s = result.selected_segment
            selected_segment = Segment(label=s.label, start_time=s.start_index, end_time=s.end_index)
        return (result.data if result.data is not None else pd.DataFrame()), selected_segment

    def sift(self, data: pd.DataFrame, without_simple_filter: bool = False) -> SiftResult:
        """Run the pipeline and return a diagnostic, explainable :class:`SiftResult`.

        The result contains the filtered DataFrame, per-metric change points, the
        reason each metric was kept or dropped, and every candidate segment with
        its selection score. When the input has a ``DatetimeIndex``, change points
        and segments are additionally expressed as wall-clock timestamps.

        Args:
            data: Input time series data
            without_simple_filter: If True, skip STEP0 simple filter

        Returns:
            SiftResult: Diagnostic result of the feature reduction pipeline

        Raises:
            ValueError: If ``data`` is not a DataFrame or its column names are
                not unique.
        """
        self._validate_input(data)
        input_metrics = list(data.columns)

        if without_simple_filter:
            X = data
        else:
            # STEP0: simple filter
            X = self._filter_no_changes(data, n_jobs=self.n_jobs)

        filtered_no_change = frozenset(input_metrics) - frozenset(X.columns)
        index = X.index
        has_datetime = isinstance(index, pd.DatetimeIndex)

        # STEP1: detect change points
        flatten_change_points, cp_to_metrics, metric_to_cps, penalty_tuning = self._detect_changepoints(X)

        metric_to_change_points = {metric: [int(cp) for cp in cps] for metric, cps in metric_to_cps.items()}
        filtered_no_change_points = frozenset(
            metric for metric, cps in metric_to_change_points.items() if len(cps) == 0
        )
        metric_to_change_times = None
        if has_datetime:
            metric_to_change_times = {
                metric: [index[cp] for cp in cps] for metric, cps in metric_to_change_points.items()
            }

        if not flatten_change_points:
            # Return a fully empty DataFrame (no index) to preserve the legacy
            # behavior of run() / run_with_selected_segment().
            bandwidth_tuning = None
            if self.bandwidth == AUTO:
                bandwidth_tuning = BandwidthTuning(
                    requested=self.bandwidth,
                    resolved=segmentation.BANDWIDTH_FALLBACK,
                    reason="no_change_points",
                )
            return SiftResult(
                data=pd.DataFrame(),
                selected_metrics=frozenset(),
                filtered_no_change=filtered_no_change,
                filtered_no_change_points=filtered_no_change_points,
                filtered_out_of_segment=frozenset(),
                metric_to_change_points=metric_to_change_points,
                metric_to_change_times=metric_to_change_times,
                segments=[],
                selected_segment=None,
                penalty_tuning=penalty_tuning,
                bandwidth_tuning=bandwidth_tuning,
            )

        # STEP2: segment change points (resolving bandwidth="auto" first)
        bandwidth, bandwidth_tuning = self._resolve_bandwidth(
            flatten_change_points, cp_to_metrics, metric_to_cps, time_series_length=X.shape[0]
        )
        cluster_label_to_metrics, label_to_change_points = segmentation.segment_nested_changepoints(
            flatten_change_points=flatten_change_points,
            cp_to_metrics=cp_to_metrics,
            time_series_length=X.shape[0],
            kde_bandwidth=bandwidth,
        )

        # STEP3: select the largest (densest) segment
        selected_label, remained_metrics = self.select_largest_segment_with_label(
            cluster_label_to_metrics, metric_to_cps, label_to_change_points
        )

        segments: list[SegmentInfo] = []
        selected_segment: SegmentInfo | None = None
        for label, change_points in sorted(label_to_change_points.items()):
            if len(change_points) == 0:
                continue
            metrics = frozenset(cluster_label_to_metrics.get(label, set()))
            start_index = int(min(change_points))
            end_index = int(max(change_points))
            segment = SegmentInfo(
                label=int(label),
                metrics=metrics,
                start_index=start_index,
                end_index=end_index,
                start_time=index[start_index] if has_datetime else None,
                end_time=index[end_index] if has_datetime else None,
                score=self._score_candidate(
                    self._build_candidate(label, metrics, metric_to_cps, label_to_change_points)
                ),
                selected=(label == selected_label),
            )
            segments.append(segment)
            if segment.selected:
                selected_segment = segment

        has_change_points = frozenset(metric for metric, cps in metric_to_change_points.items() if len(cps) > 0)
        filtered_out_of_segment = has_change_points - frozenset(remained_metrics)

        selected_columns = [c for c in X.columns if c in remained_metrics]
        return SiftResult(
            data=X[selected_columns],
            selected_metrics=frozenset(remained_metrics),
            filtered_no_change=filtered_no_change,
            filtered_no_change_points=filtered_no_change_points,
            filtered_out_of_segment=filtered_out_of_segment,
            metric_to_change_points=metric_to_change_points,
            metric_to_change_times=metric_to_change_times,
            segments=segments,
            selected_segment=selected_segment,
            penalty_tuning=penalty_tuning,
            bandwidth_tuning=bandwidth_tuning,
        )

    @staticmethod
    def _build_candidate(
        label: int,
        metrics,
        metric_to_cps: dict[str, list[int]],
        label_to_change_points: dict | None = None,
    ) -> SegmentCandidate:
        """Assemble the :class:`SegmentCandidate` view handed to a scoring strategy."""
        metrics_fs = frozenset(metrics)
        raw_cps = (label_to_change_points or {}).get(label, [])
        change_points = sorted(int(cp) for cp in raw_cps)
        # Restrict the change-point map to this segment's metrics (all of which are
        # guaranteed to be present in metric_to_cps, since they originate from it).
        sub_metric_to_cps = {m: list(metric_to_cps[m]) for m in metrics_fs if m in metric_to_cps}
        return SegmentCandidate(
            label=int(label),
            metrics=metrics_fs,
            change_points=change_points,
            metric_to_cps=sub_metric_to_cps,
        )

    def _score_candidate(self, candidate: SegmentCandidate) -> float:
        """Score a candidate segment under the configured selection method.

        Mirrors select_largest_segment_with_label() so that the selected segment
        holds the maximum score among all candidates. Supports the built-in string
        methods and any custom ``Callable[[SegmentCandidate], float]``.
        """
        method = self.segment_selection_method
        if callable(method):
            return float(method(candidate))
        match method:
            case "max" | "":
                return float(len(candidate.metrics))
            case "weighted_max":
                return float(sum(1 / len(candidate.metric_to_cps[m]) for m in candidate.metrics))
            case _:
                raise ValueError(f"Unknown segment_selection_method: {method!r}")

    def select_largest_segment(
        self,
        cluster_label_to_metrics: dict,
        metric_to_cps: dict[str, list[int]],
    ) -> set[str]:
        """Select the largest segment and return its metrics

        This method is a wrapper around select_largest_segment_with_label() that only
        returns the metrics for backward compatibility.

        Args:
            cluster_label_to_metrics: Mapping from segment ID to metrics set
            metric_to_cps: Mapping from metric name to change points list

        Returns:
            set[str]: Set of metrics in the selected segment
        """
        _, remained_metrics = self.select_largest_segment_with_label(cluster_label_to_metrics, metric_to_cps)
        return remained_metrics

    def select_largest_segment_with_label(
        self,
        cluster_label_to_metrics: dict,
        metric_to_cps: dict[str, list[int]],
        label_to_change_points: dict | None = None,
    ) -> tuple[int | None, set[str]]:
        """Select the largest segment and return its label and metrics

        Args:
            cluster_label_to_metrics: Mapping from segment ID to metrics set
            metric_to_cps: Mapping from metric name to change points list
            label_to_change_points: Mapping from segment ID to its change points.
                Only needed by a custom ``Callable`` selection strategy so it can
                read ``SegmentCandidate.change_points``; ignored by the built-in
                string strategies and optional for backward compatibility.

        Returns:
            tuple[int | None, set[str]]:
                - Label of the selected segment (None if no segments exist)
                - Set of metrics in the selected segment
        """
        if not cluster_label_to_metrics:
            return None, set()

        method = self.segment_selection_method
        if callable(method):
            choiced_cluster = max(
                cluster_label_to_metrics.items(),
                key=lambda item: self._score_candidate(
                    self._build_candidate(item[0], item[1], metric_to_cps, label_to_change_points)
                ),
            )
        else:
            match method:
                case "max" | "":
                    choiced_cluster = max(cluster_label_to_metrics.items(), key=lambda x: len(x[1]))
                case "weighted_max":
                    if metric_to_cps is None:
                        raise ValueError("metric_to_cps should not be None")
                    choiced_cluster = max(
                        cluster_label_to_metrics.items(), key=lambda x: sum(1 / len(metric_to_cps[m]) for m in x[1])
                    )
                case _:
                    raise ValueError(f"Unknown segment_selection_method: {method!r}")

        selected_label: int = choiced_cluster[0]
        remained_metrics: set[str] = set(choiced_cluster[1])
        return selected_label, remained_metrics
