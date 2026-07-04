from typing import Callable

import numpy as np
import pandas as pd

from metricsifter import utils
from metricsifter.algo import detection, segmentation
from metricsifter.types import Segment, SegmentInfo, SiftResult


class Sifter:
    def __init__(
        self,
        search_method: str = "pelt",
        cost_model: str = "l2",
        penalty: str | float = "bic",
        penalty_adjust: float = 2.0,
        bandwidth: float = 2.5,
        segment_selection_method: str = "weighted_max",
        n_jobs: int = 1,
    ) -> None:
        self.search_method = search_method
        self.cost_model = cost_model
        self.bandwidth = bandwidth
        self.penalty = penalty
        self.penalty_adjust = penalty_adjust
        self.segment_selection_method = segment_selection_method
        self.n_jobs = n_jobs

    @staticmethod
    def _filter_no_changes(X: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
        vf: Callable = np.vectorize(lambda x: np.isnan(x) or x == 0)

        def filter(x: pd.Series) -> bool:
            # pd.Series.diff returns a series with the first element is NaN
            if x.isna().all() or (x == x.iat[0]).all() or ((diff_x := np.diff(x)) == diff_x[0]).all():
                return False
            # remove an array including only the same value or nan
            return not vf(diff_x).all()

        if n_jobs != 1:
            return X.loc[:, utils.parallel_apply(X, filter, n_jobs)]
        return X.loc[:, X.apply(filter)]

    def run_upto_cpd(self, data: pd.DataFrame, without_simple_filter: bool = False) -> pd.DataFrame:
        """Run up to change point detection"""
        if without_simple_filter:
            X = data
        else:
            # STEP0: simple filter
            X = self._filter_no_changes(data, n_jobs=self.n_jobs)

        # STEP1: detect change points
        _, _, metric_to_cps = detection.detect_multi_changepoints(
            X,
            search_method=self.search_method,
            cost_model=self.cost_model,
            penalty=self.penalty,
            penalty_adjust=self.penalty_adjust,
            n_jobs=self.n_jobs,
        )
        remained_metrics = set(metric for metric, cps in metric_to_cps.items() if len(cps) > 0)
        return X.loc[:, list(remained_metrics)]

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
        """
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
        flatten_change_points, cp_to_metrics, metric_to_cps = detection.detect_multi_changepoints(
            X,
            search_method=self.search_method,
            cost_model=self.cost_model,
            penalty=self.penalty,
            penalty_adjust=self.penalty_adjust,
            n_jobs=self.n_jobs,
        )

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
            )

        # STEP2: segment change points
        cluster_label_to_metrics, label_to_change_points = segmentation.segment_nested_changepoints(
            flatten_change_points=flatten_change_points,
            cp_to_metrics=cp_to_metrics,
            time_series_length=X.shape[0],
            kde_bandwidth=self.bandwidth,
        )

        # STEP3: select the largest (densest) segment
        selected_label, remained_metrics = self.select_largest_segment_with_label(
            cluster_label_to_metrics, metric_to_cps
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
                score=self._segment_score(metrics, metric_to_cps),
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
        )

    def _segment_score(self, metrics: frozenset[str], metric_to_cps: dict[str, list[int]]) -> float:
        """Compute the selection score of a segment under the configured method.

        Mirrors select_largest_segment_with_label() so that the selected segment
        holds the maximum score among all candidates.
        """
        match self.segment_selection_method:
            case "max" | "":
                return float(len(metrics))
            case "weighted_max":
                return float(sum(1 / len(metric_to_cps[m]) for m in metrics))
            case _:
                raise ValueError(f"Unknown segment_selection_method: {self.segment_selection_method}")

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
    ) -> tuple[int | None, set[str]]:
        """Select the largest segment and return its label and metrics

        Args:
            cluster_label_to_metrics: Mapping from segment ID to metrics set
            metric_to_cps: Mapping from metric name to change points list

        Returns:
            tuple[int | None, set[str]]:
                - Label of the selected segment (None if no segments exist)
                - Set of metrics in the selected segment
        """
        if not cluster_label_to_metrics:
            return None, set()

        match self.segment_selection_method:
            case "max" | "":
                choiced_cluster = max(cluster_label_to_metrics.items(), key=lambda x: len(x[1]))
            case "weighted_max":
                if metric_to_cps is None:
                    raise ValueError("metric_to_cps should not be None")
                choiced_cluster = max(
                    cluster_label_to_metrics.items(), key=lambda x: sum(1 / len(metric_to_cps[m]) for m in x[1])
                )
            case _:
                raise ValueError(f"Unknown segment_selection_method: {self.segment_selection_method}")

        selected_label: int = choiced_cluster[0]
        remained_metrics: set[str] = set(choiced_cluster[1])
        return selected_label, remained_metrics
