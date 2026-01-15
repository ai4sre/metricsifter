from typing import Callable

import numpy as np
import pandas as pd

from metricsifter import utils
from metricsifter.algo import detection, segmentation


class Sifter:
    def __init__(
        self,
        search_method: str = "pelt",
        cost_model: str = "l2",
        penalty: str | float = "bic",
        penalty_adjust: float = 2.,
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
        """ Run up to change point detection"""
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

        This method is a wrapper around run_with_selected_segment() that only returns
        the filtered DataFrame for backward compatibility.

        Args:
            data: Input time series data
            without_simple_filter: If True, skip STEP0 simple filter

        Returns:
            pd.DataFrame: DataFrame containing only the selected metrics
        """
        filtered_data, _ = self.run_with_selected_segment(data, without_simple_filter)
        return filtered_data

    def run_with_selected_segment(
        self,
        data: pd.DataFrame,
        without_simple_filter: bool = False
    ) -> tuple[pd.DataFrame, "Segment | None"]:
        """Extract anomalous metrics from time series data and return information about the selected segment

        Args:
            data: Input time series data
            without_simple_filter: If True, skip STEP0 simple filter

        Returns:
            tuple[pd.DataFrame, Segment | None]:
                - DataFrame of extracted metrics
                - Information about the selected segment (None if not found)
        """
        if without_simple_filter:
            X = data
        else:
            # STEP0: simple filter
            X = self._filter_no_changes(data, n_jobs=self.n_jobs)

        # STEP1: detect change points
        flatten_change_points, cp_to_metrics, metric_to_cps = detection.detect_multi_changepoints(
            X,
            search_method=self.search_method,
            cost_model=self.cost_model,
            penalty=self.penalty,
            penalty_adjust=self.penalty_adjust,
            n_jobs=self.n_jobs,
        )
        if not flatten_change_points:
            return pd.DataFrame(), None

        # STEP2: segment change points
        cluster_label_to_metrics, label_to_change_points = segmentation.segment_nested_changepoints(
            flatten_change_points=flatten_change_points,
            cp_to_metrics=cp_to_metrics,
            time_series_length=X.shape[0],
            kde_bandwidth=self.bandwidth,
        )

        # STEP3: select the largest segment
        selected_label, remained_metrics = self.select_largest_segment_with_label(cluster_label_to_metrics, metric_to_cps)

        # Build information about the selected segment
        selected_segment = None
        if selected_label is not None and selected_label in label_to_change_points:
            change_points = label_to_change_points[selected_label]
            if len(change_points) > 0:
                from metricsifter.types import Segment
                selected_segment = Segment(
                    label=selected_label,
                    start_time=int(change_points.min()),
                    end_time=int(change_points.max())
                )

        return X.loc[:, list(remained_metrics)], selected_segment

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
                assert metric_to_cps is not None, "metric_to_cps should not be None"
                choiced_cluster = max(
                    cluster_label_to_metrics.items(), key=lambda x: sum(1 / len(metric_to_cps[m]) for m in x[1])
                )
            case _:
                raise ValueError(f"Unknown segment_selection_method: {self.segment_selection_method}")

        selected_label: int = choiced_cluster[0]
        remained_metrics: set[str] = set(choiced_cluster[1])
        return selected_label, remained_metrics
