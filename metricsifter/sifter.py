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
        penalty: str = "bic",
        penalty_adjust: float = 2.,
        bandwidth: float = 2.5,
        segment_selection_method: str = "weighted_max",
        n_jobs: int = -1,
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

    def run(self, time_series: pd.DataFrame) -> pd.DataFrame:
        # STEP0: simple filter
        X: pd.DataFrame = self._filter_no_changes(time_series, n_jobs=self.n_jobs)

        metrics: list[str] = X.columns.tolist()

        # STEP1: detect change points
        change_point_indexes = detection.detect_multi_changepoints(
            X,
            search_method=self.search_method,
            cost_model=self.cost_model,
            penalty=self.penalty,
            penalty_adjust=self.penalty_adjust,
            n_jobs=self.n_jobs,
        )
        metric_to_cps = {metric: cps for metric, cps in zip(metrics, change_point_indexes)}

        # STEP2: segment change points
        cluster_label_to_metrics, _ = segmentation.segment_nested_changepoints(
            multi_change_points=change_point_indexes,
            metrics=metrics,
            time_series_length=X.shape[0],
            kde_bandwidth=self.bandwidth,
        )

        # STEP3: select the largest segment
        remained_metrics = self.select_largest_segment(cluster_label_to_metrics, metrics, metric_to_cps)

        return X.loc[:, list(remained_metrics)]


    def select_largest_segment(
        self,
        cluster_label_to_metrics: dict,
        metrics: list[str],
        metric_to_cps: dict[str, list[int]],
    ) -> set[str]:
        if not cluster_label_to_metrics:
            return set()
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
        remained_metrics: set[str] = set(choiced_cluster[1])
        return remained_metrics
