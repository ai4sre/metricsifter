import time
from itertools import product

import pandas as pd
from joblib import Parallel, delayed

from evaluation.reduction import scores_of_synthetic
from reduction.runner import REDUCTION_METHODS, reduce_by_method


def _run_reduction(method: str, data_param: dict, data: dict) -> dict:
    normal_data = data["normal_data"]
    abnormal_data = data["abnormal_data"]
    ground_truth = data["ground_truth"]
    time_series = pd.concat([normal_data, abnormal_data], axis=0, ignore_index=True)

    time_start: float = time.perf_counter()

    remained_metrics = reduce_by_method(method, time_series, ground_truth)

    time_reduction: float = time.perf_counter() - time_start

    scores = scores_of_synthetic(
        pred_anomalous_metrics=remained_metrics,
        true_root_fault_metrics=set(ground_truth["root_fault_nodes"]),
        true_fault_propagated_metrics=set(ground_truth["fault_propagated_nodes"]),
        total_metrics=set(time_series.columns.tolist()),
    )
    scores["time_reduction"] = time_reduction

    return data_param | {"reduction_method": method} | scores


def sweep_reduction_on_synthetic(dataset: list[tuple[dict, dict]], n_jobs: int = -1) -> pd.DataFrame:
    ret = Parallel(n_jobs=n_jobs)(
        delayed(_run_reduction)(method, data_param, data)
        for method, (data_param, data) in product(REDUCTION_METHODS, dataset)
    )
    assert ret is not None
    return pd.DataFrame(ret)
