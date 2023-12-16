import time
from itertools import product

import pandas as pd
from joblib import Parallel, delayed

from evaluation.empirical_ground_truth import select_root_fault_metrics
from evaluation.reduction import scores_of_empirical, scores_of_synthetic
from priorknowledge.base import PriorKnowledge
from priorknowledge.sockshop import SockShopKnowledge
from priorknowledge.trainticket import TrainTicketKnowledge
from reduction.runner import (
    EMPIRICAL_REDUCTION_METHODS,
    REDUCTION_METHODS,
    reduce_by_method,
)

MEDIUM_TARGET_METRIC_TYPES = {
    "containers": True,
    "services": True,
    "nodes": False,
    "middlewares": False,
}

LARGE_TARGET_METRIC_TYPES = {
    "containers": True,
    "services": True,
    "nodes": False,
    "middlewares": False,
}


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


def _run_reduction_on_empirical(method: str, data_param: dict, data_body: dict) -> dict:
    data: pd.DataFrame = data_body["data"]
    data_spec: dict[str, str | int] = data_body["data_spec"]

    pk: PriorKnowledge
    match data_spec["dataset_name"].split("-"):
        case ["SS", "small"]:
            pk = SockShopKnowledge(
                target_metric_types=MEDIUM_TARGET_METRIC_TYPES
            )
        case ["SS", "medium"]:
            pk = SockShopKnowledge(
                target_metric_types=MEDIUM_TARGET_METRIC_TYPES
            )
        case ["SS", "large"]:
            pk = SockShopKnowledge(
                target_metric_types=LARGE_TARGET_METRIC_TYPES
            )
        case ["TT", "small"]:
            pk = TrainTicketKnowledge(
                target_metric_types=MEDIUM_TARGET_METRIC_TYPES
            )
        case ["TT", "medium"]:
            pk = TrainTicketKnowledge(
                target_metric_types=MEDIUM_TARGET_METRIC_TYPES
            )
        case ["TT", "large"]:
            pk = TrainTicketKnowledge(
                target_metric_types=LARGE_TARGET_METRIC_TYPES
            )
        case _:
            raise ValueError(f"Unknown dataset: {data_spec['dataset_name']}")

    selected_root_fault_metrics = select_root_fault_metrics(
        pk=pk,
        metrics=set(data.columns.tolist()),
        fault_type=data_spec["fault_type"],
        fault_comp=data_spec["fault_comp"],
    )

    time_start: float = time.perf_counter()

    remained_metrics = reduce_by_method(
        method, data,
        {
            "root_fault_nodes": selected_root_fault_metrics,
            "fault_propagated_nodes": [],  # not available labels in empirical datasets
        },
    )

    time_reduction: float = time.perf_counter() - time_start
    scores = scores_of_empirical(
        pred_anomalous_metrics=remained_metrics,
        true_root_fault_metrics=set(selected_root_fault_metrics),
        total_metrics=set(data.columns.tolist()),
    )
    scores["time_reduction"] = time_reduction

    return data_param | {"reduction_method": method} | scores


def sweep_reduction_on_empirical(dataset: list[tuple[dict, dict]], n_jobs: int = -1) -> pd.DataFrame:
    ret = Parallel(n_jobs=n_jobs)(
        delayed(_run_reduction_on_empirical)(method, data_param, data_body)
        for method, (data_param, data_body) in product(EMPIRICAL_REDUCTION_METHODS, dataset)
    )
    assert ret is not None
    return pd.DataFrame(ret)
