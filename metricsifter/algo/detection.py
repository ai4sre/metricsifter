import warnings
from collections import defaultdict
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd
import ruptures as rpt
from joblib import Parallel, delayed

NO_CHANGE_POINTS: Final[int] = -1


def _detect_changepoints_with_missing_values(x: np.ndarray) -> npt.ArrayLike:
    """
    Detect changepoints with missing values
    Sample:
        input: [1, 2, np.nan, np.nan, 5, 6, np.nan, 8, 9, np.nan, np.nan])
        output: [2 6 9]
    """
    is_nan = np.isnan(x)
    # Get the index where NaN value changes
    nans = np.where(is_nan, 1, 0)
    change_indexes = np.where(np.diff(nans) == 1)[0] + 1
    # If array starts with NaN, add 0 to indices
    if is_nan[0]:
        change_indexes = np.concatenate((np.array([0]), change_indexes))
    return change_indexes


def detect_univariate_changepoints(x: np.ndarray, search_method: str, cost_model: str, penalty: str | float, penalty_adjust: float) -> list[int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        match search_method:
            case "pelt":
                searcher = rpt.KernelCPD(kernel="linear", min_size=2, jump=1)  # written in C lang
            case "binseg":
                searcher = rpt.Binseg(model=cost_model, jump=1)
            case "bottomup":
                searcher = rpt.BottomUp(model=cost_model, jump=1)
            case _:
                assert False, f"search_method={search_method} is not supported."
    sigma = np.std(x)
    match penalty:
        case "aic":
            pen = sigma * sigma
        case "bic":
            pen = np.log(x.size) * sigma * sigma
        case _:
            pen = float(penalty)
    cps = searcher.fit(x).predict(pen=pen * penalty_adjust)
    assert cps is not None, "cps should not be None"
    cps = cps[:-1]  # remove the last index
    mvs = _detect_changepoints_with_missing_values(x)
    return sorted(list(set(cps) | set(mvs)))


def detect_multi_changepoints(
    X: pd.DataFrame,
    search_method: str,
    cost_model: str,
    penalty: str | float,
    penalty_adjust: float,
    n_jobs: int = -1,
) -> tuple[list[int], dict[int, list[str]], dict[str, list[int]]]:
    metrics: list[str] = X.columns.tolist()
    multi_change_points = Parallel(n_jobs=n_jobs)(
        delayed(detect_univariate_changepoints)(X[metric].to_numpy(), search_method, cost_model, penalty, penalty_adjust)
        for metric in metrics
    )
    cp_to_metrics: dict[int, list[str]] = defaultdict(list)
    for metric, change_points in zip(metrics, multi_change_points):
        if change_points is None or len(change_points) < 1:
            cp_to_metrics[NO_CHANGE_POINTS].append(metric)  # cp == -1 means no change point
            continue
        for cp in change_points:
            cp_to_metrics[cp].append(metric)

    flatten_change_points: list[int] = sum(multi_change_points, [])
    metric_to_cps = {metric: cps for metric, cps in zip(metrics, multi_change_points) if cps is not None}

    return flatten_change_points, cp_to_metrics, metric_to_cps
