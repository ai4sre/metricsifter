from typing import Final, Literal

import pandas as pd
from joblib import Parallel, delayed
from metricsifter import sifter

from dataset.separater import separate_data_by_component
from priorknowledge.base import PriorKnowledge

from .algo.birch import Birch
from .algo.fluxinfer_ad import FluxInferAD
from .algo.hdbs import HDBS
from .algo.kstest import KSTest
from .algo.nsigma import NSigma

LOOKBACK_WINDOW_SIZE: Final[int] = 4 * 20  # 20min
REDUCTION_METHODS: Final[list[str]] = [
    "None",
    "Ideal",
    "MetricSifter upto CPD",
    "MetricSifter",
    "NSigma",
    "Birch",
    "K-S test",
    "FluxInfer-AD",
    "HDBS-SBD",
    "HDBS-R",
]
EMPIRICAL_REDUCTION_METHODS: Final[list[str]] = [
    "None",
    "MetricSifter",
    "NSigma",
    "Birch",
    "K-S test",
    "FluxInfer-AD",
    "HDBS-SBD",
    "HDBS-R",
]

def reduce_by_method(
    method: str,
    data: pd.DataFrame,
    ground_truth: dict,
    enable_preprocessing: bool = False,
) -> set[str]:
    if enable_preprocessing:
        data = sifter.Sifter._filter_no_changes(data, n_jobs=1)

    remained_metrics: set
    match method:
        case "None":
            remained_metrics = set(data.columns.tolist())
        case "Ideal":
            remained_metrics = set(ground_truth["root_fault_nodes"] + ground_truth["fault_propagated_nodes"])
        case "MetricSifter upto CPD":
            reducer = sifter.Sifter(
                search_method="pelt", cost_model="l2", penalty="bic", penalty_adjust=2.5, n_jobs=1,
            )
            ret = reducer.run_upto_cpd(data, without_simple_filter=True)
            remained_metrics = set(ret.columns.tolist())
        case "MetricSifter":
            reducer = sifter.Sifter(
                search_method="pelt", cost_model="l2", penalty="bic",
                penalty_adjust=2.5, bandwidth=3.5,
                segment_selection_method="weighted_max",
                n_jobs=1,
            )
            ret = reducer.run(data, without_simple_filter=True)
            remained_metrics = set(ret.columns.tolist())
        case "NSigma":
            nsigma = NSigma(data, n_sigmas=3, nsigma_anomalous_start_idx=LOOKBACK_WINDOW_SIZE)
            ret = nsigma.run()
            remained_metrics = set(ret.columns.tolist())
        case "Birch":
            birch = Birch(data, birch_anomalous_start_idx=-LOOKBACK_WINDOW_SIZE)
            ret = birch.run()
            remained_metrics = set(ret.columns.tolist())
        case "K-S test":
            kstest = KSTest(data, kstest_anomalous_start_idx=-LOOKBACK_WINDOW_SIZE, kstest_alpha=0.05)
            ret = kstest.run()
            remained_metrics = set(ret.columns.tolist())
        case "FluxInfer-AD":
            fluxinfer = FluxInferAD(data, n_sigmas=3)
            ret = fluxinfer.run()
            remained_metrics = set(ret.columns.tolist())
        case "HDBS-SBD":
            hdbs = HDBS(data, dist_type="sbd")
            ret = hdbs.run()
            remained_metrics = set(ret.columns.tolist())
        case "HDBS-R":
            hdbs = HDBS(data, dist_type="pearson")
            ret = hdbs.run()
            remained_metrics = set(ret.columns.tolist())
        case _:
            raise ValueError(f"Unknown reduction method: {method}")

    return remained_metrics


def run_by_component_for_empirical(
    data: pd.DataFrame,
    ground_truth: dict,
    pk: PriorKnowledge,
    method: Literal["HDBS-R", "HDBS-SBD"],
    granularity: Literal["service", "container"] = "service",
    n_jobs: int = 1,
) -> set[str]:
    """ Run reduction methods by component unit for empirical data """

    comp_to_metrics_df = separate_data_by_component(data, pk, granularity=granularity)
    ret = Parallel(n_jobs=n_jobs)(delayed(reduce_by_method)(method, _data, ground_truth) for _data in comp_to_metrics_df.values())
    assert ret is not None, "Parallel execution failed"
    return set.union(*ret)
