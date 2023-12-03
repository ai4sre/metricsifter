from typing import Final

import pandas as pd
from metricsifter import sifter

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

def reduce_by_method(
    method: str,
    data: pd.DataFrame,
    ground_truth: dict,
) -> set[str]:
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
                penalty_adjust=2.5, bandwidth=2.5,
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
