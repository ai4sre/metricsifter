import warnings
from typing import Final

import networkx as nx
import pandas as pd
from internal_logger.logger import logger
from priorknowledge.base import PriorKnowledge
from priorknowledge.call_graph import get_forbits, prepare_init_graph
from pyrca.analyzers.epsilon_diagnosis import EpsilonDiagnosis, EpsilonDiagnosisConfig
from pyrca.analyzers.ht import HT, HTConfig
from pyrca.graphs.causal.lingam import LiNGAM, LiNGAMConfig
from pyrca.graphs.causal.pc import PC, PCConfig
from threadpoolctl import threadpool_limits

from .rcd import run_rcd

num_trials: int = 5
top_k: int = 10
LOCALIZATUON_METHODS: Final[list[str]] = [
    "RCD",
    "EpsilonDiagnosis",
    "PC+HT",
    "LiNGAM+HT",
    "PC+PageRank",
    "LiNGAM+PageRank",
]

def method_to_method_pair(method: str) -> tuple[str, str]:
    """ Convert method name to method pair (buiding step, scoring step)."""
    match method:
        case "RCD":
            return "RCD", ""
        case "EpsilonDiagnosis":
            return "e-Diagnosis", ""
        case "PC+HT":
            return "PC", "HT"
        case "LiNGAM+HT":
            return "LiNGAM", "HT"
        case "PC+PageRank":
            return "PC", "PageRank"
        case "LiNGAM+PageRank":
            return "LiNGAM", "PageRank"
        case _:
            raise ValueError(f"Unknown localization method: {method}")


def run_rca(
    pk: PriorKnowledge,
    data_df: pd.DataFrame,
    graph: pd.DataFrame,
    building_step: str,
    scoring_step: str,
    top_k: int,
    boundary_index: int,
    abnormal_metrics: list[str] = ["X1"],
    **kwargs: dict,
) -> list[tuple[str, float]]:
    if len(data_df.columns) == 0:
        return []
    normal_data_df = data_df[data_df.index < boundary_index]
    abnormal_data_df = data_df[data_df.index >= boundary_index]
    use_call_graph = kwargs.get("use_call_graph", True)
    match building_step:
        case "e-Diagnosis":
            normal_df = normal_data_df.iloc[-len(abnormal_data_df):, :]
            model = EpsilonDiagnosis(config=EpsilonDiagnosisConfig(root_cause_top_k=top_k))
            with threadpool_limits(limits=1):
                model.train(normal_df)
                results = model.find_root_causes(abnormal_data_df).to_list()
            return [(r["root_cause"], r["score"]) for r in results]
        case "RCD":
            return run_rcd(data_df, boundary_index, top_k, kwargs.get("rcd_n_iters", 1))
        case "PC":
            forbits = get_forbits(set(data_df.columns.tolist()), pk) if use_call_graph else []
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                with threadpool_limits(limits=1):
                    graph = PC(PCConfig(run_pdag2dag=True)).train(
                        pd.concat([normal_data_df, abnormal_data_df], axis=0, ignore_index=True),
                        forbits=forbits,
                    )
        case "LiNGAM":
            forbits = get_forbits(set(data_df.columns.tolist()), pk) if use_call_graph else []
            with warnings.catch_warnings(action='ignore', category=UserWarning):
                with threadpool_limits(limits=1):
                    graph = LiNGAM(LiNGAMConfig(run_pdag2dag=True)).train(
                        pd.concat([normal_data_df, abnormal_data_df], axis=0, ignore_index=True),
                        forbits=forbits,
                    )
        case "CG":
            nx_g = prepare_init_graph(
                set(data_df.columns.tolist()), pk, enable_orientation=True,
            )
            graph = nx.to_pandas_adjacency(nx_g)
        case _:
            raise ValueError(f"Model {method} is not supported.")

    abnormal_metrics_for_walk: list[str] | None = []
    for abnormal_metric in abnormal_metrics:
        if abnormal_metric in graph.columns:
            abnormal_metrics_for_walk.append(abnormal_metric)
    if not abnormal_metrics_for_walk:
        abnormal_metrics_for_walk = None

    match scoring_step:
        case "HT":
            model = HT(config=HTConfig(graph=graph, root_cause_top_k=top_k))
            with threadpool_limits(limits=1):
                model.train(normal_data_df)
                try:
                    results = model.find_root_causes(
                        abnormal_data_df,
                        abnormal_metrics_for_walk[0] if abnormal_metrics_for_walk is not None else None,
                        adjustment=True,
                    ).to_list()
                except nx.exception.NetworkXUnfeasible:
                    logger.warning("skip to run 'ht' because the graph has a cycle")
                    return []
        case "PageRank":
            with threadpool_limits(limits=1):
                rank = nx.pagerank(nx.DiGraph(graph).reverse())
                results = [{"root_cause": k, "score": v} for k, v in sorted(rank.items(), key=lambda item: item[1], reverse=True)][:top_k]
        case _:
            raise ValueError(f"Unknown walk method: {scoring_step}")

    return [(r["root_cause"], r["score"]) for r in results]


def get_rank_items_with_rca(
    pk: PriorKnowledge,
    method: str, top_k: int, data_df: pd.DataFrame,
    true_root_fault_metrics: set[str], graph: pd.DataFrame,
    boundary_index: int,
    sli_metrics: set[str],
    **kwargs: dict,
) -> list[dict[str, int | float | bool]]:
    building_step, scoring_step = method_to_method_pair(method)

    ranks = run_rca(
        pk, data_df, graph,
        building_step=building_step, scoring_step=scoring_step,
        top_k=top_k, boundary_index=boundary_index,
        abnormal_metrics=list(sli_metrics),
        **kwargs,
    )

    items: list = []
    for k, (metric, score) in enumerate(ranks, 1):
        hit = metric in true_root_fault_metrics
        items.append(
            dict(
                {
                    "k": k,"metric": metric, "score": score, "hit": hit,
                },
            )
        )
    return items
