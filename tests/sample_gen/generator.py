import networkx as nx
import numpy as np
import pandas as pd
from pyrca.simulation import data_gen


def generate_synthetic_data(
    num_node: int,
    num_edge: int,
    num_normal_samples: int,
    num_abnormal_samples: int,
    anomaly_type: int,
    func_type: str = 'identity',
    noise_type: str = 'uniform',
    weight_generator: str = 'uniform',
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame, set[str]] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    graph_matrix = data_gen.DAGGen(
        data_gen.DAGGenConfig(num_node=num_node, num_edge=num_edge)
    ).gen()
    G = nx.DiGraph(graph_matrix)

    # transform node names from 0 to N-1 to X1 to XN
    no_of_var = graph_matrix.shape[0]
    original_names = [i for i in range(no_of_var)]
    node_names = [("X%d" % (i + 1)) for i in range(no_of_var)]
    mapping = dict(zip(original_names, node_names))
    G = nx.relabel_nodes(G, mapping)
    adjacency_df = pd.DataFrame({node_names[i]: graph_matrix[:, i] for i in range(len(node_names))}, index=node_names)

    normal_data, parent_weights, noise_weights, func_form, noise_form = data_gen.DataGen(
        data_gen.DataGenConfig(
            dag=graph_matrix,
            func_type=func_type,
            noise_type=noise_type,
            weight_generator=weight_generator,
            num_samples=num_normal_samples,
        )
    ).gen()

    # --- Abnormal data ---
    _SLI = 0
    tau = 3
    baseline = normal_data[:, _SLI].mean()
    sli_sigma = normal_data[:, _SLI].std()
    threshold = tau * sli_sigma
    anomaly_data, fault = data_gen.AnomalyDataGen(
        data_gen.AnomalyDataGenConfig(
            parent_weights=parent_weights,
            noise_weights=noise_weights,
            func_type=func_form,
            noise_type=noise_form,
            threshold=threshold,
            baseline=baseline,
            anomaly_type=anomaly_type,
            num_samples=num_abnormal_samples,
        )
    ).gen()

    true_root_causes = [mapping[i] for i in np.where(fault != 0)[0]]
    anomaly_propagated_nodes = set()
    for root_cause in true_root_causes:
        paths = nx.all_simple_paths(G, source=root_cause, target=["X1"])
        for path in paths:
            for node in path:
                anomaly_propagated_nodes.add(node)

    normal_data_df = pd.DataFrame(normal_data, columns=node_names)
    abnormal_data_df = pd.DataFrame(anomaly_data, columns=node_names)

    return normal_data_df, abnormal_data_df, true_root_causes, adjacency_df, anomaly_propagated_nodes
