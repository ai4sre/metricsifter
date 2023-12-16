from typing import Literal

import pandas as pd

from priorknowledge.base import PriorKnowledge


def separate_data_by_component(
    data: pd.DataFrame,
    pk: PriorKnowledge,
    granularity: Literal["service", "container"] = "service",
) -> dict[str, pd.DataFrame]:
    comp_to_metrics_df: dict[str, pd.DataFrame] = {}

    match granularity:
        case "service":
            for service, containers in pk.get_containers_of_service().items():
                metrics_dfs: list[pd.DataFrame] = []

                service_metrics_df = data.loc[
                    :, data.columns.str.startswith(f"s-{service}_")
                ]
                if len(service_metrics_df.columns) > 0:
                    metrics_dfs.append(service_metrics_df)

                for container in containers:
                    container_metrics_df = data.loc[
                        :,
                        data.columns.str.startswith(
                            (f"c-{container}_", f"m-{container}_")
                        ),
                    ]
                    if len(container_metrics_df.columns) > 0:
                        metrics_dfs.append(container_metrics_df)

                if len(metrics_dfs) > 0:
                    comp_to_metrics_df[service] = pd.concat(metrics_dfs, axis=1)

        case "container":
            # Clustering metrics by service including services, containers and middlewares metrics
            for service, containers in pk.get_containers_of_service().items():
                # 1. service-level clustering
                service_metrics_df = data.loc[
                    :, data.columns.str.startswith(f"s-{service}_")
                ]
                if len(service_metrics_df.columns) > 1:
                    comp_to_metrics_df[f"s-{service}"] = service_metrics_df

                # 2. container-level clustering
                for container in containers:
                    # perform clustering in each type of metric
                    container_metrics_df = data.loc[
                        :,
                        data.columns.str.startswith(
                            (f"c-{container}_", f"m-{container}_")
                        ),
                    ]
                    if len(container_metrics_df.columns) <= 1:
                        continue
                    comp_to_metrics_df[f"c-{container}"] = container_metrics_df
        case _:
            assert False, f"Invalid granularity: {granularity}"

    # 3. node-level clustering
    for node in pk.get_nodes():
        node_metrics_df = data.loc[:, data.columns.str.startswith(f"n-{node}_")]
        if len(node_metrics_df.columns) <= 1:
            continue
        comp_to_metrics_df[f"n-{node}"] = node_metrics_df

    return comp_to_metrics_df
