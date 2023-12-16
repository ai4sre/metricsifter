from abc import ABC, abstractmethod
from collections import defaultdict
from functools import cache

import networkx as nx


class PriorKnowledge(ABC):
    def __init__(self, target_metric_types: dict[str, bool]) -> None:
        self.target_metric_types = target_metric_types

    @abstractmethod
    def get_root_service(self) -> str:
        pass

    @abstractmethod
    def get_root_container(self) -> str:
        pass

    @abstractmethod
    def get_services(self) -> list[str]:
        pass

    @abstractmethod
    def get_containers(self, skip: bool = False) -> list[str]:
        pass

    @abstractmethod
    def get_root_metrics(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def get_root_metric_by_type(self, red_type: str) -> str:
        pass

    @abstractmethod
    def get_service_call_digraph(self) -> nx.DiGraph:
        pass

    @abstractmethod
    def get_container_call_digraph(self) -> nx.DiGraph:
        pass

    @abstractmethod
    def get_container_call_graph(self, ctnr: str) -> list[str]:
        pass

    def get_container_neighbors_in_service(self, ctnr: str) -> list[str]:
        neighbors: list[str] = []
        service: str | None = self.get_service_by_container(ctnr)
        for neighbor in self.get_container_call_graph(ctnr):
            if service == self.get_service_by_container(neighbor):
                neighbors.append(neighbor)
        return neighbors

    @abstractmethod
    def get_service_routes(self, service: str) -> list[tuple[str, ...]]:
        pass

    @abstractmethod
    def get_containers_of_service(self) -> dict[str, list[str]]:
        pass

    @abstractmethod
    def get_service_containers(self, service: str) -> list[str]:
        pass

    @abstractmethod
    def get_service_by_container(self, ctnr: str) -> str:
        pass

    @abstractmethod
    def get_service_by_container_or_empty(self, ctnr: str) -> str:
        pass

    @abstractmethod
    def get_role_and_runtime_by_container(self, ctnr: str) -> tuple[str, str]:
        pass

    @abstractmethod
    def get_skip_containers(self) -> list[str]:
        pass

    @abstractmethod
    def get_skip_services(self) -> list[str]:
        pass

    @abstractmethod
    def get_diagnoser_target_data(self) -> dict[str, list[str]]:
        pass

    def group_metrics_by_service(self, metrics: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(lambda: list())
        for metric in metrics:
            # TODO: resolve duplicated code of MetricNode class.
            if (service := self.get_service_by_metric(metric)) is None:
                continue
            groups[service].append(metric)
        return groups

    def get_service_by_metric(self, metric: str) -> str | None:
        comp, _ = metric.split("-", maxsplit=1)[1].split("_", maxsplit=1)
        service: str | None
        if metric.startswith("c-"):
            service = self.get_service_by_container(comp)
        elif metric.startswith("s-"):
            service = comp
        elif metric.startswith("m-"):
            service = self.get_service_by_container(comp)
        elif metric.startswith("n-"):
            # 'node' doesn't belong to any service.
            service = None
        else:
            raise ValueError(f"{metric} is invalid")
        return service

    def get_container_by_metric(self, metric: str) -> str | None:
        comp, _ = metric.split("-", maxsplit=1)[1].split("_", maxsplit=1)
        container: str | None
        if metric.startswith("c-"):
            container = comp
        elif metric.startswith("s-"):
            container = None
        elif metric.startswith("m-"):
            container = comp
        elif metric.startswith("n-"):
            container = None
        else:
            raise ValueError(f"{metric} is invalid")
        return container

    def get_node_by_metric(self, metric: str) -> str | None:
        comp, _ = metric.split("-", maxsplit=1)[1].split("_", maxsplit=1)
        node: str | None
        if metric.startswith("c-"):
            node = None
        elif metric.startswith("s-"):
            node = None
        elif metric.startswith("m-"):
            node = None
        elif metric.startswith("n-"):
            node = comp
        else:
            raise ValueError(f"{metric} is invalid")
        return node

    def is_target_metric_type(self, metric_type: str) -> bool:
        assert metric_type in self.target_metric_types, f"{metric_type} is not defined in target_metric_types"
        return self.target_metric_types[metric_type]

    @staticmethod
    @cache
    def _generate_service_to_service_routes(
        service_call_g: nx.DiGraph,
        root_service: str,
    ) -> dict[str, list[tuple[str, ...]]]:
        """Generate adjacency list of service to service routes."""
        stos_routes: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        nodes = [n for n in service_call_g.nodes if n not in root_service]
        paths = nx.all_simple_paths(service_call_g, source=root_service, target=nodes)
        for path in paths:
            path.reverse()
            source_service = path[0]
            stos_routes[source_service].append(tuple(path[1:]))
        stos_routes[root_service] = [tuple([root_service])]
        return stos_routes
