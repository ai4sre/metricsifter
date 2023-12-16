from typing import Final

import networkx as nx

from .base import PriorKnowledge

TARGET_APP_NAME: Final[str] = "sock-shop"

ROOT_SERVICE: Final[str] = "front-end"
ROOT_CONTAINER: Final[str] = "front-end"
ROOT_METRIC_LABELS: Final[tuple[str, str, str]] = (
    "s-front-end_latency",
    "s-front-end_throughput",
    "s-front-end_errors",
)

ROOT_METRIC_TYPES_AS_RED: Final[dict[str, str]] = {
    "latency": "s-front-end_latency",
    "throughput": "s-front-end_throughput",
    "errors": "s-front-end_errors",
}

SERVICE_CALL_DIGRAPH: Final[nx.DiGraph] = nx.DiGraph(
    [
        ("front-end", "orders"),
        ("front-end", "catalogue"),
        ("front-end", "user"),
        ("front-end", "carts"),
        ("orders", "shipping"),
        ("orders", "payment"),
        ("orders", "user"),
        ("orders", "carts"),
    ]
)

CONTAINER_CALL_DIGRAPH: Final[nx.DiGraph] = nx.DiGraph(
    [
        ("front-end", "orders"),
        ("front-end", "carts"),
        ("front-end", "user"),
        ("front-end", "catalogue"),
        ("front-end", "session-db"),
        ("orders", "shipping"),
        ("orders", "payment"),
        ("orders", "user"),
        ("orders", "carts"),
        ("orders", "orders-db"),
        ("catalogue", "catalogue-db"),
        ("user", "user-db"),
        ("carts", "carts-db"),
        ("shipping", "rabbitmq"),
        ("rabbitmq", "queue-master"),
    ]
)

CONTAINER_CALL_GRAPH: Final[dict[str, list[str]]] = {
    "front-end": ["orders", "carts", "user", "catalogue"],
    "catalogue": ["front-end", "catalogue-db"],
    "catalogue-db": ["catalogue"],
    "orders": ["front-end", "orders-db", "carts", "user", "payment", "shipping"],
    "orders-db": ["orders"],
    "user": ["front-end", "user-db", "orders"],
    "user-db": ["user"],
    "payment": ["orders"],
    "shipping": ["orders", "rabbitmq"],
    "queue-master": ["rabbitmq"],
    "rabbitmq": ["shipping", "queue-master"],
    "carts": ["front-end", "carts-db", "orders"],
    "carts-db": ["carts"],
    "session-db": ["front-end"],
}

# Use list of tuple because of supporting multiple routes
SERVICE_TO_SERVICES: Final[dict[str, list[str]]] = {
    "orders": ["front-end"],
    "carts": ["orders", "front-end"],
    "user": ["orders", "front-end"],
    "catalogue": ["front-end"],
    "payment": ["orders"],
    "shipping": ["orders"],
    "front-end": [],
}

SERVICE_TO_SERVICE_ROUTES: Final[dict[str, list[tuple[str, ...]]]] = {
    "orders": [("front-end",)],
    "carts": [("orders", "front-end"), ("front-end",)],
    "user": [("orders", "front-end"), ("front-end",)],
    "catalogue": [("front-end",)],
    "payment": [("orders",)],
    "shipping": [("orders",)],
    "front-end": [()],
}

SERVICE_CONTAINERS: Final[dict[str, list[str]]] = {
    "carts": ["carts", "carts-db"],
    "payment": ["payment"],
    "shipping": ["shipping"],
    "front-end": ["front-end"],
    "user": ["user", "user-db"],
    "catalogue": ["catalogue", "catalogue-db"],
    "orders": ["orders", "orders-db"],
}

CONTAINER_TO_SERVICE: Final[dict[str, str]] = {c: s for s, ctnrs in SERVICE_CONTAINERS.items() for c in ctnrs}

CONTAINER_TO_RUNTIME: dict[str, tuple[str, str]] = {
    "carts": ("web", "jvm"),
    "carts-db": ("db", "mongodb"),
    "shipping": ("web", "jvm"),
    "payment": ("web", "jvm"),
    "front-end": ("web", "nodejs"),
    "user": ("web", "jvm"),
    "user-db": ("db", "mongodb"),
    "orders": ("web", "jvm"),
    "orders-db": ("db", "mongodb"),
    "catalogue": ("web", "go"),
    "catalogue-db": ("db", "mongodb"),
    "queue-master": ("web", "jvm"),
    "session-db": ("db", "mysql"),
    "rabbitmq": ("mq", "rabbitmq"),
}

SKIP_CONTAINERS: Final[list[str]] = ["queue-master", "rabbitmq", "session-db"]
SKIP_SERVICES: Final[list[str]] = [""]

DIAGNOSER_TARGET_DATA: Final[dict[str, list[str]]] = {
    "containers": [],  # all
    "services": [],  # all
    "nodes": [],  # all
    "middlewares": [],  # all
}



class SockShopKnowledge(PriorKnowledge):
    def __init__(self, target_metric_types: dict[str, bool]) -> None:
        super().__init__(target_metric_types)

    def get_root_service(self) -> str:
        return ROOT_SERVICE

    def get_root_container(self) -> str:
        return ROOT_CONTAINER

    def get_services(self) -> list[str]:
        return list(SERVICE_CONTAINERS.keys())

    def get_containers(self, skip: bool = False) -> list[str]:
        ctnrs = list(CONTAINER_CALL_GRAPH.keys())
        if skip:
            return [ctnr for ctnr in ctnrs if ctnr not in self.get_skip_containers()]
        return ctnrs

    def get_root_metrics(self) -> tuple[str, ...]:
        return ROOT_METRIC_LABELS

    def get_root_metric_by_type(self, red_type: str) -> str:
        return ROOT_METRIC_TYPES_AS_RED[red_type]

    def get_service_call_digraph(self) -> nx.DiGraph:
        return SERVICE_CALL_DIGRAPH

    def get_container_call_digraph(self) -> nx.DiGraph:
        return CONTAINER_CALL_DIGRAPH

    def get_container_call_graph(self, ctnr: str) -> list[str]:
        assert ctnr in CONTAINER_CALL_GRAPH, f"{ctnr} is not defined in container_call_graph"
        return CONTAINER_CALL_GRAPH[ctnr]

    def get_service_routes(self, service: str) -> list[tuple[str, ...]]:
        routes = self._generate_service_to_service_routes(SERVICE_CALL_DIGRAPH, ROOT_SERVICE)
        return routes[service]

    def get_containers_of_service(self) -> dict[str, list[str]]:
        return SERVICE_CONTAINERS

    def get_service_containers(self, service: str) -> list[str]:
        assert service in SERVICE_CONTAINERS, f"{service} is not defined in service_containers"
        return SERVICE_CONTAINERS[service]

    def get_service_by_container(self, ctnr: str) -> str:
        assert ctnr in CONTAINER_TO_SERVICE, f"{ctnr} is not defined in container_service"
        return CONTAINER_TO_SERVICE[ctnr]

    def get_service_by_container_or_empty(self, ctnr: str) -> str:
        return CONTAINER_TO_SERVICE.get(ctnr, "")

    def get_role_and_runtime_by_container(self, ctnr: str) -> tuple[str, str]:
        assert ctnr in CONTAINER_TO_RUNTIME, f"{ctnr} is not defined in container_role_runtime"
        return CONTAINER_TO_RUNTIME[ctnr]

    def get_skip_containers(self) -> list[str]:
        return SKIP_CONTAINERS

    def get_skip_services(self) -> list[str]:
        return SKIP_SERVICES

    def get_diagnoser_target_data(self) -> dict[str, list[str]]:
        return DIAGNOSER_TARGET_DATA
