from functools import cache
from typing import Final

import networkx as nx

from .base import PriorKnowledge

TARGET_APP_NAME: Final[str] = "train-ticket"

ROOT_SERVICE: Final[str] = "ts-ui-dashboard"
ROOT_CONTAINER: Final[str] = "ts-ui-dashboard"
ROOT_METRIC_LABELS: Final[tuple[str, ...]] = (
    "s-ts-ui-dashboard_request_duration_seconds",
    "s-ts-ui-dashboard_requests_count",
    "s-ts-ui-dashboard_requests_errors_count",
    # FIXME: this is temporary solusion because the other 3 metrics are not available.
    "m-ts-ui-dashboard_nginx_http_request_duration_seconds",
    "m-ts-ui-dashboard_nginx_http_response_count_total",
    "m-ts-ui-dashboard_nginx_http_response_size_bytes",
    "m-ts-ui-dashboard_nginx_http_request_size_bytes",
)

ROOT_METRIC_TYPES_AS_RED: Final[dict[str, str]] = {
    "latency": "s-ts-ui-dashboard_request_duration_seconds",
    "throughput": "s-ts-ui-dashboard_requests_count",
    "errors": "s-ts-ui-dashboard_requests_errors_count",
}

SERVICE_CALL_DIGRAPH: Final[nx.DiGraph] = nx.DiGraph(
    [
        ("ts-ui-dashboard", "ts-travel"),
        ("ts-ui-dashboard", "ts-travel2"),
        ("ts-ui-dashboard", "ts-user"),
        ("ts-ui-dashboard", "ts-auth"),
        ("ts-ui-dashboard", "ts-verification-code"),
        ("ts-ui-dashboard", "ts-station"),
        ("ts-ui-dashboard", "ts-train"),
        ("ts-ui-dashboard", "ts-config"),
        ("ts-ui-dashboard", "ts-security"),
        ("ts-ui-dashboard", "ts-execute"),
        ("ts-ui-dashboard", "ts-contacts"),
        ("ts-ui-dashboard", "ts-order"),
        ("ts-ui-dashboard", "ts-order-other"),
        ("ts-ui-dashboard", "ts-preserve"),
        ("ts-ui-dashboard", "ts-preserve-other"),
        ("ts-ui-dashboard", "ts-price"),
        ("ts-ui-dashboard", "ts-basic"),
        ("ts-ui-dashboard", "ts-ticketinfo"),
        ("ts-ui-dashboard", "ts-notification"),
        ("ts-ui-dashboard", "ts-inside-payment"),
        ("ts-ui-dashboard", "ts-rebook"),
        ("ts-ui-dashboard", "ts-cancel"),
        ("ts-ui-dashboard", "ts-route"),
        ("ts-ui-dashboard", "ts-assurance"),
        ("ts-ui-dashboard", "ts-ticket-office"),
        ("ts-ui-dashboard", "ts-travel-plan"),
        ("ts-ui-dashboard", "ts-consign"),
        ("ts-ui-dashboard", "ts-voucher"),
        ("ts-ui-dashboard", "ts-route-plan"),
        ("ts-ui-dashboard", "ts-food"),
        ("ts-ui-dashboard", "ts-news"),
        ("ts-ui-dashboard", "ts-admin-basic-info"),
        ("ts-ui-dashboard", "ts-admin-order"),
        ("ts-ui-dashboard", "ts-admin-route"),
        ("ts-ui-dashboard", "ts-admin-travel"),
        ("ts-ui-dashboard", "ts-admin-user"),
        ("ts-ui-dashboard", "ts-avatar"),
        ("ts-admin-basic-info", "ts-station"),
        ("ts-admin-basic-info", "ts-train"),
        ("ts-admin-basic-info", "ts-config"),
        ("ts-admin-basic-info", "ts-price"),
        ("ts-admin-basic-info", "ts-contacts"),
        ("ts-admin-order", "ts-order"),
        ("ts-admin-order", "ts-order-other"),
        ("ts-admin-route", "ts-route"),
        ("ts-admin-travel", "ts-travel"),
        ("ts-admin-travel", "ts-travel2"),
        ("ts-admin-user", "ts-user"),
        ("ts-auth", "ts-verification-code"),
        ("ts-basic", "ts-station"),
        ("ts-basic", "ts-train"),
        ("ts-basic", "ts-route"),
        ("ts-basic", "ts-price"),
        ("ts-cancel", "ts-notification"),
        ("ts-cancel", "ts-order"),
        ("ts-cancel", "ts-order-other"),
        ("ts-cancel", "ts-inside-payment"),
        ("ts-cancel", "ts-user"),
        ("ts-cancel", "ts-order"),
        ("ts-cancel", "ts-order-other"),
        ("ts-consign", "ts-consign-price"),
        ("ts-execute", "ts-order"),
        ("ts-execute", "ts-order-other"),
        ("ts-execute", "ts-order-other"),
        ("ts-food", "ts-food-map"),
        ("ts-food", "ts-travel"),
        ("ts-food", "ts-station"),
        ("ts-inside-payment", "ts-order"),
        ("ts-inside-payment", "ts-order-other"),
        ("ts-inside-payment", "ts-payment"),
        ("ts-order", "ts-station"),
        ("ts-order-other", "ts-station"),
        ("ts-preserve", "ts-ticketinfo"),
        ("ts-preserve", "ts-seat"),
        ("ts-preserve", "ts-user"),
        ("ts-preserve", "ts-assurance"),
        ("ts-preserve", "ts-station"),
        ("ts-preserve", "ts-security"),
        ("ts-preserve", "ts-travel"),
        ("ts-preserve", "ts-security"),
        ("ts-preserve", "ts-contacts"),
        ("ts-preserve", "ts-order"),
        ("ts-preserve", "ts-food"),
        ("ts-preserve", "ts-consign"),
        ("ts-preserve-other", "ts-ticketinfo"),
        ("ts-preserve-other", "ts-seat"),
        ("ts-preserve-other", "ts-user"),
        ("ts-preserve-other", "ts-assurance"),
        ("ts-preserve-other", "ts-station"),
        ("ts-preserve-other", "ts-security"),
        ("ts-preserve-other", "ts-travel2"),
        ("ts-preserve-other", "ts-security"),
        ("ts-preserve-other", "ts-contacts"),
        ("ts-preserve-other", "ts-order-other"),
        ("ts-preserve-other", "ts-food"),
        ("ts-preserve-other", "ts-consign"),
        ("ts-rebook", "ts-seat"),
        ("ts-rebook", "ts-travel"),
        ("ts-rebook", "ts-travel2"),
        ("ts-rebook", "ts-order"),
        ("ts-rebook", "ts-order-other"),
        ("ts-rebook", "ts-station"),
        ("ts-rebook", "ts-inside-payment"),
        ("ts-route-plan", "ts-route"),
        ("ts-route-plan", "ts-travel"),
        ("ts-route-plan", "ts-travel2"),
        ("ts-route-plan", "ts-station"),
        ("ts-seat", "ts-travel"),
        ("ts-seat", "ts-travel2"),
        ("ts-seat", "ts-order"),
        ("ts-seat", "ts-order-other"),
        ("ts-seat", "ts-config"),
        ("ts-security", "ts-order"),
        ("ts-security", "ts-order-other"),
        ("ts-ticketinfo", "ts-basic"),
        ("ts-travel-plan", "ts-seat"),
        ("ts-travel-plan", "ts-route-plan"),
        ("ts-travel-plan", "ts-travel"),
        ("ts-travel-plan", "ts-travel2"),
        ("ts-travel-plan", "ts-ticketinfo"),
        ("ts-travel-plan", "ts-station"),
        ("ts-travel", "ts-ticketinfo"),
        ("ts-travel", "ts-order"),
        ("ts-travel", "ts-train"),
        ("ts-travel", "ts-route"),
        ("ts-travel", "ts-seat"),
        ("ts-travel2", "ts-ticketinfo"),
        ("ts-travel2", "ts-order"),
        ("ts-travel2", "ts-train"),
        ("ts-travel2", "ts-route"),
        ("ts-travel2", "ts-seat"),
        ("ts-user", "ts-auth"),
        ("ts-voucher", "ts-order"),
        ("ts-voucher", "ts-order-other"),
    ]
)


SERVICE_CONTAINERS: Final[dict[str, list[str]]] = {
    "rabbitmq": ["rabbitmq"],
    "ts-admin-basic-info": ["ts-admin-basic-info-service"],
    "ts-admin-order": ["ts-admin-order-service"],
    "ts-admin-route": ["ts-admin-route-service"],
    "ts-admin-travel": ["ts-admin-travel-service"],
    "ts-admin-user": ["ts-admin-user-service"],
    "ts-assurance": ["ts-assurance-service", "ts-assurance-mongo"],
    "ts-auth": ["ts-auth-service", "ts-auth-mongo"],
    "ts-avatar": ["ts-avatar-service"],
    "ts-basic": ["ts-basic-service"],
    "ts-cancel": ["ts-cancel-service"],
    "ts-config": ["ts-config-service", "ts-config-mongo"],
    "ts-consign": ["ts-consign-service", "ts-consign-mongo"],
    "ts-consign-price": ["ts-consign-price-service", "ts-consign-price-mongo"],
    "ts-contacts": ["ts-contacts-service", "ts-contacts-mongo"],
    # "ts-delivery": ["ts-delivery-service", "ts-delivery-mongo"],
    "ts-execute": ["ts-execute-service"],
    "ts-food-map": ["ts-food-map-service", "ts-food-map-mongo"],
    "ts-food": ["ts-food-service", "ts-food-mongo"],
    "ts-inside-payment": ["ts-inside-payment-service", "ts-inside-payment-mongo"],
    "ts-news": ["ts-news-service"],
    "ts-notification": ["ts-notification-service", "ts-notification-mongo"],
    "ts-order": ["ts-order-service", "ts-order-mongo"],
    "ts-order-other": ["ts-order-other-service", "ts-order-other-mongo"],
    "ts-payment": ["ts-payment-service", "ts-payment-mongo"],
    "ts-preserve": ["ts-preserve-service"],
    "ts-preserve-other": ["ts-preserve-other-service"],
    "ts-price": ["ts-price-service", "ts-price-mongo"],
    "ts-rebook": ["ts-rebook-service"],
    "ts-route-plan": ["ts-route-plan-service"],
    "ts-route": ["ts-route-service", "ts-route-mongo"],
    "ts-seat": ["ts-seat-service"],
    "ts-security": ["ts-security-service", "ts-security-mongo"],
    "ts-station": ["ts-station-service", "ts-station-mongo"],
    "ts-ticket-office": ["ts-ticket-office-service", "ts-ticket-office-mongo"],
    "ts-ticketinfo": ["ts-ticketinfo-service"],
    "ts-train": ["ts-train-service", "ts-train-mongo"],
    "ts-travel": ["ts-travel-service", "ts-travel-mongo"],
    "ts-travel-plan": ["ts-travel-plan-service"],
    "ts-travel2": ["ts-travel2-service", "ts-travel2-mongo"],
    "ts-ui-dashboard": ["ts-ui-dashboard"],
    "ts-user": ["ts-user-service", "ts-user-mongo"],
    "ts-verification-code": ["ts-verification-code-service"],
    "ts-voucher": ["ts-voucher-service", "ts-voucher-mysql"],
}

CONTAINER_TO_SERVICE: Final[dict[str, str]] = {c: s for s, ctnrs in SERVICE_CONTAINERS.items() for c in ctnrs}


def generate_container_call_graph() -> nx.DiGraph:
    ctnr_g = nx.DiGraph()
    # build call graph of ctnr-to-ctnr in a service
    for ctnrs in SERVICE_CONTAINERS.values():
        if len(ctnrs) >= 2:
            ctnr_g.add_edges_from(nx.utils.pairwise(ctnrs))
    # build call graph of service-to-service
    for edge in nx.edges(SERVICE_CALL_DIGRAPH):
        src_service, dst_service = edge[0], edge[1]
        ctnr_g.add_edge(SERVICE_CONTAINERS[src_service][0], SERVICE_CONTAINERS[dst_service][0])
    return ctnr_g


CONTAINER_CALL_DIGRAPH: Final[nx.DiGraph] = generate_container_call_graph()

CONTAINER_CALL_GRAPH: Final[dict[str, list[str]]] = {
    n: list(nbr.keys()) for n, nbr in CONTAINER_CALL_DIGRAPH.adjacency()
}


@cache
def generate_container_runtime() -> dict[str, tuple[str, str]]:
    """Return a dict of container name to (role name, runtime name)"""
    ctnrs = list(CONTAINER_CALL_GRAPH.keys())
    ctnr_to_runtime: dict[str, tuple[str, str]] = {}
    for ctnr in ctnrs:
        if ctnr == "ts-ui-dashboard":
            ctnr_to_runtime[ctnr] = ("proxy", "nginx")
        elif ctnr == "rabbitmq":
            ctnr_to_runtime[ctnr] = ("mq", "rabbitmq")
        elif ctnr == "ts-ticketinfo-service":
            ctnr_to_runtime[ctnr] = ("web", "nodejs")
        elif ctnr == "ts-news-service":
            ctnr_to_runtime[ctnr] = ("web", "go")
        elif ctnr == "ts-voucher-service":
            ctnr_to_runtime[ctnr] = ("web", "python")
        elif ctnr.endswith("-mongo"):
            ctnr_to_runtime[ctnr] = ("db", "mongodb")
        elif ctnr.endswith("-mysql"):
            ctnr_to_runtime[ctnr] = ("db", "mysql")
        elif ctnr.endswith("-service"):
            ctnr_to_runtime[ctnr] = ("web", "jvm")
        else:
            assert False, f"unknown container: {ctnr}"
    return ctnr_to_runtime


SKIP_CONTAINERS: Final[list[str]] = ["ts-delivery-service", "ts-delivery-mongo"]
SKIP_SERVICES: Final[list[str]] = ["ts-delivery"]

DIAGNOSER_TARGET_DATA: Final[dict[str, list[str]]] = {
    "containers": [],  # all
    "services": [],  # all
    "nodes": [],  # all
    "middlewares": [],  # all
}


class TrainTicketKnowledge(PriorKnowledge):
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
        assert service in routes, f"{service} is not defined in service_call_graph"
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

    def get_skip_containers(self) -> list[str]:
        return SKIP_CONTAINERS

    def get_skip_services(self) -> list[str]:
        return SKIP_SERVICES

    def get_role_and_runtime_by_container(self, ctnr: str) -> tuple[str, str]:
        ctnr_runtime = generate_container_runtime()
        assert ctnr in ctnr_runtime, f"{ctnr} is not defined in container_runtime"
        return ctnr_runtime[ctnr]

    def get_diagnoser_target_data(self) -> dict[str, list[str]]:
        return DIAGNOSER_TARGET_DATA
