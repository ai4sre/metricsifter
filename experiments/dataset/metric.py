from enum import IntEnum, auto, unique
from typing import Final

METRIC_TYPE_SERVICES: Final[str] = "services"
METRIC_TYPE_CONTAINERS: Final[str] = "containers"
METRIC_TYPE_NODES: Final[str] = "nodes"
METRIC_TYPE_MIDDLEWARES: Final[str] = "middlewares"

METRIC_TYPE_MAP: Final[list[tuple[str, str]]] = [
    ("c-", METRIC_TYPE_CONTAINERS),
    ("s-", METRIC_TYPE_SERVICES),
    ("m-", METRIC_TYPE_MIDDLEWARES),
    ("n-", METRIC_TYPE_NODES),
]

METRIC_PREFIX_TO_TYPE: Final[dict[str, str]] = dict([(v, k) for k, v in METRIC_TYPE_MAP])

ALL_METRIC_TYPES: Final[dict[str, bool]] = {
    METRIC_TYPE_SERVICES: True,
    METRIC_TYPE_NODES: True,
    METRIC_TYPE_CONTAINERS: True,
    METRIC_TYPE_MIDDLEWARES: True,
}

@unique
class MetricType(IntEnum):
    UNEXPECTED = auto()
    CONTAINER = auto()
    SERVICE = auto()
    NODE = auto()
    MIDDLEWARE = auto()


def parse_metric(metric: str) -> tuple[str, str, MetricType]:
    """Parse metric name to get metric type and metric name.
    Example:
        >>> parse_metric("c-user_cpu_usage_seconds_total")
        ("cpu_usage", "seconds_total")
    """
    if metric.startswith("c-"):
        comp_type = MetricType.CONTAINER
    elif metric.startswith("s-"):
        comp_type = MetricType.SERVICE
    elif metric.startswith("n-"):
        comp_type = MetricType.NODE
    elif metric.startswith("m-"):
        comp_type = MetricType.MIDDLEWARE
    else:
        comp_type = MetricType.UNEXPECTED
        return "", "", comp_type

    comp, base_name = metric.split("-", maxsplit=1)[1].split("_", maxsplit=1)
    return comp, base_name, comp_type


def is_container_metric(metric: str) -> bool:
    return metric.startswith("c-")

def is_middleware_metric(metric: str) -> bool:
    return metric.startswith("m-")

def is_service_metric(metric: str) -> bool:
    return metric.startswith("s-")

def is_node_metric(metric: str) -> bool:
    return metric.startswith("n-")
