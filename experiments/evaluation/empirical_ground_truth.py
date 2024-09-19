from typing import Final

from dataset.metric import MetricType, parse_metric
from priorknowledge.base import PriorKnowledge

FAULT_TO_ROOT_FAULT_METRIC_PATTERNS: Final[dict[str, dict[tuple[str, str], list[str]]]] = {
    "pod-cpu-hog": {
        ("*", "container"): [
            "cpu_usage_seconds_total",
            "cpu_user_seconds_total",
            "threads",
        ],
        ("*", "jvm"): [
            "java_lang_OperatingSystem_SystemCpuLoad",
            "java_lang_OperatingSystem_ProcessCpuLoad",
            "java_lang_OperatingSystem_ProcessCpuTime",
        ],
        ("*", "mongodb"): [
            "mongodb_sys_cpu_processes",
            "mongodb_sys_cpu_procs_running",
            "mongodb_sys_cpu_user_ms",
            "mongodb_sys_cpu_idle_ms",
            "mongodb_sys_cpu_ctxt",
        ],
    },
    "pod-memory-hog": {
        ("*", "container"): [
            "memory_rss",
            "memory_usage_bytes",
            "memory_working_set_bytes",
            "memory_cache",
            "memory_mapped_file",
            "threads",
        ],
        ("*", "jvm"): [
            "java_lang_MemoryPool_Usage_used",
            "java_lang_OperatingSystem_FreePhysicalMemorySize",
        ],
        ("*", "mongodb"): [
            "mongodb_sys_memory_Buffers_kb",
            "mongodb_sys_memory_MemAvailable_kb",
            "mongodb_sys_memory_MemFree_kb",
            "mongodb_sys_memory_Active_kb",
            "mongodb_sys_memory_Active_file_kb",
        ],
    },
}

def get_fault_to_root_fault_base_metrics(
    chaos_type: str, role: str, runtime: str, optional_candidates: bool = False
) -> list[str]:
    return FAULT_TO_ROOT_FAULT_METRIC_PATTERNS[chaos_type].get((role, runtime), [])


def select_root_fault_metrics(
    pk: PriorKnowledge, metrics: set[str], fault_type: str, fault_comp: str,
) -> set[str]:
    root_fault_metrics: set[str] = set()
    for metric in metrics:
        comp, base_metric, metric_type = parse_metric(metric)
        if comp in pk.get_skip_containers() or comp in pk.get_skip_services():
            continue
        match metric_type:
            case MetricType.UNEXPECTED:
                pass
            case MetricType.CONTAINER:
                role, _ = pk.get_role_and_runtime_by_container(comp)
                for _role in ["*", role]:
                    root_fault_base_metrics = get_fault_to_root_fault_base_metrics(fault_type, _role, "container")
                    if root_fault_base_metrics is not None and len(root_fault_base_metrics) > 0:
                        for root_fault_base_metrics in root_fault_base_metrics:
                            if metric == f"c-{fault_comp}_{root_fault_base_metrics}":
                                root_fault_metrics.add(metric)
            case MetricType.MIDDLEWARE:
                role, _ = pk.get_role_and_runtime_by_container(comp)
                for _role in ["*", role]:
                    root_fault_base_metrics = get_fault_to_root_fault_base_metrics(fault_type, _role, "middleware")
                    if root_fault_base_metrics is not None and len(root_fault_base_metrics) > 0:
                        for root_fault_base_metrics in root_fault_base_metrics:
                            if metric == f"m-{fault_comp}_{root_fault_base_metrics}":
                                root_fault_metrics.add(metric)
            case MetricType.SERVICE:
                pass
            case MetricType.NODE:
                # TODO: handling any other role
                root_fault_base_metrics = get_fault_to_root_fault_base_metrics(fault_type, "*", "node")
                if root_fault_base_metrics is not None and len(root_fault_base_metrics) > 0:
                    for root_fault_base_metrics in root_fault_base_metrics:
                        if metric == f"n-{fault_comp}_{root_fault_base_metrics}":
                            root_fault_metrics.add(metric)
            case _:
                assert False, f"Unknown metric node type: {metric}, {metric_type}"
    return root_fault_metrics
