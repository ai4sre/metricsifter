from itertools import combinations

import networkx as nx
from dataset.metric import (
    is_container_metric,
    is_middleware_metric,
    is_node_metric,
    is_service_metric,
    parse_metric,
)
from priorknowledge.base import PriorKnowledge


def reverse_edge_direction(G: nx.DiGraph, u: str, v: str) -> None:
    attr = G[u][v]
    G.remove_edge(u, v)
    G.add_edge(v, u, attr=attr) if attr else G.add_edge(v, u)


def set_bidirected_edge(G: nx.DiGraph, u: str, v: str) -> None:
    G.add_edge(u, v)
    G.add_edge(v, u)



def get_forbits(metrics: set[str], pk: PriorKnowledge) -> list[tuple[str, str]]:
    init_g: nx.Graph = prepare_init_graph(metrics, pk)
    init_dg = fix_edge_directions_in_causal_graph(init_g, pk)
    forbits = []
    for node1, node2 in combinations(init_dg.nodes, 2):
        if not init_dg.has_edge(node1, node2):
            forbits.append((node1.label, node2.label))
        if not init_dg.has_edge(node2, node1):
            forbits.append((node2.label, node1.label))
    return forbits


def prepare_init_graph(
    metrics: set[str],
    pk: PriorKnowledge,
    enable_orientation: bool = False,
) -> nx.Graph:
    """Prepare initialized causal graph."""
    init_g = nx.Graph()
    for u, v in combinations(metrics, 2):
        init_g.add_edge(u, v)
    RG: nx.Graph = build_subgraph_of_removal_edges(metrics, pk)
    init_g.remove_edges_from(RG.edges())
    if enable_orientation:
        return fix_edge_directions_in_causal_graph(init_g, pk)
    return init_g


def build_subgraph_of_removal_edges(
    metrics: set[str], pk: PriorKnowledge
) -> nx.Graph:
    """Build a subgraph consisting of removal edges with prior knowledges."""
    ctnr_graph: nx.Graph = pk.get_container_call_digraph().to_undirected()
    service_graph: nx.Graph = pk.get_service_call_digraph().to_undirected()
    node_ctnr_graph: nx.Graph = pk.get_nodes_to_containers_graph()

    G: nx.Graph = nx.Graph()
    for u, v in combinations(metrics, 2):
        u_comp = parse_metric(u)[0]
        v_comp = parse_metric(v)[0]
        # "container" and "middleware" is the same.
        if (is_container_metric(u) or is_middleware_metric(u)) and (
            is_container_metric(v) or is_middleware_metric(v)
        ):
            if u_comp == v_comp or ctnr_graph.has_edge(u_comp, v_comp):
                continue
        elif (is_container_metric(u) or is_middleware_metric(u)) and is_service_metric(v):
            u_service: str = pk.get_service_by_container(u_comp)
            if u_service == v_comp or service_graph.has_edge(u_service, v_comp):
                continue
        elif is_service_metric(u) and (is_container_metric(v) or is_middleware_metric(v)):
            v_service: str = pk.get_service_by_container(v_comp)
            if u_comp == v_service or service_graph.has_edge(u_comp, v_service):
                continue
        elif is_service_metric(u) and is_service_metric(v):
            if u_comp == v_comp or service_graph.has_edge(u_comp, v_comp):
                continue
        elif is_node_metric(u) and is_node_metric(v):
            # each node has no connectivity.
            pass
        elif is_node_metric(u) and (is_container_metric(v) or is_middleware_metric(v)):
            if node_ctnr_graph.has_edge(u_comp, v_comp):
                continue
        elif (is_container_metric(u) or is_middleware_metric(u)) and is_node_metric(v):
            if node_ctnr_graph.has_edge(u_comp, v_comp):
                continue
        elif is_node_metric(u) and is_service_metric(v):
            v_ctnrs: list[str] = pk.get_service_containers(v_comp)
            has_ctnr_on_node = False
            for v_ctnr in v_ctnrs:
                if node_ctnr_graph.has_edge(u_comp, v_ctnr):
                    has_ctnr_on_node = True
                    break
            if has_ctnr_on_node:
                continue
        elif is_service_metric(u) and is_node_metric(v):
            u_ctnrs: list[str] = pk.get_service_containers(u_comp)
            has_ctnr_on_node = False
            for u_ctnr in u_ctnrs:
                if node_ctnr_graph.has_edge(u_ctnr, v_comp):
                    has_ctnr_on_node = True
                    break
            if has_ctnr_on_node:
                continue
        else:
            raise ValueError(f"'{u}' and '{v}' is an unexpected pair")
        # use node number because 'pgmpy' package handles only graph nodes consisted with numpy array.
        G.add_edge(u, v)
    return G


def fix_edge_direction_based_hieralchy(
    G: nx.DiGraph, u: str, v: str, pk: PriorKnowledge
) -> None:
    u_comp, v_comp = parse_metric(u)[0], parse_metric(v)[0]
    # Force direction from (container -> service) to (service -> container) in same service
    if is_service_metric(u) and is_container_metric(v):
        # check whether u and v in the same service
        v_service = pk.get_service_by_container(v_comp)
        if u_comp == v_service:
            reverse_edge_direction(G, u, v)


def fix_edge_direction_based_network_call(
    G: nx.DiGraph, u: str, v: str,
    service_dep_graph: nx.DiGraph,
    container_dep_graph: nx.DiGraph,
    pk: PriorKnowledge,
) -> None:
    u_comp, v_comp = parse_metric(u)[0], parse_metric(v)[0]

    # From service to service
    if is_service_metric(u) and is_service_metric(v):
        # If u and v is in the same service, force bi-directed edge.
        if u_comp == v_comp:
            set_bidirected_edge(G, u, v)
        elif (v_comp not in service_dep_graph[u_comp]) and (
            u_comp in service_dep_graph[v_comp]
        ):
            reverse_edge_direction(G, u, v)

    # From container to container
    if (is_container_metric(u) or is_middleware_metric(u)) and is_container_metric(v) or is_middleware_metric(v):
        # If u and v is in the same container, force bi-directed edge.
        if u_comp == v_comp:
            set_bidirected_edge(G, u, v)
        elif (v_comp not in container_dep_graph[u_comp]) and (
            u_comp in container_dep_graph[v_comp]
        ):
            reverse_edge_direction(G, u, v)

    # From service to container
    if is_service_metric(u) and (is_container_metric(v) or is_middleware_metric(v)):
        v_service = pk.get_service_by_container(v_comp)
        if (v_service not in service_dep_graph[u_comp]) and (
            u_comp in service_dep_graph[v_service]
        ):
            reverse_edge_direction(G, u, v)

    # From container to service
    if (is_container_metric(u) or is_middleware_metric(u)) and is_service_metric(v):
        u_service = pk.get_service_by_container(u_comp)
        if (v_comp not in service_dep_graph[u_service]) and (
            u_service in service_dep_graph[v_comp]
        ):
            reverse_edge_direction(G, u, v)


def fix_edge_directions_in_causal_graph(
    G: nx.Graph | nx.DiGraph,
    pk: PriorKnowledge,
) -> nx.DiGraph:
    """Fix the edge directions in the causal graphs.
    1. Fix directions based on the system hieralchy such as a service and a container
    2. Fix directions based on the network call graph.
    """
    if not G.is_directed():
        G = G.to_directed()
    service_dep_graph: nx.DiGraph = pk.get_service_call_digraph().reverse()
    container_dep_graph: nx.DiGraph = pk.get_container_call_digraph().reverse()
    # Traverse the all edges of G via the neighbors
    for u, nbrsdict in G.adjacency():
        nbrs = list(
            nbrsdict.keys()
        )  # to avoid 'RuntimeError: dictionary changed size during iteration'
        for v in nbrs:
            # u -> v
            fix_edge_direction_based_hieralchy(G, u, v, pk)
            fix_edge_direction_based_network_call(
                G, u, v, service_dep_graph, container_dep_graph, pk
            )
    return G
