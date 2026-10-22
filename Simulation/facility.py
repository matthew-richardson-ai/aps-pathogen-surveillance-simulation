"""Facility graph used by the APS synthetic environment."""

from __future__ import annotations

import networkx as nx

DEFAULT_EDGES = (
    ("zone_1", "zone_3"),
    ("zone_2", "zone_3"),
    ("zone_3", "zone_4"),
    ("zone_3", "zone_5"),
    ("zone_4", "zone_6"),
    ("zone_5", "zone_6"),
    ("zone_6", "zone_7"),
    ("zone_7", "zone_8"),
)


def build_facility() -> nx.Graph:
    """Return the fixed eight-zone facility used for the first APS experiments."""
    graph = nx.Graph()
    graph.add_edges_from(DEFAULT_EDGES)
    return graph


def graph_distance(graph: nx.Graph, source: str, target: str) -> int:
    """Return shortest-path distance between two facility zones."""
    return nx.shortest_path_length(graph, source=source, target=target)
