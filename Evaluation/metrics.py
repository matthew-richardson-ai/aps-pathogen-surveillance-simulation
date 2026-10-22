"""Minimal evaluation metrics available at the APS halfway checkpoint."""

from __future__ import annotations

import networkx as nx


def source_correct(predicted: str, actual: str) -> int:
    """Return 1 for an exact source match and 0 otherwise."""
    return int(predicted == actual)


def source_distance(
    graph: nx.Graph,
    predicted: str,
    actual: str,
) -> int:
    """Return graph-distance error between predicted and actual sources."""
    return nx.shortest_path_length(
        graph,
        source=predicted,
        target=actual,
    )
