"""Hidden-source selection and synthetic signal propagation for APS."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping

import networkx as nx


def choose_source(graph: nx.Graph, rng: random.Random) -> str:
    """Choose one hidden outbreak source from the facility zones."""
    return rng.choice(sorted(graph.nodes))


def expected_signals(
    graph: nx.Graph,
    source: str,
    source_intensity: float = 1.0,
    decay_rate: float = 0.65,
) -> Mapping[str, float]:
    """Generate ideal signal strength using exponential decay by graph distance.

    This is a controlled spatial abstraction, not a clinical transmission model.
    """
    if source not in graph:
        raise ValueError(f"Unknown source zone: {source}")
    if source_intensity <= 0:
        raise ValueError("source_intensity must be positive")
    if decay_rate < 0:
        raise ValueError("decay_rate cannot be negative")

    signals: dict[str, float] = {}
    for zone in graph.nodes:
        distance = nx.shortest_path_length(graph, source=source, target=zone)
        signals[zone] = source_intensity * math.exp(-decay_rate * distance)
    return signals
