import random

import networkx as nx
import numpy as np

from Simulation.facility import build_facility
from Simulation.outbreak import choose_source, expected_signals
from Simulation.sensors import observe_signals


def test_facility_has_eight_connected_zones():
    graph = build_facility()
    assert graph.number_of_nodes() == 8
    assert nx.is_connected(graph)


def test_source_selection_is_reproducible():
    graph = build_facility()

    assert choose_source(
        graph,
        random.Random(42),
    ) == choose_source(
        graph,
        random.Random(42),
    )


def test_expected_signal_is_strongest_at_source_without_noise():
    graph = build_facility()

    signals = expected_signals(
        graph,
        "zone_5",
        source_intensity=1.0,
        decay_rate=0.65,
    )

    assert signals["zone_5"] == max(signals.values())


def test_signal_decreases_with_graph_distance():
    graph = build_facility()
    signals = expected_signals(graph, "zone_1", decay_rate=0.65)

    d1 = nx.shortest_path_length(graph, "zone_1", "zone_3")
    d2 = nx.shortest_path_length(graph, "zone_1", "zone_6")

    assert d1 < d2
    assert signals["zone_3"] > signals["zone_6"]


def test_zero_noise_zero_missingness_preserves_expected_signal():
    graph = build_facility()
    expected = expected_signals(graph, "zone_3")

    observed = observe_signals(
        expected,
        np.random.default_rng(7),
        noise_std=0.0,
        missing_rate=0.0,
    )

    assert observed == expected
