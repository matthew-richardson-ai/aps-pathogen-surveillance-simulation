"""Blind sampling must never request the true scenario configuration."""
import inspect

import numpy as np
import pytest

from Adaptive.sampler import choose_next_zone_blind
from Simulation.facility import build_facility


def test_selector_uses_observed_readings_and_public_parameter_grid():
    graph = build_facility()
    zones = sorted(graph.nodes)
    readings = dict.fromkeys(zones)
    readings["zone_4"] = 0.33
    readings["zone_5"] = 0.15
    prior = {zone: 1 / len(zones) for zone in zones}
    choice = choose_next_zone_blind(graph, readings, prior)
    assert choice is not None
    assert choice.zone in zones
    assert readings[choice.zone] is None
    assert choice.expected_gain_bits >= 0
    assert set(choice.candidate_gains) == {
        zone for zone in zones if readings[zone] is None
    }


def test_selector_has_no_actual_scenario_parameter_arguments():
    signature = inspect.signature(choose_next_zone_blind)
    for name in ("true_source", "source_intensity", "decay_rate", "noise_std"):
        assert name not in signature.parameters
    graph = build_facility()
    readings = dict.fromkeys(graph.nodes)
    prior = {zone: 1 / len(graph) for zone in graph.nodes}
    with pytest.raises(TypeError):
        choose_next_zone_blind(
            graph, readings, prior, source_intensity=0.45,
        )


def test_selector_rejects_invalid_priors_and_exhausted_candidates():
    graph = build_facility()
    prior = {zone: 1 / len(graph) for zone in graph.nodes}
    readings = {zone: 0.2 for zone in graph.nodes}
    assert choose_next_zone_blind(graph, readings, prior) is None
    readings["zone_1"] = None
    bad = {zone: float("nan") for zone in graph.nodes}
    with pytest.raises(ValueError, match="probabilities"):
        choose_next_zone_blind(graph, readings, bad)
