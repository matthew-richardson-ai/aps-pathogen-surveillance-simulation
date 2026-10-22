from Evaluation.metrics import source_correct, source_distance
from Simulation.facility import build_facility


def test_source_correct():
    assert source_correct("zone_2", "zone_2") == 1
    assert source_correct("zone_2", "zone_3") == 0


def test_source_distance():
    graph = build_facility()

    assert source_distance(graph, "zone_1", "zone_1") == 0
    assert source_distance(graph, "zone_1", "zone_3") == 1
