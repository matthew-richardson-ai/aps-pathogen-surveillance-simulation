"""Tests for the interactive 3D APS cutaway facility visualization."""
import pytest

from OB.facility_3d import ROOM_CENTERS, facility_3d_plot, room_color
from OB.live_engine import Scenario, simulate, sample_one_more, predict_baseline
from Simulation.facility import build_facility


def room_mesh(figure, zone):
    return next(trace for trace in figure.data
                if trace.type == "mesh3d" and trace.name == zone)


def test_3d_rooms_follow_the_real_eight_zone_graph():
    graph = build_facility()
    snapshot = simulate(Scenario(seed=42, initial_sensors=5))
    figure = facility_3d_plot(snapshot, graph, predict_baseline(snapshot.readings))
    assert set(ROOM_CENTERS) == set(graph.nodes)
    assert len([trace for trace in figure.data if trace.type == "mesh3d"]) == 8
    assert len(figure.data[0].x) == 3 * graph.number_of_edges()
    assert figure.layout.scene.dragmode == "orbit"
    for zone in graph.nodes:
        assert room_mesh(figure, zone).color == room_color(snapshot.readings[zone])


def test_missing_room_is_distinct_from_zero_reading():
    assert room_color(None) != room_color(0.0)
    assert room_color(0.0) != room_color(0.90)


def test_one_additional_sample_updates_room_color():
    initial = simulate(Scenario(seed=32, initial_sensors=3, missing_rate=0))
    selected_zone = next(zone for zone, reading in initial.readings.items()
                         if reading is None)
    graph = build_facility()
    before = facility_3d_plot(initial, graph, initial.initial_prediction)
    measured = sample_one_more(initial, selected_zone)
    after = facility_3d_plot(
        measured, graph, predict_baseline(measured.readings),
        show_source=True, selected_zone=selected_zone,
    )
    assert room_mesh(before, selected_zone).color == room_color(None)
    assert room_mesh(after, selected_zone).color != room_color(None)
    labels = [trace.name for trace in after.data]
    assert "Additional measurement" in labels
    assert "True source" in labels


def test_cannot_use_mismatched_facility_graph():
    snapshot = simulate(Scenario(seed=42))
    graph = build_facility()
    graph.remove_node("zone_1")
    with pytest.raises(ValueError, match="rooms must match"):
        facility_3d_plot(snapshot, graph, None)
