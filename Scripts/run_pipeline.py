"""Run one end-to-end APS halfway-checkpoint simulation."""

from __future__ import annotations

import random

import numpy as np

from Evaluation.metrics import source_distance
from Localization.baselines import strongest_sensor
from Simulation.facility import build_facility
from Simulation.outbreak import choose_source, expected_signals
from Simulation.sensors import observe_signals


def main(seed: int = 42) -> None:
    graph = build_facility()

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    source = choose_source(graph, py_rng)
    expected = expected_signals(graph, source)
    readings = observe_signals(expected, np_rng)
    prediction = strongest_sensor(readings)

    print(f"Hidden source: {source}")
    print("Sensor readings:")

    for zone in sorted(readings):
        value = readings[zone]
        formatted = "missing" if value is None else f"{value:.3f}"
        print(f"  {zone}: {formatted}")

    print(f"Baseline prediction: {prediction}")
    print(
        "Graph-distance error: "
        f"{source_distance(graph, prediction, source)}"
    )


if __name__ == "__main__":
    main()
