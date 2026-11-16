import random
import numpy as np

from Simulation.facility import build_facility
from Simulation.outbreak import choose_source, expected_signals
from Simulation.sensors import observe_signals


def generate_dataset(n_samples=10000, seed=42):
    graph = build_facility()

    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    X = []
    y = []

    zones = sorted(graph.nodes)

    for _ in range(n_samples):
        source = choose_source(graph, py_rng)

        expected = expected_signals(graph, source)

        readings = observe_signals(expected, np_rng)

        features = []

        for zone in zones:
            value = readings[zone]

            if value is None:
                features.append(0.0)
                features.append(1)
            else:
                features.append(value)
                features.append(0)

        X.append(features)
        y.append(source)

    return np.array(X), np.array(y)

if __name__ == "__main__":
    X, y = generate_dataset()

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)
    print("First training example:", X[0])
    print("First outbreak source:", y[0])