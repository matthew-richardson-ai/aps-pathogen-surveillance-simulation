import random
import sys
from pathlib import Path

import numpy as np

# Allow imports from the project's Simulation folder.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from Simulation.facility import build_facility, graph_distance
from Simulation.outbreak import choose_source, expected_signals
from Simulation.sensors import observe_signals


def main():
    seed = 43

    # Keep the same facility and source for all four scenarios.
    facility = build_facility()
    source = choose_source(facility, random.Random(seed))
    ideal = expected_signals(facility, source)

    # Each scenario contains: name, noise level, missing rate.
    scenarios = [
        ("Ideal conditions", 0.00, 0.00),
        ("Noise only", 0.08, 0.00),
        ("Missing readings only", 0.00, 0.30),
        ("Noise and missing readings", 0.08, 0.30),
    ]

    print(f"Random seed: {seed}")
    print(f"True source: {source}")

    for name, noise_std, missing_rate in scenarios:
        # Reset the sensor seed so each scenario is repeatable.
        observed = observe_signals(
            ideal,
            np.random.default_rng(seed),
            noise_std=noise_std,
            missing_rate=missing_rate,
        )

        print("\n" + "=" * 60)
        print(f"Scenario: {name}")
        print(f"Noise standard deviation: {noise_std}")
        print(f"Missing rate: {missing_rate:.0%}")
        print("=" * 60)

        print(f"{'Zone':<10}{'Distance':<12}{'Ideal':<12}{'Observed'}")
        print("-" * 46)

        for zone in sorted(facility.nodes):
            distance = graph_distance(facility, source, zone)
            reading = observed[zone]

            if reading is None:
                reading_text = "Missing"
            else:
                reading_text = f"{reading:.3f}"

            print(
                f"{zone:<10}{distance:<12}"
                f"{ideal[zone]:<12.3f}{reading_text}"
            )

        missing_count = sum(
            reading is None for reading in observed.values()
        )
        print(f"\nMissing sensors: {missing_count}/{len(observed)}")


if __name__ == "__main__":
    main()