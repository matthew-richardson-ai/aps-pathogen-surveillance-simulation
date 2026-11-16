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
    facility = build_facility()
    seed = 43
    passed = 0
    total = 0

    def check(name, condition):
        nonlocal passed, total
        total += 1

        if condition:
            passed += 1
            print(f"PASS: {name}")
        else:
            print(f"FAIL: {name}")

    print("APS SIMULATION VALIDATION")
    print("=" * 60)

    # Check that source selection repeats with the same seed.
    source_a = choose_source(facility, random.Random(seed))
    source_b = choose_source(facility, random.Random(seed))

    check(
        "Same seed selects the same source",
        source_a == source_b,
    )

    # Explicitly test every zone as the source.
    for source in sorted(facility.nodes):
        print(f"\nSource: {source}")
        print("-" * 60)

        ideal = expected_signals(
            facility,
            source,
            source_intensity=1.0,
            decay_rate=0.65,
        )

        check(
            "Source signal is 1.0 and strictly strongest",
            ideal[source] == 1.0
            and all(
                ideal[source] > ideal[zone]
                for zone in facility.nodes
                if zone != source
            ),
        )

        # Compare every pair of zones at different distances.
        decreases_with_distance = all(
            ideal[near] > ideal[far]
            for near in facility.nodes
            for far in facility.nodes
            if graph_distance(facility, source, near)
            < graph_distance(facility, source, far)
        )

        check(
            "Ideal signal decreases with distance",
            decreases_with_distance,
        )

        perfect = observe_signals(
            ideal,
            np.random.default_rng(seed),
            noise_std=0.0,
            missing_rate=0.0,
        )

        check(
            "Zero noise and missingness preserve ideal readings",
            perfect == ideal,
        )

        observed_a = observe_signals(
            ideal,
            np.random.default_rng(seed),
            noise_std=0.08,
            missing_rate=0.30,
        )

        observed_b = observe_signals(
            ideal,
            np.random.default_rng(seed),
            noise_std=0.08,
            missing_rate=0.30,
        )

        check(
            "Same seed reproduces noisy and missing readings",
            observed_a == observed_b,
        )

        check(
            "Available readings stay between 0.0 and 1.0",
            all(
                0.0 <= reading <= 1.0
                for reading in observed_a.values()
                if reading is not None
            ),
        )

        all_missing = observe_signals(
            ideal,
            np.random.default_rng(seed),
            noise_std=0.0,
            missing_rate=1.0,
        )

        check(
            "100% missing rate removes every reading",
            all(reading is None for reading in all_missing.values()),
        )

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} checks passed")

    if passed == total:
        print("All validation checks passed.")
    else:
        print("Some checks failed. Review the FAIL messages.")

    # Return a failure status if any check fails.
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())