"""Generate labeled APS sensor snapshots and save them beside this script.

From the project root: python Justin/generate_dataset.py --samples 10000
Requires the project's numpy and networkx dependencies.
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from Simulation.facility import build_facility
from Simulation.outbreak import choose_source, expected_signals
from Simulation.sensors import observe_signals


def generate_dataset(samples=10000, seed=42, noise_std=0.08, missing_rate=None,
                     missing_rates=None):
    """Write one row per outbreak; return the saved CSV path.

    Missing readings are blank, with a separate missing flag per zone.
    true_source_node is the training target, never an input feature.
    Each run produces a new file so previous datasets are preserved.
    samples is the total across all rates, allocated as evenly as possible.
    Each rate is a per-sensor probability, not an exact fraction per row.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")
    if missing_rate is not None and missing_rates is not None:
        raise ValueError("Choose missing_rate or missing_rates, not both")
    rates = list(missing_rates) if missing_rates is not None else (
        [missing_rate] if missing_rate is not None else [0.0, 0.10, 0.25, 0.50]
    )
    if not rates or any(not 0 <= rate <= 1 for rate in rates):
        raise ValueError("missing rates must be a nonempty list of values between 0 and 1")
    if len(set(rates)) != len(rates):
        raise ValueError("missing rates must be distinct")
    if samples < len(rates):
        raise ValueError("samples must be at least the number of missing rates")

    graph = build_facility()
    zones = sorted(graph.nodes)
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    output_dir = Path(__file__).resolve().parent / "Data"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_path = output_dir / f"outbreak_dataset_{timestamp}.csv"
    fields = ["run_id"]
    for zone in zones:
        fields.extend([f"{zone}_reading", f"{zone}_missing"])
    fields.extend(["true_source_node", "seed", "noise_std", "missing_rate"])

    with output_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index in range(1, samples + 1):
            missing_rate = rates[(index - 1) % len(rates)]
            source = choose_source(graph, py_rng)
            expected = expected_signals(graph, source)
            readings = observe_signals(
                expected, np_rng, noise_std=noise_std, missing_rate=missing_rate
            )
            row = {
                "run_id": f"run_{index:06d}",
                "true_source_node": source,
                "seed": seed,
                "noise_std": noise_std,
                "missing_rate": missing_rate,
            }
            for zone in zones:
                row[f"{zone}_reading"] = readings[zone]
                row[f"{zone}_missing"] = int(readings[zone] is None)
            writer.writerow(row)
    return output_path

# python generate_dataset.py --samples 10000 --seed 42 --noise-std 0.08 --missing-rate 0.10
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000) # number of samples
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.08)
    rate_options = parser.add_mutually_exclusive_group()
    rate_options.add_argument("--missing-rate", type=float,
                              help="Use one missing-data probability (0 to 1)")
    rate_options.add_argument("--missing-rates", type=float, nargs="+",
                              help="Probabilities to include; default: 0 0.10 0.25 0.50")
    args = parser.parse_args()
    try:
        path = generate_dataset(args.samples, args.seed, args.noise_std,
                                args.missing_rate, args.missing_rates)
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved {args.samples:,} outbreak scenarios to {path}")


if __name__ == "__main__":
    main()
