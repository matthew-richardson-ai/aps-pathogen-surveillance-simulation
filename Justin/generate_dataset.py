"""Create one mixed APS training CSV. Run from the project root:

    python Justin/generate_dataset.py --samples 10000

Scenario percentages are configurable training assumptions, not measured failure
frequencies. Only sensor readings and missing flags are initial model features.
"""

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Simulation.facility import build_facility
from Simulation.outbreak import expected_signals
from Simulation.sensors import observe_signals


# Edit these percentages or pass --mix name=percentage ... on the command line.
DEFAULT_MIX = {
    "normal": 45,
    "missing_data": 15,
    "broken_sensor": 10,
    "noisy_sensor": 10,
    "weak_signal": 10,
    "variable_spread": 5,
    "combined_stress": 5,
}


@dataclass(frozen=True)
class Scenario:
    noise_std: float = 0.08
    missing_rate: float = 0.02
    source_intensity: float = 1.0
    decay_rate: float = 0.65
    broken_count: int = 0


def normal_scenario(rng):
    """Routine conditions with small background noise and occasional missingness."""
    return Scenario(noise_std=rng.uniform(0.03, 0.08),
                    missing_rate=rng.uniform(0.0, 0.05))


def missing_data_scenario(rng):
    """Independent intermittent missing readings, without persistent failures."""
    return Scenario(missing_rate=rng.choice([0.10, 0.25, 0.50]))


def broken_sensor_scenario(rng):
    """Usually one, occasionally two sensors unavailable for an entire group."""
    return Scenario(broken_count=rng.choices([1, 2], weights=[80, 20])[0])


def noisy_sensor_scenario(rng):
    """Elevated measurement noise, clipped by the shared simulator to 0..1."""
    return Scenario(noise_std=rng.uniform(0.15, 0.30))


def weak_signal_scenario(rng):
    """A weaker source signal, with the same spatial propagation rule."""
    return Scenario(source_intensity=rng.uniform(0.25, 0.60))


def variable_spread_scenario(rng):
    """Slower or faster spatial decay; these are snapshots, not time sequences."""
    decay = rng.uniform(0.25, 0.45) if rng.random() < 0.5 else rng.uniform(0.85, 1.20)
    return Scenario(decay_rate=decay)


def combined_stress_scenario(rng):
    """Mix weaker signals, noise, missingness, altered decay and broken sensors."""
    return Scenario(noise_std=rng.uniform(0.15, 0.25),
                    missing_rate=rng.choice([0.10, 0.25, 0.50]),
                    source_intensity=rng.uniform(0.30, 0.70),
                    decay_rate=rng.uniform(0.35, 1.0),
                    broken_count=rng.choice([1, 2]))


SCENARIO_FUNCTIONS = {
    "normal": normal_scenario,
    "missing_data": missing_data_scenario,
    "broken_sensor": broken_sensor_scenario,
    "noisy_sensor": noisy_sensor_scenario,
    "weak_signal": weak_signal_scenario,
    "variable_spread": variable_spread_scenario,
    "combined_stress": combined_stress_scenario,
}


def allocate_counts(total, percentages):
    """Largest-remainder allocation preserves the requested total after rounding."""
    if not isinstance(total, int) or total <= 0:
        raise ValueError("samples must be a positive integer")
    if not percentages or any(not math.isfinite(v) or v < 0 for v in percentages.values()):
        raise ValueError("Percentages must be finite, nonnegative numbers")
    if not math.isclose(sum(percentages.values()), 100.0, abs_tol=1e-8):
        raise ValueError("Scenario percentages must sum to 100")
    raw = {name: total * value / 100 for name, value in percentages.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:total - sum(counts.values())]:
        counts[name] += 1
    return counts


def balanced_sources(zones, count, rng):
    """Balance true source locations within each scenario type (difference <= 1)."""
    sources = []
    while len(sources) < count:
        cycle = list(zones)
        rng.shuffle(cycle)
        sources.extend(cycle[:count - len(sources)])
    rng.shuffle(sources)
    return sources


def additional_manual_sample(expected, readings, rng, np_rng):
    """One extra manual sample: missing zone first, otherwise a random zone.

    Selection uses observed availability, never the hidden source or ideal signal.
    This is a baseline sampling policy, not a confidence-based adaptive model.
    A separate working sampler can measure a zone with a broken installed sensor.
    Its noise is 0.04; it is assumed to return a reading successfully.
    """
    candidates = [zone for zone, value in readings.items() if value is None]
    zone = rng.choice(sorted(candidates or list(readings)))
    value = float(np.clip(expected[zone] + np_rng.normal(0.0, 0.04), 0.0, 1.0))
    return zone, value


def assign_splits(rows, rng):
    """Assign 70/15/15 train/validation/test by group within each scenario type.

    All rows sharing a persistent-failure group stay in the same split. Small
    datasets may lack a scenario in a split due to whole-group rounding.
    """
    for name in SCENARIO_FUNCTIONS:
        groups = sorted({row["group_id"] for row in rows if row["scenario_type"] == name})
        if not groups:
            continue
        rng.shuffle(groups)
        counts = allocate_counts(len(groups), {"train": 70, "validation": 15, "test": 15})
        assignments = {}
        offset = 0
        for split, count in counts.items():
            for group in groups[offset:offset + count]:
                assignments[group] = split
            offset += count
        for row in rows:
            if row["scenario_type"] == name:
                row["split"] = assignments[row["group_id"]]


def generate_rows(samples=10000, seed=42, mix=None, failure_group_size=25):
    """Build mixed labeled snapshots using the shared facility/spread/sensors."""
    mix = DEFAULT_MIX if mix is None else mix
    if set(mix) - set(SCENARIO_FUNCTIONS):
        raise ValueError("Unknown scenario type in mix")
    counts = allocate_counts(samples, mix)
    if not isinstance(failure_group_size, int) or failure_group_size < 1:
        raise ValueError("failure_group_size must be a positive integer")
    graph = build_facility()
    zones = sorted(graph.nodes)
    rng = random.Random(seed)
    failure_rng = random.Random(seed + 1)
    sampling_rng = random.Random(seed + 2)
    np_rng = np.random.default_rng(seed)
    sampling_np_rng = np.random.default_rng(seed + 1)
    rows = []
    for name, count in counts.items():
        sources = balanced_sources(zones, count, rng)
        persistent = name in ("broken_sensor", "combined_stress")
        group_size = failure_group_size if persistent else 1
        for index, source in enumerate(sources):
            if index % group_size == 0:
                group_id = f"{name}_{index // group_size + 1:06d}"
                settings = SCENARIO_FUNCTIONS[name](rng)
                broken = set(failure_rng.sample(zones, settings.broken_count))
            ideal = expected_signals(graph, source,
                                     source_intensity=settings.source_intensity,
                                     decay_rate=settings.decay_rate)
            readings = observe_signals(ideal, np_rng, noise_std=settings.noise_std,
                                       missing_rate=settings.missing_rate)
            for zone in broken:
                readings[zone] = None
            extra_zone, extra_reading = additional_manual_sample(
                ideal, readings, sampling_rng, sampling_np_rng)
            row = {
                "run_id": f"run_{len(rows) + 1:06d}",
                "group_id": group_id,
                "split": "",
                "scenario_type": name,
                "true_source_node": source,
                "seed": seed,
                "noise_std": settings.noise_std,
                "missing_rate": settings.missing_rate,
                "actual_missing_fraction": sum(v is None for v in readings.values()) / len(zones),
                "source_intensity": settings.source_intensity,
                "decay_rate": settings.decay_rate,
                "broken_sensor_count": len(broken),
                "additional_sample_zone": extra_zone,
                "additional_sample_reading": extra_reading,
                "additional_sample_method": "manual_missing_first_else_random",
                "additional_sample_noise_std": 0.04,
            }
            for zone in zones:
                row[f"{zone}_reading"] = readings[zone]
                row[f"{zone}_missing"] = int(readings[zone] is None)
                row[f"{zone}_broken"] = int(zone in broken)
            rows.append(row)
    assign_splits(rows, random.Random(seed + 3))
    rng.shuffle(rows)
    return rows


def save_dataset(rows, output_dir=None):
    """Write one timestamped CSV without replacing existing datasets."""
    if not rows:
        raise ValueError("Cannot save an empty dataset")
    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "Data"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"outbreak_dataset_{timestamp}.csv"
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def generate_dataset(samples=10000, seed=42, mix=None, failure_group_size=25):
    """Generate and save one mixed dataset in Justin/Data."""
    return save_dataset(generate_rows(samples, seed, mix, failure_group_size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--failure-group-size", type=int, default=25,
                        help="Snapshots sharing the same broken sensors (default: 25)")
    parser.add_argument("--mix", nargs="+", metavar="NAME=PERCENT",
                        help="Scenario percentages, totaling 100; unspecified types are excluded")
    args = parser.parse_args()
    try:
        mix = None
        if args.mix is not None:
            mix = {}
            for item in args.mix:
                name, value = item.split("=", 1)
                if name in mix:
                    raise ValueError(f"Duplicate scenario: {name}")
                mix[name] = float(value)
        rows = generate_rows(args.samples, args.seed, mix, args.failure_group_size)
        path = save_dataset(rows)
    except ValueError as error:
        parser.error(str(error))
    print(f"Saved {len(rows):,} outbreak scenarios to {path}")
    for name, count in sorted(Counter(row["scenario_type"] for row in rows).items()):
        print(f"  {name}: {count:,} ({100 * count / len(rows):.1f}%)")


if __name__ == "__main__":
    main()
