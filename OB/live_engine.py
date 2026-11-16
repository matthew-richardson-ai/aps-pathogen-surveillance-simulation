"""Reproducible, simulation-only APS dashboard engine.

This is a spatial snapshot simulation, not a temporal transport model.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import random
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simulation.facility import build_facility, graph_distance
from Simulation.outbreak import choose_source, expected_signals
from Simulation.sensors import observe_signals
from Localization.baselines import strongest_sensor


@dataclass(frozen=True)
class Scenario:
    seed: int = 42
    source: str | None = None
    initial_sensors: int = 5
    noise_std: float = 0.08
    missing_rate: float = 0.10
    source_intensity: float = 1.0
    decay_rate: float = 0.65


@dataclass(frozen=True)
class Snapshot:
    parameters: Scenario
    true_source: str
    ideal_signals: dict[str, float]
    readings: dict[str, float | None]
    initially_selected: tuple[str, ...]
    initial_prediction: str | None
    requested_zone: str | None = None
    requested_reading: float | None = None


def predict_baseline(readings: dict[str, float | None]) -> str | None:
    """Return None when there are no valid readings instead of crashing."""
    if not any(v is not None for v in readings.values()):
        return None
    return strongest_sensor(readings)


def simulate(scenario: Scenario) -> Snapshot:
    """Generate a controlled snapshot with some initially unobserved zones."""
    graph = build_facility()
    zones = sorted(graph.nodes)
    if scenario.source is not None and scenario.source not in graph:
        raise ValueError("Requested source is not in the eight-zone facility")
    if not 1 <= scenario.initial_sensors <= len(zones):
        raise ValueError("initial_sensors must be between 1 and eight")
    source = scenario.source or choose_source(graph, random.Random(scenario.seed))
    ideal = dict(expected_signals(
        graph, source,
        source_intensity=scenario.source_intensity,
        decay_rate=scenario.decay_rate,
    ))
    all_readings = observe_signals(
        ideal, np.random.default_rng(scenario.seed + 1001),
        noise_std=scenario.noise_std, missing_rate=scenario.missing_rate,
    )
    selected = tuple(sorted(np.random.default_rng(scenario.seed + 2001).choice(
        zones, size=scenario.initial_sensors, replace=False).tolist()))
    readings = {z: (all_readings[z] if z in selected else None) for z in zones}
    return Snapshot(scenario, source, ideal, readings, selected, predict_baseline(readings))


def sample_one_more(snapshot: Snapshot, zone: str, noise_std: float = 0.04) -> Snapshot:
    """Simulate a single manually chosen follow-up observation.

    This is NOT an implemented adaptive/AI sampling policy. The manual
    measurement succeeds even if the installed sensor was missing.
    """
    if snapshot.requested_zone is not None:
        raise ValueError("Only one additional sample is allowed per run")
    if zone not in snapshot.readings:
        raise ValueError("Zone is not in this facility")
    if snapshot.readings[zone] is not None:
        raise ValueError("Additional sample must target a zone without a reading")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")
    rng = np.random.default_rng(snapshot.parameters.seed + 30_001 + int(zone.split('_')[-1]) * 67)
    extra = float(np.clip(snapshot.ideal_signals[zone] + rng.normal(0, noise_std), 0.0, 1.0))
    readings = {**snapshot.readings, zone: extra}
    return replace(snapshot, readings=readings, requested_zone=zone, requested_reading=extra)


def baseline_error(snapshot: Snapshot, *, after_sample: bool = True) -> int | None:
    prediction = predict_baseline(snapshot.readings) if after_sample else snapshot.initial_prediction
    return None if prediction is None else graph_distance(build_facility(), prediction, snapshot.true_source)


def batch_baseline(scenario: Scenario, n: int = 100) -> list[dict]:
    """Evaluate only the baseline on independent seeded synthetic snapshots.

    All trials count toward accuracy; abstentions are incorrect and have
    no graph distance. Forces random sources to cover multiple zones.
    """
    if not 1 <= n <= 2000:
        raise ValueError("Number of trials must be in [1, 2000]")
    output = []
    for i in range(n):
        trial = replace(scenario, seed=scenario.seed + i, source=None)
        snap = simulate(trial)
        pred = snap.initial_prediction
        output.append({
            "run_id": f"baseline_{i+1:04d}",
            "seed": trial.seed,
            "true_source": snap.true_source,
            "predicted_source": pred,
            "correct": int(pred == snap.true_source) if pred is not None else 0,
            "graph_distance": baseline_error(snap, after_sample=False),
            "observed_count": sum(v is not None for v in snap.readings.values()),
            "initial_sensors": trial.initial_sensors,
            "noise_std": trial.noise_std,
            "missing_rate": trial.missing_rate,
            "source_intensity": trial.source_intensity,
            "decay_rate": trial.decay_rate,
        })
    return output
