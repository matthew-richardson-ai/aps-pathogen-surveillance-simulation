"""Simple source-localization baselines for APS."""

from __future__ import annotations

from collections.abc import Mapping


def strongest_sensor(readings: Mapping[str, float | None]) -> str:
    """Maximum-Sensor-Reading (Argmax) Baseline.

    A deterministic heuristic that returns the zone associated with the
    maximum observed sensor value. This algorithm has no training step.
    """
    available = {
        zone: value
        for zone, value in readings.items()
        if value is not None
    }

    if not available:
        raise ValueError(
            "Cannot localize source when every sensor reading is missing"
        )

    return max(available, key=available.get)
