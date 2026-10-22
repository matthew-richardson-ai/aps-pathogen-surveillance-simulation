"""Simple source-localization baselines for APS."""

from __future__ import annotations

from collections.abc import Mapping


def strongest_sensor(readings: Mapping[str, float | None]) -> str:
    """Predict the source as the zone with the strongest available reading."""
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
