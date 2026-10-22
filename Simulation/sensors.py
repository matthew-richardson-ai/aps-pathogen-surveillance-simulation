"""Sensor observation generation for the APS simulator."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def observe_signals(
    expected: Mapping[str, float],
    rng: np.random.Generator,
    noise_std: float = 0.08,
    missing_rate: float = 0.10,
    min_reading: float = 0.0,
    max_reading: float = 1.0,
) -> dict[str, float | None]:
    """Add measurement noise and missingness to ideal facility signals."""
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be between 0 and 1")

    readings: dict[str, float | None] = {}
    for zone, signal in expected.items():
        if rng.random() < missing_rate:
            readings[zone] = None
            continue

        noisy = signal + rng.normal(0.0, noise_std)
        readings[zone] = float(np.clip(noisy, min_reading, max_reading))

    return readings
