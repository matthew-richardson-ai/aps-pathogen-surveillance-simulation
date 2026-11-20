"""Graph-aware source localization using the APS observation model.

Unlike Argmax, this evaluates *every* possible source, even unobserved rooms.
It is a model-based likelihood method, not a classifier trained on Justin's data.

IMPORTANT FAIRNESS NOTE: This estimator uses the scenario's source intensity,
spatial decay and sensor noise settings. Those are *known simulation parameters*
rather than readings. It therefore has more prior physical-model information
than Logistic Regression or Random Forest. Report this assumption explicitly.
The true source and ideal signal array are never inputs to this estimator.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np
from scipy.special import log_ndtr, logsumexp

from Simulation.facility import build_facility
from Simulation.outbreak import expected_signals

GRAPH_MODEL_NAME = "Graph-Aware Likelihood (Known Params)"
ZONES = tuple(f"zone_{i}" for i in range(1, 9))


def source_posterior(
    readings: Mapping[str, float | None],
    *,
    source_intensity: float,
    decay_rate: float,
    noise_std: float,
    requested_zone: str | None = None,
    sample_noise_std: float = 0.04,
) -> dict[str, float]:
    """Return eight normalized source probabilities under an assumed model.

    Uses independent Gaussian observation noise and treats readings clipped at
    zero and one as *censored* rather than interior Gaussian measurements.
    Missing measurements contribute no evidence. The extra measurement has a
    separate instrument noise level (0.04 in the current APS simulator).

    With no observations, returns a uniform prior. Call graph_prediction() for
    the abstaining decision-rule version of this estimate.
    """
    if set(readings) != set(ZONES):
        raise ValueError("Graph likelihood requires readings for eight APS zones")
    if source_intensity <= 0 or decay_rate < 0 or noise_std < 0 or sample_noise_std <= 0:
        raise ValueError("Invalid graph-likelihood signal or noise parameters")
    if requested_zone is not None and requested_zone not in ZONES:
        raise ValueError("Additional measurement zone is not in the facility")
    if any(v is not None and (not math.isfinite(float(v)) or not 0 <= float(v) <= 1)
           for v in readings.values()):
        raise ValueError("Sensor readings must lie within 0..1 or be missing")

    graph = build_facility()
    means = {
        zone: expected_signals(
            graph, zone, source_intensity=source_intensity, decay_rate=decay_rate
        )
        for zone in ZONES
    }

    # Flat source prior: we do not secretly favor any room or access truth.
    log_scores = np.full(len(ZONES), -math.log(len(ZONES)), dtype=float)
    for zone, reading in readings.items():
        if reading is None:
            continue
        # Nonzero floor is only numerical protection when the simulator uses
        # exactly zero noise. Both the primary and follow-up instruments are
        # otherwise compared against their respective settings.
        sigma = max(1e-4, sample_noise_std if zone == requested_zone else noise_std)
        expected = np.array([means[source][zone] for source in ZONES])
        value = float(reading)
        if value <= 0.0:
            likelihood = log_ndtr(-expected / sigma)
        elif value >= 1.0:
            likelihood = log_ndtr((expected - 1.0) / sigma)
        else:
            likelihood = (
                -0.5 * ((value - expected) / sigma) ** 2
                -math.log(sigma) -0.5 * math.log(2.0 * math.pi)
            )
        log_scores += likelihood

    probs = np.exp(log_scores - logsumexp(log_scores))
    return dict(zip(ZONES, probs.astype(float)))


def graph_prediction(
    readings: Mapping[str, float | None],
    *,
    source_intensity: float,
    decay_rate: float,
    noise_std: float,
    requested_zone: str | None = None,
    sample_noise_std: float = 0.04,
) -> tuple[str | None, dict[str, float]]:
    """Return maximum-likelihood source and the full posterior-like score map.

    This is an estimated probability conditional on a *correctly specified*
    spatial observation model. It must not be presented as real-world certainty.
    The model abstains when all eight readings are absent.
    """
    posterior = source_posterior(
        readings,
        source_intensity=source_intensity,
        decay_rate=decay_rate,
        noise_std=noise_std,
        requested_zone=requested_zone,
        sample_noise_std=sample_noise_std,
    )
    if all(value is None for value in readings.values()):
        return None, posterior
    return max(ZONES, key=posterior.get), posterior
