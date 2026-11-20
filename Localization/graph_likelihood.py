"""Source localization with a Bayesian graph model and *unknown* signal settings.

NO ORACLE PARAMETERS: Inference accepts only observed sensor readings, the
canonical APS graph, and the identity of the separately measured follow-up
zone. The simulator's actual source, signal intensity, decay and primary noise
are never passed to the predictor.

The model assumes an exponential graph-distance signal family and averages
(marginalizes) over a fixed, publicly declared grid of plausible nuisance
parameters. These priors are modeling assumptions, NOT learned ground truth,
and accuracy is not guaranteed on sparse/noisy observations.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import math

import networkx as nx
import numpy as np
from scipy.special import log_ndtr, logsumexp

from Simulation.facility import build_facility

GRAPH_MODEL_NAME = "Graph-Aware Bayesian Inference"
ZONES = tuple(f"zone_{i}" for i in range(1, 9))

# Prespecified parameter ranges are independent of any particular scenario.
# They intentionally cover normal, noisy, weak-signal and combined-stress cases.
INTENSITIES = (0.30, 0.50, 0.75, 1.00)
DECAYS = (0.35, 0.65, 0.95, 1.25)
NOISE_LEVELS = (0.06, 0.16, 0.30)
FOLLOWUP_NOISE_SD = 0.04


@lru_cache(maxsize=1)
def _hypotheses() -> tuple[np.ndarray, np.ndarray]:
    """Return signal means and primary noise scales for eight source hypotheses.

    Each source receives exactly the same number and prior weight of plausible
    physical settings. Array shape: (8 sources, 48 nuisance settings, 8 rooms).
    The source truth and per-event simulator settings are never looked up.
    """
    graph = build_facility()
    distances = np.asarray([
        [nx.shortest_path_length(graph, source, sensor) for sensor in ZONES]
        for source in ZONES
    ], dtype=float)
    configs = [(intensity, decay, noise)
               for intensity in INTENSITIES
               for decay in DECAYS
               for noise in NOISE_LEVELS]
    means = np.stack([
        intensity * np.exp(-decay * distances)
        for intensity, decay, _ in configs
    ], axis=1)
    noise = np.asarray([config[2] for config in configs], dtype=float)
    return means, noise


def _joint_log_weights(
    readings: Mapping[str, float | None],
    requested_zone: str | None = None,
) -> np.ndarray:
    if set(readings) != set(ZONES):
        raise ValueError("Graph inference requires exactly eight APS zones")
    if requested_zone is not None and requested_zone not in ZONES:
        raise ValueError("Invalid follow-up zone")
    for reading in readings.values():
        if reading is not None and (
            not math.isfinite(float(reading)) or not 0 <= float(reading) <= 1
        ):
            raise ValueError("Sensor readings must be within 0..1 or missing")

    means, sigma_primary = _hypotheses()
    n_settings = means.shape[1]
    log_weights = np.full(
        (len(ZONES), n_settings),
        -math.log(len(ZONES) * n_settings),
        dtype=float,
    )
    for index, zone in enumerate(ZONES):
        reading = readings[zone]
        if reading is None:
            continue
        sigma = (
            np.full(n_settings, FOLLOWUP_NOISE_SD)
            if zone == requested_zone else sigma_primary
        )
        signal_means = means[:, :, index]
        scaled = (float(reading) - signal_means) / sigma[None, :]
        # The sensor clips values to [0, 1]. Endpoints are censored events,
        # not ordinary Gaussian point densities.
        if float(reading) == 0:
            likelihood = log_ndtr(-signal_means / sigma[None, :])
        elif float(reading) == 1:
            likelihood = log_ndtr((signal_means - 1.0) / sigma[None, :])
        else:
            likelihood = (
                -0.5 * scaled**2 - np.log(sigma[None, :])
                - 0.5 * math.log(2.0 * math.pi)
            )
        log_weights += likelihood
    return log_weights - logsumexp(log_weights)


def source_posterior(
    readings: Mapping[str, float | None],
    *,
    requested_zone: str | None = None,
) -> dict[str, float]:
    """Eight source probabilities after integrating out unknown model settings.

    These are conditional on the stated model family and discrete priors;
    they are NOT calibrated real-world confidence estimates.
    """
    normalized = _joint_log_weights(readings, requested_zone)
    by_source = np.exp(logsumexp(normalized, axis=1))
    return dict(zip(ZONES, map(float, by_source)))


def graph_prediction(
    readings: Mapping[str, float | None],
    *,
    requested_zone: str | None = None,
) -> tuple[str | None, dict[str, float]]:
    """Return the most probable source from readings alone; abstain if missing.

    There is deliberately no source_intensity, decay_rate or noise_std argument,
    preventing per-event true settings from leaking into this estimator.
    """
    probabilities = source_posterior(readings, requested_zone=requested_zone)
    if all(value is None for value in readings.values()):
        return None, probabilities
    return max(ZONES, key=probabilities.get), probabilities
