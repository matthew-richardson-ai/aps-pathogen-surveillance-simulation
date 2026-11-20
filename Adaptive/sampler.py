"""One-step model-assisted information-gain sampling for APS.

The selection policy receives only the observed readings, learned model
probabilities and known simulation parameters. It never sees the true source.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.special import ndtr

from Localization.ml_pipeline import ZONES, reading_features
from Simulation.outbreak import expected_signals


@dataclass(frozen=True)
class SampleChoice:
    zone: str
    expected_gain_bits: float
    candidate_gains: dict[str, float]


def source_prior(estimator, readings: Mapping[str, float | None]) -> dict[str, float]:
    """Convert ML probabilities into a smoothed prior over possible sources.

    Random Forest probabilities can be zero for some source classes. A small
    uniform component allows an actual observation to correct that estimate.
    These probabilities are NOT calibrated confidence scores.
    """
    if set(readings) != set(ZONES):
        raise ValueError("Expected exactly eight APS zones")
    if all(value is None or bool(np.isnan(value)) for value in readings.values()):
        probabilities = np.full(len(ZONES), 1 / len(ZONES))
    else:
        model_probs = estimator.predict_proba(reading_features(readings))[0]
        labels = {str(name): float(prob) for name, prob in zip(estimator.classes_, model_probs)}
        probabilities = np.asarray([labels.get(zone, 0.0) for zone in ZONES], dtype=float)
        if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0) or probabilities.sum() <= 0:
            raise ValueError("Model returned invalid probabilities")
        probabilities /= probabilities.sum()
        probabilities = 0.96 * probabilities + 0.04 / len(ZONES)
    return dict(zip(ZONES, map(float, probabilities)))


def _entropy(probs: np.ndarray) -> float:
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def information_gains(
    graph: nx.Graph,
    readings: Mapping[str, float | None],
    prior: Mapping[str, float],
    source_intensity: float = 1.0,
    decay_rate: float = 0.65,
    sample_noise_std: float = 0.04,
    bins: int = 50,
) -> dict[str, float]:
    """Estimate expected information gain for each currently unobserved zone.

    The likelihood uses APS's exponential graph-distance signal model,
    the known scenario settings and a noisy manual sampling instrument.
    The true source is not used. Measurement outcomes are integrated over
    bins in [0, 1], with clipping probability in the edge bins.
    """
    if set(readings) != set(ZONES) or set(prior) != set(ZONES):
        raise ValueError("Readings and prior must cover all eight APS zones")
    if sample_noise_std <= 0 or bins < 5:
        raise ValueError("Sample noise must be positive and bins >= 5")
    if source_intensity <= 0 or decay_rate < 0:
        raise ValueError("Invalid spatial signal parameters")
    weights = np.asarray([prior[zone] for zone in ZONES], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid source probability distribution")
    weights /= weights.sum()
    entropy_before = _entropy(weights)
    edges = np.r_[-np.inf, np.linspace(0.0, 1.0, bins + 1)[1:-1], np.inf]
    expected_by_source = {
        source: expected_signals(graph, source, source_intensity, decay_rate)
        for source in ZONES
    }
    gains = {}
    for zone in ZONES:
        value = readings[zone]
        if value is not None and not bool(np.isnan(value)):
            continue
        means = np.asarray([expected_by_source[source][zone] for source in ZONES])
        # Each likelihood row corresponds to one possible hidden source.
        likelihood = np.diff(
            ndtr((edges[None, :] - means[:, None]) / sample_noise_std), axis=1
        )
        outcome_probabilities = weights @ likelihood
        joint = weights[:, None] * likelihood
        posteriors = np.divide(
            joint, outcome_probabilities[None, :],
            out=np.zeros_like(joint),
            where=outcome_probabilities[None, :] > 0,
        )
        expected_entropy = sum(
            outcome_probabilities[j] * _entropy(posteriors[:, j])
            for j in range(len(outcome_probabilities))
        )
        gains[zone] = max(0.0, float(entropy_before - expected_entropy))
    return gains


def choose_next_zone(
    graph: nx.Graph,
    readings: Mapping[str, float | None],
    prior: Mapping[str, float],
    source_intensity: float = 1.0,
    decay_rate: float = 0.65,
    sample_noise_std: float = 0.04,
) -> SampleChoice | None:
    """Choose the unobserved zone with the largest expected information gain."""
    gains = information_gains(
        graph, readings, prior, source_intensity, decay_rate, sample_noise_std,
    )
    if not gains:
        return None
    # Stable tie-breaking for reproducibility.
    selected = max(sorted(gains), key=lambda zone: gains[zone])
    return SampleChoice(selected, gains[selected], gains)


def choose_next_zone_blind(
    graph: nx.Graph,
    readings: Mapping[str, float | None],
    prior: Mapping[str, float],
    *,
    sample_noise_std: float = 0.04,
    bins: int = 50,
) -> SampleChoice | None:
    """Choose a measurement without accessing actual simulator parameters.

    For each candidate source and as-yet-unobserved sensor, this policy
    integrates hypothetical measurements over *fixed* plausible intensity
    and graph-decay values. It conditions on the input source probabilities,
    not on the true hidden source or actual event's signal/decay/noise settings.

    This is a discretized expected-entropy policy under an assumed model
    family. The estimated gain is not guaranteed to match realized gain.
    """
    from Localization.graph_likelihood import INTENSITIES, DECAYS

    if set(readings) != set(ZONES) or set(prior) != set(ZONES):
        raise ValueError("Readings and prior must cover the eight APS zones")
    if sample_noise_std <= 0 or bins < 5:
        raise ValueError("Invalid observation noise or histogram bin count")

    weights = np.asarray([prior[zone] for zone in ZONES], dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid source probabilities")
    weights /= weights.sum()
    starting_entropy = _entropy(weights)

    # Binning includes clipped endpoints at zero and one.
    edges = np.r_[-np.inf, np.linspace(0.0, 1.0, bins + 1)[1:-1], np.inf]
    gains = {}
    for zone in ZONES:
        if readings[zone] is not None:
            continue
        # Integrate over source strength and decay, not an event-specific
        # hidden parameter value. Shape: sources x nuisance settings x bins.
        means = np.asarray([
            [
                intensity * np.exp(
                    -decay * nx.shortest_path_length(graph, source, zone)
                )
                for intensity in INTENSITIES for decay in DECAYS
            ]
            for source in ZONES
        ], dtype=float)
        likelihood = np.diff(
            ndtr((edges[None, None, :] - means[:, :, None]) / sample_noise_std),
            axis=-1,
        ).mean(axis=1)
        outcomes = weights @ likelihood
        joint = weights[:, None] * likelihood
        posterior = np.divide(
            joint, outcomes[None, :], out=np.zeros_like(joint),
            where=outcomes[None, :] > 0,
        )
        entropy_after = sum(
            outcomes[j] * _entropy(posterior[:, j])
            for j in range(len(outcomes))
        )
        gains[zone] = max(0.0, float(starting_entropy - entropy_after))
    if not gains:
        return None
    selected = max(sorted(gains), key=gains.get)
    return SampleChoice(selected, gains[selected], gains)
