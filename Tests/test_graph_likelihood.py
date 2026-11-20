"""Tests for parameter-aware graph likelihood in the APS spatial simulator."""
import math

import numpy as np
import pytest

from Localization.graph_likelihood import source_posterior, graph_prediction, ZONES
from Simulation.facility import build_facility
from Simulation.outbreak import expected_signals


def scenario_readings(source="zone_3", intensity=0.45, decay=0.95):
    ideal = expected_signals(build_facility(), source, intensity, decay)
    return dict(ideal)


def test_model_identifies_known_source_even_without_its_own_sensor():
    readings = scenario_readings()
    readings["zone_3"] = None
    prediction, posterior = graph_prediction(
        readings, source_intensity=0.45, decay_rate=0.95, noise_std=0.03,
    )
    assert prediction == "zone_3"
    assert set(posterior) == set(ZONES)
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in posterior.values())


def test_censored_sensor_values_remain_finite():
    readings = dict.fromkeys(ZONES)
    readings["zone_1"] = 0.0
    readings["zone_7"] = 1.0
    prediction, posterior = graph_prediction(
        readings, source_intensity=0.6, decay_rate=0.9, noise_std=0.23,
    )
    assert prediction in ZONES
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert all(np.isfinite(list(posterior.values())))


def test_all_missing_abstains_and_returns_uniform_prior():
    readings = dict.fromkeys(ZONES)
    guess, posterior = graph_prediction(
        readings, source_intensity=1.0, decay_rate=0.65, noise_std=0.2,
    )
    assert guess is None
    assert all(prob == pytest.approx(1 / 8) for prob in posterior.values())


def test_additional_measurement_uses_its_own_noise_model():
    readings = dict.fromkeys(ZONES)
    readings["zone_2"] = 0.4
    regular = source_posterior(
        readings, source_intensity=1.0, decay_rate=0.65, noise_std=0.25
    )
    precise = source_posterior(
        readings, source_intensity=1.0, decay_rate=0.65, noise_std=0.25,
        requested_zone="zone_2", sample_noise_std=0.04,
    )
    assert regular != precise
    assert sum(precise.values()) == pytest.approx(1.0)


def test_rejects_invalid_readings_and_parameters():
    readings = dict.fromkeys(ZONES)
    readings["zone_2"] = 2.0
    with pytest.raises(ValueError, match="0..1"):
        graph_prediction(readings, source_intensity=1, decay_rate=.65, noise_std=.1)
    readings["zone_2"] = None
    with pytest.raises(ValueError, match="noise"):
        graph_prediction(readings, source_intensity=1, decay_rate=.65, noise_std=-.1)
