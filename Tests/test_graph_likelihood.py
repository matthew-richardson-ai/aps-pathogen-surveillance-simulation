"""Regression tests for APS graph inference with *unknown* scenario parameters."""
import inspect
import math

import numpy as np
import pytest

from Localization.graph_likelihood import (
    ZONES, graph_prediction, source_posterior, GRAPH_MODEL_NAME,
    INTENSITIES, DECAYS, NOISE_LEVELS,
)
from Simulation.facility import build_facility
from Simulation.outbreak import expected_signals


def scenario_readings(source="zone_3"):
    return dict(expected_signals(
        build_facility(), source, source_intensity=0.45, decay_rate=0.95
    ))


def test_no_oracle_configuration_inputs_are_accepted():
    signature = inspect.signature(graph_prediction)
    assert "source_intensity" not in signature.parameters
    assert "decay_rate" not in signature.parameters
    assert "noise_std" not in signature.parameters
    assert "Known Params" not in GRAPH_MODEL_NAME
    with pytest.raises(TypeError):
        graph_prediction(scenario_readings(), source_intensity=0.45)


def test_predicts_source_without_observing_its_own_sensor():
    readings = scenario_readings()
    readings["zone_3"] = None
    predicted, probabilities = graph_prediction(readings)
    assert predicted == "zone_3"
    assert sum(probabilities.values()) == pytest.approx(1.0)
    assert set(probabilities) == set(ZONES)


def test_marginalizes_over_settings_from_public_grid():
    assert len(INTENSITIES) > 1 and len(DECAYS) > 1 and len(NOISE_LEVELS) > 1
    p = source_posterior(scenario_readings())
    assert len(p) == 8
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
    assert sum(p.values()) == pytest.approx(1.0)


def test_censored_readings_are_finite():
    readings = dict.fromkeys(ZONES)
    readings["zone_1"] = 0.0
    readings["zone_7"] = 1.0
    result, probabilities = graph_prediction(readings)
    assert result in ZONES
    assert np.isfinite(list(probabilities.values())).all()
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_missing_inputs_abstain_and_are_uniform():
    result, probabilities = graph_prediction(dict.fromkeys(ZONES))
    assert result is None
    assert all(p == pytest.approx(1 / 8) for p in probabilities.values())


def test_followup_measurement_uses_separate_instrument_noise():
    readings = dict.fromkeys(ZONES)
    readings["zone_2"] = 0.4
    p1 = source_posterior(readings)
    p2 = source_posterior(readings, requested_zone="zone_2")
    assert p1 != p2
    assert sum(p2.values()) == pytest.approx(1.0)


def test_bad_observations_are_rejected():
    bad = dict.fromkeys(ZONES)
    bad["zone_2"] = 1.2
    with pytest.raises(ValueError, match="0..1"):
        graph_prediction(bad)
    bad["zone_2"] = None
    with pytest.raises(ValueError, match="follow-up"):
        graph_prediction(bad, requested_zone="zone_9")
