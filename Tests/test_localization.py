import pytest

from Localization.baselines import strongest_sensor


def test_strongest_sensor_baseline():
    readings = {
        "zone_1": 0.2,
        "zone_2": 0.8,
        "zone_3": 0.4,
    }

    assert strongest_sensor(readings) == "zone_2"


def test_baseline_ignores_missing_readings():
    readings = {
        "zone_1": None,
        "zone_2": 0.7,
        "zone_3": 0.3,
    }

    assert strongest_sensor(readings) == "zone_2"


def test_baseline_fails_cleanly_when_all_readings_missing():
    with pytest.raises(ValueError):
        strongest_sensor({
            "zone_1": None,
            "zone_2": None,
        })
