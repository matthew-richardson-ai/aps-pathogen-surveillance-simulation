"""Unit tests for the APS interactive dashboard's simulation engine."""
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "OB"))
from live_engine import Scenario, simulate, sample_one_more, predict_baseline, baseline_error, batch_baseline


def test_seed_reproducibility():
    config = Scenario(seed=55, initial_sensors=4, noise_std=0.15, missing_rate=0.30)
    a, b = simulate(config), simulate(config)
    assert (a.true_source, a.readings, a.initial_prediction) == (b.true_source, b.readings, b.initial_prediction)


def test_respects_initial_observation_budget():
    snapshot = simulate(Scenario(seed=42, initial_sensors=3, missing_rate=0))
    assert len(snapshot.initially_selected) == 3
    assert sum(value is not None for value in snapshot.readings.values()) == 3
    assert set(snapshot.readings) == {f"zone_{i}" for i in range(1, 9)}


def test_manual_sample_adds_exactly_one_reading():
    snapshot = simulate(Scenario(seed=23, initial_sensors=4, missing_rate=0))
    target = next(z for z, value in snapshot.readings.items() if value is None)
    after = sample_one_more(snapshot, target)
    assert after.requested_zone == target
    assert after.readings[target] is not None
    assert sum(value is not None for value in after.readings.values()) == 5
    assert after.initial_prediction == snapshot.initial_prediction
    assert all(after.readings[z] == v for z, v in snapshot.readings.items() if z != target)
    with pytest.raises(ValueError, match="Only one"):
        sample_one_more(after, target)


def test_rejects_already_observed_zone():
    snapshot = simulate(Scenario(seed=20, initial_sensors=6, missing_rate=0))
    target = next(z for z, value in snapshot.readings.items() if value is not None)
    with pytest.raises(ValueError, match="without a reading"):
        sample_one_more(snapshot, target)


def test_all_missing_means_abstain_not_crash():
    snapshot = simulate(Scenario(seed=3, initial_sensors=2, missing_rate=1))
    assert predict_baseline(snapshot.readings) is None
    assert snapshot.initial_prediction is None
    assert baseline_error(snapshot) is None
    trials = batch_baseline(Scenario(seed=0, initial_sensors=2, missing_rate=1), 3)
    assert all(row["correct"] == 0 and row["graph_distance"] is None for row in trials)


def test_batch_evaluation_uses_separate_seeded_scenarios():
    trials = batch_baseline(Scenario(seed=100, source="zone_2", initial_sensors=5, missing_rate=0), 20)
    assert len(trials) == 20
    assert [row["seed"] for row in trials] == list(range(100, 120))
    assert all(row["correct"] == int(row["predicted_source"] == row["true_source"]) for row in trials)
