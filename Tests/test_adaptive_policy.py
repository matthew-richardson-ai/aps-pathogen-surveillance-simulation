"""Tests for APS information-gain sampling and matched scenario comparisons."""
import numpy as np
import pandas as pd

from Adaptive.sampler import choose_next_zone, source_prior
from OB.adaptive_experiments import compare_sampling_policies, summarize_comparison
from OB.live_engine import Scenario
from Localization.ml_pipeline import ZONES, train_models
from Simulation.facility import build_facility
from Simulation.outbreak import expected_signals
from Simulation.sensors import observe_signals


def _test_dataset(tmp_path):
    graph = build_facility()
    records = []
    for i in range(200):
        source = ZONES[i % len(ZONES)]
        readings = observe_signals(
            expected_signals(graph, source), np.random.default_rng(i + 100),
            noise_std=0.08, missing_rate=0.15,
        )
        row = {
            "run_id": f"run_{i}",
            "group_id": f"group_{i}",
            "split": "train" if i < 160 else "test",
            "scenario_type": "normal",
            "true_source_node": source,
        }
        for zone in ZONES:
            row[f"{zone}_reading"] = readings[zone]
            row[f"{zone}_missing"] = int(readings[zone] is None)
        records.append(row)
    path = tmp_path / "training.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_only_unobserved_zones_can_be_chosen():
    readings = {zone: 0.5 for zone in ZONES}
    readings["zone_2"] = None
    readings["zone_8"] = None
    choice = choose_next_zone(
        build_facility(), readings, {zone: 1 / 8 for zone in ZONES},
    )
    assert choice.zone in ("zone_2", "zone_8")
    assert set(choice.candidate_gains) == {"zone_2", "zone_8"}
    assert 0 <= choice.expected_gain_bits <= 3


def test_choice_is_deterministic_and_uses_no_hidden_source():
    readings = dict.fromkeys(ZONES)
    prior = {zone: 1 / 8 for zone in ZONES}
    a = choose_next_zone(build_facility(), readings, prior)
    b = choose_next_zone(build_facility(), dict(reversed(list(readings.items()))), prior)
    assert a == b


def test_no_candidates_returns_none():
    readings = {zone: 0.1 for zone in ZONES}
    assert choose_next_zone(
        build_facility(), readings, {zone: 1 / 8 for zone in ZONES}
    ) is None


def test_paired_sampling_comparison_is_reproducible(tmp_path):
    artifacts = train_models(_test_dataset(tmp_path))
    params = Scenario(seed=900, initial_sensors=3, missing_rate=0.25)
    first = compare_sampling_policies(artifacts, params, trials=6)
    second = compare_sampling_policies(artifacts, params, trials=6)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 6 * 2 * 3
    assert first.groupby(["run_id", "model"]).strategy.nunique().eq(3).all()
    assert first[first.strategy == "static"].extra_samples.eq(0).all()
    assert first[first.strategy != "static"].extra_samples.eq(1).all()
    assert first.groupby("run_id").true_source.nunique().eq(1).all()
    assert not summarize_comparison(first).empty


def test_uniform_prior_when_every_sensor_missing(tmp_path):
    artifacts = train_models(_test_dataset(tmp_path))
    prior = source_prior(
        artifacts.models["Random Forest"], dict.fromkeys(ZONES)
    )
    assert all(abs(value - 1 / 8) < 1e-9 for value in prior.values())
