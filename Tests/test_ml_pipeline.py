"""Small deterministic tests of the APS ML training and inference integration."""
import numpy as np
import pandas as pd
import pytest

from Localization.ml_pipeline import ZONES, reading_features, train_models, predict_models
from Simulation.facility import build_facility
from Simulation.outbreak import expected_signals
from Simulation.sensors import observe_signals


def create_dataset(tmp_path):
    graph = build_facility()
    records = []
    for index in range(200):
        truth = ZONES[index % len(ZONES)]
        readings = observe_signals(
            expected_signals(graph, truth),
            np.random.default_rng(index + 100),
            noise_std=0.08, missing_rate=0.15,
        )
        record = {
            "run_id": f"run_{index}",
            "group_id": f"group_{index}",
            "split": "train" if index < 160 else "test",
            "scenario_type": "normal",
            "true_source_node": truth,
        }
        for zone in ZONES:
            record[f"{zone}_reading"] = readings[zone]
            record[f"{zone}_missing"] = int(readings[zone] is None)
        records.append(record)
    path = tmp_path / "dataset.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def test_feature_schema_preserves_missingness():
    readings = {zone: 0.2 for zone in ZONES}
    readings["zone_4"] = None
    features = reading_features(readings)
    assert features.shape == (1, 16)
    assert np.isnan(features.loc[0, "zone_4_reading"])
    assert features.loc[0, "zone_4_missing"] == 1
    assert features.loc[0, "zone_1_missing"] == 0


def test_train_and_predict(tmp_path):
    artifacts = train_models(create_dataset(tmp_path))
    assert artifacts.train_count == 160
    assert artifacts.test_count == 40
    assert set(artifacts.scores.model) == {
        "Strongest sensor", "Logistic Regression", "Random Forest",
    }
    assert len(artifacts.test_predictions) == 120
    assert artifacts.scores.trials.eq(40).all()
    live = {zone: 0.3 for zone in ZONES}
    live["zone_4"] = None
    predictions = predict_models(artifacts, live)
    assert set(predictions.predicted_source).issubset(set(ZONES))
    assert predictions.max_class_probability.between(0.0, 1.0).all()
    no_readings = predict_models(artifacts, dict.fromkeys(ZONES))
    assert no_readings.predicted_source.isna().all()


def test_reject_train_test_group_leakage(tmp_path):
    path = create_dataset(tmp_path)
    frame = pd.read_csv(path)
    frame.loc[160, "group_id"] = frame.loc[0, "group_id"]
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Group leakage"):
        train_models(path)
