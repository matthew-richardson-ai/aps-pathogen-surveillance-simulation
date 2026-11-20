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
        "Strongest sensor",
        "Logistic Regression (Original)", "Random Forest (Original)",
        "Logistic Regression", "Random Forest",
    }
    assert len(artifacts.test_predictions) == 200
    assert artifacts.augmented_train_count == 640
    assert set(artifacts.reference_models) == {"Logistic Regression", "Random Forest"}
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


def test_model_based_scores_require_known_simulator_settings(tmp_path):
    """Only data rows containing scenario parameters get graph-likelihood scores."""
    from Localization.graph_likelihood import GRAPH_MODEL_NAME
    path = create_dataset(tmp_path)
    original = train_models(path, augmentation_copies=0)
    assert GRAPH_MODEL_NAME not in set(original.scores.model)
    assert original.augmented_train_count == original.train_count

    frame = pd.read_csv(path)
    frame["noise_std"] = 0.08
    frame["source_intensity"] = 1.0
    frame["decay_rate"] = 0.65
    frame.to_csv(path, index=False)
    with_graph = train_models(path, augmentation_copies=0)
    assert GRAPH_MODEL_NAME in set(with_graph.scores.model)
    assert len(with_graph.test_predictions) == 240


def test_training_augmentation_never_uses_holdout_to_fit(tmp_path):
    """An arbitrary test label edit cannot change fitted robust coefficients."""
    path = create_dataset(tmp_path)
    original = train_models(path, augmentation_copies=1)
    frame = pd.read_csv(path)
    frame.loc[frame["split"] == "test", "true_source_node"] = "zone_8"
    frame.to_csv(path, index=False)
    altered = train_models(path, augmentation_copies=1)
    assert np.allclose(
        original.models["Logistic Regression"].named_steps["model"].coef_,
        altered.models["Logistic Regression"].named_steps["model"].coef_,
    )
