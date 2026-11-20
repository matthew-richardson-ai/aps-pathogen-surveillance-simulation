"""Train APS source-localization models using Justin's split synthetic data.

Only the eight sensor readings and missingness indicators are model inputs.
Source labels, scenario metadata and optional extra samples are excluded.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Localization.baselines import strongest_sensor
from Localization.robust_training import augment_sparse_training
from Localization.graph_likelihood import GRAPH_MODEL_NAME, graph_prediction
from Simulation.facility import build_facility

ZONES = tuple(f"zone_{i}" for i in range(1, 9))
FEATURE_COLUMNS = tuple(
    field for zone in ZONES
    for field in (f"{zone}_reading", f"{zone}_missing")
)


@dataclass
class ModelArtifacts:
    models: dict[str, Pipeline]
    source_path: str
    train_count: int
    test_count: int
    scores: pd.DataFrame
    test_predictions: pd.DataFrame
    augmented_train_count: int = 0
    reference_models: dict[str, Pipeline] | None = None


def default_dataset_path() -> Path:
    """Locate Justin's data in either the current or legacy branch layout."""
    root = Path(__file__).resolve().parents[1]
    directories = (
        root / "Team" / "Justin" / "Data",
        root / "Justin" / "Data",
    )
    for directory in directories:
        candidates = sorted(directory.glob("outbreak_dataset_*.csv"))
        if candidates:
            return candidates[-1]
    locations = ", ".join(str(path.relative_to(root)) for path in directories)
    raise FileNotFoundError(
        f"No APS training dataset found. Checked {locations}. "
        "The repository's Justin dataset must be present before training."
    )


def reading_features(readings: Mapping[str, float | None]) -> pd.DataFrame:
    if set(readings) != set(ZONES):
        raise ValueError("Readings must contain exactly zone_1 through zone_8")
    row = {}
    for zone in ZONES:
        value = readings[zone]
        absent = value is None or bool(pd.isna(value))
        row[f"{zone}_reading"] = np.nan if absent else float(value)
        row[f"{zone}_missing"] = int(absent)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")


def _baseline_predict(frame: pd.DataFrame) -> list[str | None]:
    guesses = []
    for _, row in frame.iterrows():
        present = {
            zone: float(row[f"{zone}_reading"])
            for zone in ZONES
            if pd.notna(row[f"{zone}_reading"])
        }
        guesses.append(strongest_sensor(present) if present else None)
    return guesses


def _new_classifiers() -> dict[str, Pipeline]:
    """Independent pipelines; every experiment receives fresh estimators."""
    return {
        "Logistic Regression": Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=1200, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", RandomForestClassifier(
                n_estimators=120, random_state=42, n_jobs=-1,
            )),
        ]),
    }


def _graph_test_predictions(testing: pd.DataFrame) -> list[str | None] | None:
    """Graph likelihood can be scored only when held-out metadata is present.

    This uses the true *simulation parameters*, not the true source label.
    Crucially, it has more information than the data-only ML classifiers.
    The distinction is annotated throughout the presentation.
    """
    metadata = {"source_intensity", "decay_rate", "noise_std"}
    if not metadata.issubset(testing.columns):
        return None
    predictions = []
    for _, record in testing.iterrows():
        observed = {
            zone: None if pd.isna(record[f"{zone}_reading"])
            else float(record[f"{zone}_reading"])
            for zone in ZONES
        }
        guess, _ = graph_prediction(
            observed,
            source_intensity=float(record["source_intensity"]),
            decay_rate=float(record["decay_rate"]),
            noise_std=float(record["noise_std"]),
        )
        predictions.append(guess)
    return predictions


def train_models(
    dataset_path: str | Path | None = None,
    *,
    augmentation_copies: int = 3,
) -> ModelArtifacts:
    """Fit robust ML models using training rows only; preserve a true control.

    The reference classifiers are trained on the untouched training split.
    The robust models receive those rows plus sparse, noise-perturbed *training*
    views. Both are scored on the same unchanged held-out test partition.
    Validation rows, if supplied, never become training examples.

    The UI keeps the recognizable names "Logistic Regression" and
    "Random Forest" for robust models, with "(Original)" references shown
    side by side in held-out evaluation. No model is selected based on test
    performance, and no performance improvement is presumed.
    """
    path = Path(dataset_path) if dataset_path is not None else default_dataset_path()
    frame = pd.read_csv(path)
    required = set(FEATURE_COLUMNS) | {"true_source_node", "split", "run_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    if frame["run_id"].duplicated().any():
        raise ValueError("Each run_id must identify a unique simulated scenario")
    bad_sources = set(frame["true_source_node"].dropna()) - set(ZONES)
    if bad_sources or frame["true_source_node"].isna().any():
        raise ValueError("Dataset has missing or invalid source labels")
    training = frame[frame["split"] == "train"].copy()
    testing = frame[frame["split"] == "test"].copy()
    if training.empty or testing.empty:
        raise ValueError("Dataset must have nonempty train and test partitions")
    if "group_id" in frame:
        if set(training["group_id"]) & set(testing["group_id"]):
            raise ValueError("Group leakage between train and test partitions")
        validation = frame[frame["split"] == "validation"]
        if set(training["group_id"]) & set(validation["group_id"]):
            raise ValueError("Group leakage between train and validation partitions")

    X_train, X_test = _features(training), _features(testing)
    y_train = training["true_source_node"].astype(str)

    # Reference controls use exactly the previous dataset representation.
    reference_models = _new_classifiers()
    for estimator in reference_models.values():
        estimator.fit(X_train, y_train)

    augmented_x, augmented_y = augment_sparse_training(
        X_train, y_train, copies=augmentation_copies, seed=4240
    )
    models = _new_classifiers()
    for estimator in models.values():
        estimator.fit(augmented_x, augmented_y)

    graph = build_facility()
    original = [(f"{name} (Original)", estimator.predict(X_test).tolist())
                for name, estimator in reference_models.items()]
    robust = [(name, estimator.predict(X_test).tolist())
              for name, estimator in models.items()]
    predictions = [("Strongest sensor", _baseline_predict(testing)), *original, *robust]
    graph_guesses = _graph_test_predictions(testing)
    if graph_guesses is not None:
        predictions.append((GRAPH_MODEL_NAME, graph_guesses))

    truths = testing["true_source_node"].astype(str).tolist()
    results = []
    for model_name, guesses in predictions:
        for (_, record), truth, guess in zip(testing.iterrows(), truths, guesses):
            error = (
                np.nan if guess is None
                else nx.shortest_path_length(graph, source=truth, target=str(guess))
            )
            results.append({
                "run_id": str(record["run_id"]),
                "scenario_type": str(record.get("scenario_type", "unknown")),
                "observed_sensors": sum(pd.notna(record[f"{z}_reading"]) for z in ZONES),
                "model": model_name,
                "true_source": truth,
                "predicted_source": guess,
                "correct": int(guess == truth),
                "graph_distance": error,
            })

    test_predictions = pd.DataFrame(results)
    scores = test_predictions.groupby("model", sort=False).agg(
        accuracy=("correct", "mean"),
        mean_graph_error=("graph_distance", "mean"),
        trials=("correct", "size"),
    ).reset_index()
    return ModelArtifacts(
        models=models,
        source_path=str(path),
        train_count=len(training),
        test_count=len(testing),
        scores=scores,
        test_predictions=test_predictions,
        augmented_train_count=len(augmented_x),
        reference_models=reference_models,
    )


def predict_models(
    artifacts: ModelArtifacts, readings: Mapping[str, float | None]
) -> pd.DataFrame:
    features = reading_features(readings)
    if features[[f"{zone}_missing" for zone in ZONES]].to_numpy().all():
        return pd.DataFrame([
            {
                "model": name,
                "predicted_source": None,
                "max_class_probability": np.nan,
            }
            for name in artifacts.models
        ])

    output = []
    for name, estimator in artifacts.models.items():
        probabilities = estimator.predict_proba(features)[0]
        index = int(np.argmax(probabilities))
        output.append({
            "model": name,
            "predicted_source": str(estimator.classes_[index]),
            "max_class_probability": float(probabilities[index]),
        })
    return pd.DataFrame(output)
