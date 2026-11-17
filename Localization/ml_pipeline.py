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


def default_dataset_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    candidates = sorted((root / "Team" / "Justin" / "Data").glob("outbreak_dataset_*.csv"))
    if not candidates:
        raise FileNotFoundError("No Justin CSV found in Team/Justin/Data")
    return candidates[-1]


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


def train_models(dataset_path: str | Path | None = None) -> ModelArtifacts:
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

    X_train, X_test = _features(training), _features(testing)
    y_train = training["true_source_node"].astype(str)

    models = {
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
    for estimator in models.values():
        estimator.fit(X_train, y_train)

    graph = build_facility()
    results = []
    predictions = [("Strongest sensor", _baseline_predict(testing))]
    predictions.extend(
        (name, estimator.predict(X_test).tolist())
        for name, estimator in models.items()
    )
    truths = testing["true_source_node"].astype(str).tolist()

    for model_name, guesses in predictions:
        for (_, record), truth, guess in zip(testing.iterrows(), truths, guesses):
            error = (
                np.nan if guess is None
                else nx.shortest_path_length(graph, source=truth, target=str(guess))
            )
            results.append({
                "run_id": str(record["run_id"]),
                "scenario_type": str(record.get("scenario_type", "unknown")),
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
