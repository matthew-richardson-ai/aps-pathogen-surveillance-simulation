"""Matched-run integration checks for graph-aware and robust APS classifiers.

The graph model and learned models must see the same seeded initial readings,
random-sampling budget, and model-selected measurement budget.
"""
from collections import Counter

import numpy as np

from Localization.graph_likelihood import GRAPH_MODEL_NAME
from Localization.ml_pipeline import ModelArtifacts, ZONES
from OB.adaptive_experiments import compare_sampling_policies, summarize_comparison
from OB.live_engine import Scenario


class UniformToyClassifier:
    """Deterministic test double with valid eight-class probabilities."""
    classes_ = np.asarray(ZONES)

    def predict_proba(self, features):
        return np.full((len(features), len(ZONES)), 1 / len(ZONES))

    def predict(self, features):
        return np.asarray(["zone_1"] * len(features))


def test_graph_and_ml_share_exactly_matched_scenarios_and_budgets():
    toy = UniformToyClassifier()
    artifacts = ModelArtifacts(
        models={"Logistic Regression": toy, "Random Forest": toy},
        source_path="synthetic-test-only",
        train_count=8,
        test_count=8,
        scores=None,
        test_predictions=None,
        reference_models={"Logistic Regression": toy, "Random Forest": toy},
    )
    methods = ["Logistic Regression", "Logistic Regression (Original)", GRAPH_MODEL_NAME]
    runs = compare_sampling_policies(
        artifacts, Scenario(seed=82, initial_sensors=4, noise_std=.22,
                            missing_rate=.35, source_intensity=.45, decay_rate=.95),
        trials=3, model_names=tuple(methods),
    )
    assert len(runs) == 3 * len(methods) * 3
    assert set(runs["strategy"]) == {"static", "random", "information_gain"}
    assert set(runs["model"]) == set(methods)
    assert runs.groupby(["run_id", "model"]).size().eq(3).all()
    assert (runs[runs["strategy"] == "static"]["extra_samples"] == 0).all()
    assert (runs[runs["strategy"] != "static"]["extra_samples"] == 1).all()
    assert (runs.groupby("run_id")["true_source"].nunique() == 1).all()
    assert (runs.groupby("run_id")["initial_observation_count"].nunique() == 1).all()
    assert (runs.groupby("run_id")["source_initially_observed"].nunique() == 1).all()
    summary = summarize_comparison(runs)
    assert len(summary) == len(methods) * 3


def test_graph_likelihood_does_not_require_a_trained_classifier():
    artifacts = ModelArtifacts(
        models={}, source_path="no-training-needed", train_count=0, test_count=0,
        scores=None, test_predictions=None,
    )
    runs = compare_sampling_policies(
        artifacts,
        Scenario(seed=80, initial_sensors=4, noise_std=.22,
                 missing_rate=.35, source_intensity=.45, decay_rate=.95),
        trials=2, model_names=(GRAPH_MODEL_NAME,),
    )
    assert len(runs) == 6
    assert Counter(runs["strategy"]) == {
        "static": 2, "random": 2, "information_gain": 2,
    }
