"""Matched one-sample comparison: static versus random versus information gain.

Each policy receives exactly the same initial readings and at most one extra
measurement. Source truth is consulted only after prediction, for scoring.
All measurements are synthetic.
"""
from __future__ import annotations

from dataclasses import replace
import random

import networkx as nx
import numpy as np
import pandas as pd

from Adaptive.sampler import source_prior, choose_next_zone
from Localization.ml_pipeline import ModelArtifacts, reading_features
from Simulation.facility import build_facility
from OB.live_engine import Scenario, simulate, sample_one_more


def _model_prediction(estimator, readings):
    if all(value is None for value in readings.values()):
        return None
    return str(estimator.predict(reading_features(readings))[0])


def compare_sampling_policies(
    artifacts: ModelArtifacts,
    scenario: Scenario,
    trials: int = 80,
    model_names: tuple[str, ...] = ("Logistic Regression", "Random Forest"),
) -> pd.DataFrame:
    """Compare methods on matched fresh simulated scenarios.

    The experiment starts from seeds far removed from the saved dataset
    generator's default seeds, and randomizes the hidden source per run.
    The model-assisted selector has no access to the source label or the
    ideal readings; the simulator reveals one measurement after selection.
    """
    if not 1 <= trials <= 500:
        raise ValueError("Trials must be between 1 and 500")
    graph = build_facility()
    rows = []
    for i in range(trials):
        config = replace(scenario, seed=scenario.seed + 1_000_003 + i, source=None)
        initial = simulate(config)
        candidates = sorted(zone for zone, reading in initial.readings.items() if reading is None)
        if not candidates:
            raise ValueError("At least one unobserved zone is required")
        random_zone = random.Random(config.seed + 7819).choice(candidates)
        random_sample = sample_one_more(initial, random_zone)
        for model_name in model_names:
            estimator = artifacts.models[model_name]
            prior = source_prior(estimator, initial.readings)
            choice = choose_next_zone(
                graph, initial.readings, prior,
                source_intensity=config.source_intensity,
                decay_rate=config.decay_rate,
                sample_noise_std=0.04,
            )
            if choice is None:
                raise ValueError("No candidate for information-gain sampling")
            selected_sample = sample_one_more(initial, choice.zone)
            for strategy, state, sampled_zone in (
                ("static", initial, None),
                ("random", random_sample, random_zone),
                ("information_gain", selected_sample, choice.zone),
            ):
                predicted = _model_prediction(estimator, state.readings)
                rows.append({
                    "run_id": f"adaptive_{i + 1:04d}",
                    "seed": config.seed,
                    "model": model_name,
                    "strategy": strategy,
                    "true_source": initial.true_source,
                    "predicted_source": predicted,
                    "correct": int(predicted == initial.true_source),
                    "graph_distance": (
                        np.nan if predicted is None
                        else nx.shortest_path_length(graph, predicted, initial.true_source)
                    ),
                    "sampled_zone": sampled_zone,
                    "extra_samples": int(sampled_zone is not None),
                    "expected_gain_bits": (
                        choice.expected_gain_bits
                        if strategy == "information_gain" else np.nan
                    ),
                    "noise_std": config.noise_std,
                    "missing_rate": config.missing_rate,
                    "initial_sensors": config.initial_sensors,
                    "source_intensity": config.source_intensity,
                    "decay_rate": config.decay_rate,
                })
    return pd.DataFrame(rows)


def summarize_comparison(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize model accuracy, distance errors, and consumed sample budget."""
    if results.empty:
        raise ValueError("No experiment results")
    return results.groupby(["model", "strategy"], sort=False).agg(
        trials=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_graph_error=("graph_distance", "mean"),
        additional_samples=("extra_samples", "mean"),
    ).reset_index()
