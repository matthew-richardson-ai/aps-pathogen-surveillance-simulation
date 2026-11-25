"""Prepare reproducible APS evaluation files before a classroom demo.

Run from the repository root:
    python Scripts/prepare_demo_results.py

This performs real synthetic experiments; no result numbers are hard-coded.
The outputs are kept outside Git until reviewed and deliberately committed.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Localization.ml_pipeline import train_models
from Localization.graph_likelihood import GRAPH_MODEL_NAME
from OB.adaptive_experiments import compare_sampling_policies, summarize_comparison
from OB.live_engine import Scenario


def main() -> None:
    output = ROOT / "Results" / "Demo"
    output.mkdir(parents=True, exist_ok=True)
    print("Training original and sparsity-augmented ML models on Justin's dataset...")
    trained = train_models()
    print(f"Independent training rows: {trained.train_count}; augmented views: {trained.augmented_train_count}; untouched held-out test rows: {trained.test_count}")
    trained.scores.to_csv(output / "heldout_model_summary.csv", index=False)
    trained.test_predictions.to_csv(output / "heldout_model_predictions.csv", index=False)

    scenarios = (
        ("Normal Conditions", Scenario(seed=10000, initial_sensors=6, noise_std=0.06,
                                       missing_rate=0.05, source_intensity=1.0, decay_rate=0.65)),
        ("High Noise", Scenario(seed=20000, initial_sensors=5, noise_std=0.26,
                                missing_rate=0.10, source_intensity=1.0, decay_rate=0.65)),
        ("Missing Readings", Scenario(seed=30000, initial_sensors=6, noise_std=0.08,
                                      missing_rate=0.50, source_intensity=1.0, decay_rate=0.65)),
        ("Combined Stress", Scenario(seed=40000, initial_sensors=4, noise_std=0.22,
                                     missing_rate=0.35, source_intensity=0.45, decay_rate=0.95)),
    )
    # All models are evaluated on the SAME events. The original, non-augmented
    # classifiers remain as controls against the robust training variants.
    evaluated_models = (
        "Logistic Regression (Original)", "Logistic Regression",
        "Random Forest (Original)", "Random Forest",
        GRAPH_MODEL_NAME,
    )
    results = []
    for label, parameters in scenarios:
        print(f"Evaluating {label}...")
        comparison = compare_sampling_policies(
            trained, replace(parameters, source=None),
            trials=100, model_names=evaluated_models,
        )
        comparison["condition"] = label
        comparison["run_id"] = comparison["run_id"].map(lambda run: f"{label.replace(' ', '_').lower()}_{run}")
        results.append(comparison)

    all_runs = pd.concat(results, ignore_index=True)
    group_cols = ["condition", "model", "strategy"]
    grouped = all_runs.groupby(group_cols, sort=False, dropna=False).agg(
        trials=("correct", "size"),
        accuracy=("correct", "mean"),
        mean_graph_error=("graph_distance", "mean"),
        mean_additional_samples=("extra_samples", "mean"),
    ).reset_index()
    # Evaluate explicitly when the source-zone sensor was unavailable at first.
    # The true source is used only for retrospective scoring here.
    missing_source = all_runs.loc[all_runs["source_initially_observed"] == 0]
    missing_accuracy = missing_source.groupby(group_cols, sort=False).agg(
        missing_source_trials=("correct", "size"),
        missing_source_accuracy=("correct", "mean"),
    ).reset_index()
    grouped = grouped.merge(missing_accuracy, on=group_cols, how="left")
    all_runs.to_csv(output / "sampling_runs.csv", index=False)
    grouped.to_csv(output / "sampling_summary.csv", index=False)

    metadata = {
        "data_source": str(Path(trained.source_path).relative_to(ROOT)),
        "train_rows": trained.train_count,
        "augmented_train_views": trained.augmented_train_count,
        "heldout_test_rows": trained.test_count,
        "simulations_per_condition": 100,
        "scenarios": [name for name, _ in scenarios],
        "models": list(evaluated_models),
        "comparison": ["static", "random", "information_gain"],
        "sample_budget": {"static": 0, "random": 1, "information_gain": 1},
        "evaluation_version": "blind_graph_marginalized_v1",
        "graph_method": GRAPH_MODEL_NAME,
        "graph_parameter_access": "NONE: inferred from readings; fixed public parameter grid",
        "scoring_only_truth_access": True,
        "limitations": (
            "Synthetic spatial snapshots only; model probabilities are "
            "conditional on assumptions and not calibrated. Graph inference "
            "integrates over scenario-independent intensity/decay/noise "
            "ranges and receives no true per-event simulator parameters. "
            "Source truth is used for scoring only."
        ),
    }
    (output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nHeld-out model performance:")
    print(trained.scores.to_string(index=False))
    print("\nAdaptive comparison:")
    print(grouped.to_string(index=False))
    print(f"\nSaved result files to: {output}")


if __name__ == "__main__":
    main()
