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
from OB.adaptive_experiments import compare_sampling_policies, summarize_comparison
from OB.live_engine import Scenario


def main() -> None:
    output = ROOT / "Results" / "Demo"
    output.mkdir(parents=True, exist_ok=True)
    print("Training Logistic Regression and Random Forest on Justin's dataset...")
    trained = train_models()
    print(f"Training rows: {trained.train_count}; held-out test rows: {trained.test_count}")
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
    results = []
    for label, parameters in scenarios:
        print(f"Evaluating {label}...")
        comparison = compare_sampling_policies(trained, replace(parameters, source=None), trials=100)
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
    all_runs.to_csv(output / "sampling_runs.csv", index=False)
    grouped.to_csv(output / "sampling_summary.csv", index=False)

    metadata = {
        "data_source": str(Path(trained.source_path).relative_to(ROOT)),
        "train_rows": trained.train_count,
        "heldout_test_rows": trained.test_count,
        "simulations_per_condition": 100,
        "scenarios": [name for name, _ in scenarios],
        "comparison": ["static", "random", "information_gain"],
        "sample_budget": {"static": 0, "random": 1, "information_gain": 1},
        "limitations": "Synthetic spatial snapshots only; model probabilities not calibrated.",
    }
    (output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nHeld-out model performance:")
    print(trained.scores.to_string(index=False))
    print("\nAdaptive comparison:")
    print(grouped.to_string(index=False))
    print(f"\nSaved result files to: {output}")


if __name__ == "__main__":
    main()
