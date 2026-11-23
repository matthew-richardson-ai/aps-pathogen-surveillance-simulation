"""Reusable evaluation diagnostics adapted from Owen's CSV dashboard.

Predictions must be computed independently of ground truth. Truth is used only
for retrospective scoring; all-missing/abstaining trials remain in the counts.
"""
from __future__ import annotations

import pandas as pd


def prediction_counts(records: pd.DataFrame, zones: list[str]) -> pd.DataFrame:
    """Rows are true source zones; columns are predictions including abstentions."""
    required = {"true_source", "predicted_source"}
    if not required.issubset(records.columns):
        raise ValueError("Prediction records require true_source and predicted_source")
    labels = list(zones) + ["No prediction"]
    predicted = records["predicted_source"].fillna("No prediction")
    return pd.crosstab(records["true_source"], predicted).reindex(
        index=zones, columns=labels, fill_value=0
    ).astype(int)


def accuracy_by_missingness(records: pd.DataFrame, total_zones: int = 8) -> pd.DataFrame:
    """Compute observed exact-source accuracy by unavailable reading count.

    A row's missing count is derived from observed_count, not its configured
    missing-rate probability. Abstentions count as incorrect.
    """
    required = {"true_source", "predicted_source", "observed_count"}
    if not required.issubset(records.columns):
        raise ValueError("Prediction records require true_source, predicted_source, observed_count")
    if records.empty:
        return pd.DataFrame(columns=["missing_readings", "accuracy", "trials"])
    count = pd.to_numeric(records["observed_count"], errors="raise")
    if count.isna().any() or not count.between(0, total_zones).all():
        raise ValueError("observed_count must be between zero and total_zones")
    working = pd.DataFrame({
        "missing_readings": total_zones - count.astype(int),
        "correct": records["predicted_source"].eq(records["true_source"]),
    })
    return working.groupby("missing_readings", as_index=False).agg(
        accuracy=("correct", "mean"), trials=("correct", "size")
    ).sort_values("missing_readings").reset_index(drop=True)
