"""Regression tests for Owen-inspired dashboard diagnostics."""
import pandas as pd
import pytest

from OB.dashboard_diagnostics import prediction_counts, accuracy_by_missingness


def test_confusion_counts_include_abstentions_and_unseen_classes():
    data = pd.DataFrame({
        "true_source": ["zone_1", "zone_1", "zone_2", "zone_2"],
        "predicted_source": ["zone_1", "zone_2", None, "zone_1"],
    })
    matrix = prediction_counts(data, ["zone_1", "zone_2", "zone_3"])
    assert matrix.loc["zone_1", "zone_1"] == 1
    assert matrix.loc["zone_1", "zone_2"] == 1
    assert matrix.loc["zone_2", "No prediction"] == 1
    assert matrix.loc["zone_3"].sum() == 0
    assert matrix.to_numpy().sum() == len(data)


def test_accuracy_uses_actual_observed_count_and_scores_abstentions_incorrect():
    data = pd.DataFrame({
        "true_source": ["zone_1", "zone_1", "zone_2", "zone_2"],
        "predicted_source": ["zone_1", None, "zone_2", "zone_1"],
        "observed_count": [8, 8, 5, 5],
        "missing_rate": [0.9, 0.9, 0, 0],
    })
    summary = accuracy_by_missingness(data, total_zones=8)
    assert summary["missing_readings"].tolist() == [0, 3]
    assert summary["trials"].tolist() == [2, 2]
    assert summary["accuracy"].tolist() == [0.5, 0.5]


def test_empty_records_have_stable_schema():
    empty = pd.DataFrame(columns=["true_source", "predicted_source", "observed_count"])
    assert accuracy_by_missingness(empty).columns.tolist() == [
        "missing_readings", "accuracy", "trials"
    ]


def test_rejects_invalid_observed_count():
    invalid = pd.DataFrame({
        "true_source": ["zone_1"], "predicted_source": ["zone_1"],
        "observed_count": [9],
    })
    with pytest.raises(ValueError):
        accuracy_by_missingness(invalid)
