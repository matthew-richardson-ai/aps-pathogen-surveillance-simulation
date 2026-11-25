"""Presentation scoring: source labels never enter the predictor path."""
import pandas as pd
import pytest

from OB.prediction_scoring import score_current_predictions, score_before_after


def test_current_predictions_get_correctness_and_shortest_path_error():
    original = pd.DataFrame([
        {"model": "Argmax", "predicted_source": "zone_3"},
        {"model": "Graph", "predicted_source": "zone_4"},
        {"model": "Abstain", "predicted_source": None},
    ])
    scored = score_current_predictions(original, "zone_3")
    assert scored["Correct?"].tolist() == ["Yes", "No", "No prediction"]
    assert scored["Graph Distance (edges)"].tolist() == [0, 1, "N/A"]
    assert "Correct?" not in original.columns
    assert "Graph Distance (edges)" not in original.columns


def test_before_after_metrics_can_improve_or_worsen():
    frame = pd.DataFrame([
        {"Model": "Improved", "Before": "zone_7", "After": "zone_3"},
        {"Model": "Worsened", "Before": "zone_3", "After": "zone_8"},
        {"Model": "Same", "Before": "zone_4", "After": "zone_4"},
    ])
    scored = score_before_after(frame, "zone_3")
    assert scored["Correct After?"].tolist() == ["Yes", "No", "No"]
    assert scored["Distance After (edges)"].tolist() == [0, 4, 1]
    assert scored["Error Reduction (edges)"].tolist() == [3, -4, 0]
    assert "Correct Before?" not in frame.columns


def test_invalid_truth_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        score_current_predictions(
            pd.DataFrame([{"predicted_source": "zone_3"}]), "zone_9",
        )
