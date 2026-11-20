"""Presentation-only APS scoring, separated from all source estimators.

Ground-truth labels are passed to these functions only *after* a presenter
reveals the source. Never import this module from prediction or sensor-selection
code. Graph distance is the number of edges on the canonical eight-zone graph;
it is not a physical distance in meters.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd

from Simulation.facility import build_facility

ZONES = tuple(f"zone_{i}" for i in range(1, 9))


def _distance(predicted: str | None, true_source: str, graph: nx.Graph) -> int | None:
    if predicted is None or pd.isna(predicted):
        return None
    if predicted not in graph:
        raise ValueError("Prediction is not a valid APS facility zone")
    return int(nx.shortest_path_length(graph, predicted, true_source))


def score_current_predictions(frame: pd.DataFrame, true_source: str) -> pd.DataFrame:
    """Add correctness and graph error for a revealed single-event outcome.

    Input must have 'predicted_source'; the estimator never sees true_source.
    """
    graph = build_facility()
    if true_source not in graph:
        raise ValueError("True source is outside the APS facility")
    scores = frame.copy()
    distances = [
        _distance(predicted, true_source, graph)
        for predicted in scores["predicted_source"]
    ]
    scores["Correct?"] = [
        "No prediction" if distance is None else
        "Yes" if distance == 0 else "No"
        for distance in distances
    ]
    scores["Graph Distance (edges)"] = [
        "N/A" if distance is None else distance for distance in distances
    ]
    return scores


def score_before_after(frame: pd.DataFrame, true_source: str) -> pd.DataFrame:
    """Add before/after truth checks without modifying model predictions.

    A positive Error Reduction means the prediction moved closer to the
    hidden source. Zero means no improvement. This is an evaluation-only view.
    """
    graph = build_facility()
    if true_source not in graph:
        raise ValueError("True source is outside the APS facility")
    result = frame.copy()
    for suffix, field in (("Before", "Before"), ("After", "After")):
        distances = [_distance(value, true_source, graph) for value in result[field]]
        result[f"Correct {suffix}?"] = [
            "No prediction" if d is None else "Yes" if d == 0 else "No"
            for d in distances
        ]
        result[f"Distance {suffix} (edges)"] = [
            "N/A" if d is None else d for d in distances
        ]
        if suffix == "Before":
            before_values = distances
        else:
            after_values = distances
    result["Error Reduction (edges)"] = [
        "N/A" if before is None or after is None else before - after
        for before, after in zip(before_values, after_values)
    ]
    return result
