"""Session-to-session APS comparison for a live classroom demonstration.

This module compares two DISTINCT simulated events (previous vs current), not
the before/after measurements of the same event. Predictions remain blind:
the true source is supplied only for the optional revealed scoring view.

A previous scenario remains available only for the active Streamlit session.
Comparing different events is descriptive, not a controlled test of adaptive
sampling or evidence that changing the seed caused an accuracy improvement.
"""
from __future__ import annotations

import pandas as pd

from OB.prediction_scoring import score_current_predictions


def _observed(snapshot) -> int:
    return sum(reading is not None for reading in snapshot.readings.values())


def comparison_table(
    previous_snapshot,
    current_snapshot,
    previous_predictions: pd.DataFrame,
    current_predictions: pd.DataFrame,
    *,
    reveal: bool = False,
) -> pd.DataFrame:
    """Compare prediction locations across two distinct snapshots.

    Required prediction inputs: model, predicted_source. The common model IDs
    must be unique. When reveal=False neither source label nor correctness or
    distance is consulted or exposed to the display table.
    """
    fields = {"model", "predicted_source"}
    for frame in (previous_predictions, current_predictions):
        if not fields.issubset(frame.columns):
            raise ValueError("Scenario comparison requires model and predicted_source")
        if frame["model"].duplicated().any():
            raise ValueError("One prediction per model is required")

    older = previous_predictions.loc[:, ["model", "predicted_source"]].copy()
    newer = current_predictions.loc[:, ["model", "predicted_source"]].copy()
    if reveal:
        older = score_current_predictions(older, previous_snapshot.true_source)
        newer = score_current_predictions(newer, current_snapshot.true_source)
    combined = older.merge(newer, how="inner", on="model", suffixes=("_previous", "_current"))
    if combined.empty:
        raise ValueError("No shared localization methods in scenario comparison")

    combined = combined.rename(columns={
        "model": "Model",
        "predicted_source_previous": "Previous Prediction",
        "predicted_source_current": "Current Prediction",
    })
    if reveal:
        combined = combined.rename(columns={
            "Correct?_previous": "Previous Correct?",
            "Correct?_current": "Current Correct?",
            "Graph Distance (edges)_previous": "Previous Distance (edges)",
            "Graph Distance (edges)_current": "Current Distance (edges)",
        })
        # The difference is between separate event difficulties, not a
        # before/after sampling improvement on a fixed hidden source.
        p_dist, c_dist = ("Previous Distance (edges)", "Current Distance (edges)")
        combined["Distance Difference (edges)"] = [
            "N/A" if isinstance(old, str) or isinstance(new, str)
            else old - new
            for old, new in zip(combined[p_dist], combined[c_dist])
        ]
    return combined


def scenario_takeaways(
    previous_snapshot,
    current_snapshot,
    comparison: pd.DataFrame,
    *,
    reveal: bool = False,
) -> list[str]:
    """Short, deterministic statements derived strictly from the table.

    Do not imply significance, causality, or that the samples are comparable
    training/test rows. Source identity is mentioned only when reveal=True.
    """
    prev_seed = previous_snapshot.parameters.seed
    curr_seed = current_snapshot.parameters.seed
    prev_stage = "after an extra measurement" if previous_snapshot.requested_zone else "before any extra measurement"
    curr_stage = "after an extra measurement" if current_snapshot.requested_zone else "before any extra measurement"
    sentences = [
        f"Previous: seed {prev_seed} ({prev_stage}); current: seed {curr_seed} ({curr_stage}).",
        f"Available readings: {_observed(previous_snapshot)}/8 previously, {_observed(current_snapshot)}/8 now.",
    ]
    if not reveal:
        sentences.append(
            "Reveal the true source to compare exact matches and graph-distance errors. "
            "No model receives the answer for prediction."
        )
    else:
        correct_prev = int((comparison["Previous Correct?"] == "Yes").sum())
        correct_curr = int((comparison["Current Correct?"] == "Yes").sum())
        count = len(comparison)
        sentences.append(
            f"Exact matches: {correct_prev}/{count} in the previous event versus "
            f"{correct_curr}/{count} in the current event "
            f"({correct_curr-correct_prev:+d} models)."
        )
        improved = comparison.loc[
            comparison["Previous Correct?"].ne("Yes") &
            comparison["Current Correct?"].eq("Yes"), "Model"
        ].tolist()
        worsened = comparison.loc[
            comparison["Previous Correct?"].eq("Yes") &
            comparison["Current Correct?"].ne("Yes"), "Model"
        ].tolist()
        if improved:
            sentences.append("Correct only in the current event: " + ", ".join(improved) + ".")
        if worsened:
            sentences.append("Correct only in the previous event: " + ", ".join(worsened) + ".")
        if not improved and not worsened:
            sentences.append("Each method's correct/incorrect status is unchanged between events.")
    sentences.append(
        "These are different simulated events, so this comparison does not "
        "measure the effect of additional sampling. Use the within-event "
        "before/after table and repeated trials for that question."
    )
    return sentences
