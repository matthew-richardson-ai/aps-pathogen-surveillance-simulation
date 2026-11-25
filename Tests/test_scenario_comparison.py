"""Tests for APS previous/current scenario comparison in Streamlit."""
import ast
from pathlib import Path

import pandas as pd
import pytest

from OB.live_engine import Scenario, simulate, sample_one_more
from OB.scenario_comparison import comparison_table, scenario_takeaways, sampling_takeaways


def example_events():
    previous = simulate(Scenario(seed=84, source="zone_5", initial_sensors=4))
    current = simulate(Scenario(seed=85, source="zone_4", initial_sensors=4))
    old_rows = pd.DataFrame([
        {"model": "Argmax Baseline", "predicted_source": "zone_5"},
        {"model": "Graph-Aware Bayesian Inference", "predicted_source": "zone_2"},
    ])
    new_rows = pd.DataFrame([
        {"model": "Argmax Baseline", "predicted_source": "zone_4"},
        {"model": "Graph-Aware Bayesian Inference", "predicted_source": "zone_2"},
    ])
    return previous, current, old_rows, new_rows


def test_hidden_comparison_contains_no_ground_truth_or_error_columns():
    previous, current, old, new = example_events()
    summary = comparison_table(previous, current, old, new)
    assert list(summary.columns) == [
        "Model", "Previous Prediction", "Current Prediction",
    ]
    texts = scenario_takeaways(previous, current, summary)
    assert any("Reveal the true source" in text for text in texts)
    assert all("source: zone_" not in text.lower() for text in texts)
    assert any("different simulated events" in text for text in texts)


def test_revealed_comparison_scores_each_event_against_its_own_source():
    previous, current, old, new = example_events()
    summary = comparison_table(previous, current, old, new, reveal=True)
    assert summary["Previous Correct?"].tolist() == ["Yes", "No"]
    assert summary["Current Correct?"].tolist() == ["Yes", "No"]
    assert summary["Previous Distance (edges)"].tolist() == [0, 2]
    assert summary["Current Distance (edges)"].tolist() == [0, 2]
    assert summary["Distance Difference (edges)"].tolist() == [0, 0]
    assert old.columns.tolist() == ["model", "predicted_source"]
    assert new.columns.tolist() == ["model", "predicted_source"]
    texts = scenario_takeaways(previous, current, summary, reveal=True)
    assert any("2" in text and "Exact matches:" in text for text in texts)


def test_takeaways_reflect_post_sampling_history():
    previous, current, old, new = example_events()
    selected = next(zone for zone, value in previous.readings.items() if value is None)
    previous = sample_one_more(previous, selected)
    comparison = comparison_table(previous, current, old, new)
    explanation = scenario_takeaways(previous, current, comparison)
    assert "after an extra measurement" in explanation[0]
    assert "before any extra measurement" in explanation[0]


def test_duplicate_methods_rejected():
    previous, current, old, new = example_events()
    duplicate = pd.concat([old, old.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="One prediction per model"):
        comparison_table(previous, current, duplicate, new)


def test_two_reveal_buttons_use_the_same_pre_render_callback():
    source = (Path(__file__).resolve().parents[1] / "OB" / "live_dashboard.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    button_nodes = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "button"
        and any(kw.arg == "key" and isinstance(kw.value, ast.Constant)
                and kw.value.value in {"reveal_near_predictions", "reveal_near_graphs"}
                for kw in node.keywords)
    ]
    assert len(button_nodes) == 2
    assert all(
        any(kw.arg == "on_click" and isinstance(kw.value, ast.Name)
            and kw.value.id == "toggle_demo_reveal" for kw in node.keywords)
        for node in button_nodes
    )
    assert "st.session_state.previous_snapshot = prior_snapshot" in source


def test_sampling_takeaways_report_exact_improvement_only_if_revealed():
    hidden = pd.DataFrame([
        {"Model": "A", "Before": "Zone 8", "After": "Zone 5"},
        {"Model": "B", "Before": "Zone 5", "After": "Zone 5"},
        {"Model": "C", "Before": "Zone 2", "After": "Zone 2"},
    ])
    lines = sampling_takeaways(hidden)
    assert "1 of 3" in lines[0]
    assert any("Reveal True Source" in line for line in lines)
    assert all("correct" not in line.lower() for line in lines)

    revealed = hidden.copy()
    revealed["Correct Before?"] = ["No", "Yes", "No"]
    revealed["Correct After?"] = ["Yes", "Yes", "No"]
    revealed["Error Reduction (edges)"] = [3, 0, 0]
    scored = sampling_takeaways(revealed, reveal=True)
    assert any("from 1/3 to 2/3" in line for line in scored)
    assert any("improved for 1 methods" in line for line in scored)
