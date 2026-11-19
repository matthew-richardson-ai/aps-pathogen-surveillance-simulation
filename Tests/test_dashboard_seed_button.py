"""Regression checks for Streamlit scenario-seed button callbacks.

Importing the full Streamlit dashboard executes the app, so these tests load
only the small callback function from its AST and exercise its state changes.
"""
import ast
from pathlib import Path
from types import SimpleNamespace


DASHBOARD = Path(__file__).resolve().parents[1] / "OB" / "live_dashboard.py"


def _callback_and_state(seed):
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    func = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "advance_seed"
    )
    state = {"seed": seed, "source": "zone_6", "demo_reveal": True, "reveal_truth": True, "preset": "Combined stress"}
    context = {"st": SimpleNamespace(session_state=state)}
    exec(compile(ast.Module(body=[func], type_ignores=[]), str(DASHBOARD), "exec"), context)
    return context["advance_seed"], state


def test_generate_seed_callback_advances_the_seed():
    advance_seed, state = _callback_and_state(81)
    advance_seed()
    assert state["seed"] == 82
    assert state["source"] == "Random"
    assert state["preset"] == "Combined stress"
    assert state["demo_reveal"] is False
    assert state["reveal_truth"] is False
    advance_seed()
    assert state["seed"] == 83
    assert state["source"] == "Random"


def test_seed_callback_stays_within_number_input_bounds():
    advance_seed, state = _callback_and_state(2_000_000_000)
    advance_seed()
    assert state["seed"] == 0


def test_scenario_button_uses_pre_render_callback():
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    buttons = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "button"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "Generate New Scenario (Random Source)"
    ]
    assert len(buttons) == 1
    callback = next(
        (kw.value for kw in buttons[0].keywords if kw.arg == "on_click"),
        None,
    )
    assert isinstance(callback, ast.Name)
    assert callback.id == "advance_seed"



def test_fixed_combined_stress_scenario_is_still_reproducible():
    """The demo preset remains fixed even though the New Scenario button randomizes."""
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    presets = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "DEMO_SCENES"
                for target in node.targets)
    )
    scenes = ast.literal_eval(presets.value)
    assert scenes["Combined Stress"]["source"] == "zone_6"
    assert scenes["Combined Stress"]["seed"] == 81
