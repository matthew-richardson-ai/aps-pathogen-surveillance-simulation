"""Interactive APS simulation dashboard built on the real eight-zone simulator.

From repository root:
    python -m streamlit run OB/live_dashboard.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Simulation.facility import build_facility
from live_engine import Scenario, simulate, sample_one_more, predict_baseline, baseline_error, batch_baseline
from Localization.ml_pipeline import train_models, predict_models, reading_features
from Localization.graph_likelihood import GRAPH_MODEL_NAME, graph_prediction, source_posterior
from Adaptive.sampler import source_prior, choose_next_zone
from OB.adaptive_experiments import compare_sampling_policies, summarize_comparison
from facility_3d import facility_3d_plot

st.set_page_config(page_title="APS | Simulation Lab", page_icon="🧬", layout="wide")

@st.cache_resource(show_spinner="Training APS models on Justin's synthetic dataset...")
def load_ml_artifacts():
    return train_models()
BG, PANEL, MUTED, CYAN, GREEN, PURPLE = "#0B1220", "#142238", "#DEE8F6", "#40DCD0", "#65E6BA", "#BBA7FF"

def zone_label(zone):
    """Use readable UI names without modifying canonical simulation IDs."""
    if zone is None or bool(pd.isna(zone)):
        return "No Reading"
    value = str(zone)
    if value.startswith("zone_") and value[5:].isdigit():
        return f"Zone {int(value[5:])}"
    return value.replace("_", " ").title()

def condition_label(value):
    return str(value).replace("_", " ").title()

ARGMAX_LABEL = "Argmax Baseline"

def method_label(value):
    """Present the existing heuristic by its mathematical algorithm name."""
    return ARGMAX_LABEL if value in ("Strongest sensor", "Strongest Sensor") else value

def graph_for_snapshot(snapshot):
    """Graph model uses known simulator parameters, never hidden-source truth."""
    cfg = snapshot.parameters
    return graph_prediction(
        snapshot.readings,
        source_intensity=cfg.source_intensity,
        decay_rate=cfg.decay_rate,
        noise_std=cfg.noise_std,
        requested_zone=snapshot.requested_zone,
    )


def sampling_prior(method, snapshot):
    """Use either learned ML probabilities or the graph likelihood posterior.

    Only robust ML models (not their original controls) drive sampling.
    The graph-aware model uses the configured simulator physics assumptions.
    """
    if method == GRAPH_MODEL_NAME:
        cfg = snapshot.parameters
        return source_posterior(
            snapshot.readings,
            source_intensity=cfg.source_intensity,
            decay_rate=cfg.decay_rate,
            noise_std=cfg.noise_std,
            requested_zone=snapshot.requested_zone,
        )
    return source_prior(load_ml_artifacts().models[method], snapshot.readings)


def all_current_predictions(artifacts, snapshot):
    """Compare all methods on exactly the same current observation snapshot.

    The Argmax heuristic is shown separately in the facility metrics.
    Original ML variants are controls; robust variants are actively used.
    All-missing views abstain rather than guessing a location.
    """
    observations = snapshot.readings
    frame = predict_models(artifacts, observations)
    missing_all = all(value is None for value in observations.values())
    if artifacts.reference_models:
        controls = []
        for label, estimator in artifacts.reference_models.items():
            if missing_all:
                guess, probability = None, float("nan")
            else:
                probabilities = estimator.predict_proba(reading_features(observations))[0]
                index = int(probabilities.argmax())
                guess, probability = str(estimator.classes_[index]), float(probabilities[index])
            controls.append({
                "model": f"{label} (Original)",
                "predicted_source": guess,
                "max_class_probability": probability,
            })
        frame = pd.concat([frame, pd.DataFrame(controls)], ignore_index=True)

    graph_guess, graph_scores = graph_for_snapshot(snapshot)
    graph_row = pd.DataFrame([{
        "model": GRAPH_MODEL_NAME,
        "predicted_source": graph_guess,
        "max_class_probability": (
            max(graph_scores.values()) if graph_guess is not None else float("nan")
        ),
    }])
    return pd.concat([frame, graph_row], ignore_index=True)


def display_zone_columns(frame, names):
    display = frame.copy()
    for field in names:
        if field in display.columns:
            display[field] = display[field].map(zone_label)
    return display

st.markdown("""
<style>
/* The native theme is set in .streamlit/config.toml; styles here
   augment it without replacing native widget colors. */
html { color-scheme: dark; }
.stApp, [data-testid="stAppViewContainer"] { background: #0B1220; color: #F0F5FC; }
[data-testid="stHeader"] { background: #0B1220; }
[data-testid="stSidebar"] { background: #121F33; border-right: 1px solid #32445D; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stWidgetLabel"] p { color: #E6EFFC !important; }
.stApp h1, .stApp h2, .stApp h3 { color: #F6F9FF; }
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] *,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stMarkdownContainer"] li {
  color: #EAF2FC !important;
  opacity: 1 !important;
}
.stApp [data-testid="stCaptionContainer"] * { font-size: 0.96rem !important; line-height: 1.6 !important; }
.stApp [data-testid="stWidgetLabel"] *,
.stApp [data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
  color: #F3F7FD !important;
  opacity: 1 !important;
}
[data-testid="stMetric"] { background: #17263D; border: 1px solid #324763;
  border-radius: 12px; padding: 14px; }
[data-testid="stMetricLabel"] p { color: #C5D6E9 !important; }
[data-testid="stMetricValue"] { color: #F1F7FF !important; }
[data-testid="stTabs"] button p { color: #BCCDE3 !important; }
[data-testid="stTabs"] button[aria-selected="true"] p { color: #65E5D9 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { border-color: #40DCD0 !important; }
.stApp button[kind="primary"],
.stApp button[data-testid="stBaseButton-primary"] {
  background: #40DDD0 !important;
  border: 1px solid #76F3E6 !important;
  color: #071C26 !important;
  font-weight: 750 !important;
}
.stApp button[kind="primary"] *,
.stApp button[data-testid="stBaseButton-primary"] * {
  color: #071C26 !important;
  opacity: 1 !important;
  font-weight: 750 !important;
}
.stApp button[kind="primary"]:hover,
.stApp button[data-testid="stBaseButton-primary"]:hover {
  background: #88F3E8 !important;
  border-color: #C7FFF6 !important;
}
.stApp button[kind="secondary"],
.stApp button[data-testid="stBaseButton-secondary"] {
  color: #F0F8FF !important;
  background: #22354E !important;
  border: 1px solid #71859F !important;
}
.stApp button[kind="secondary"] *,
.stApp button[data-testid="stBaseButton-secondary"] * { color: #F0F8FF !important; opacity: 1 !important; }
.stApp button[kind="secondary"]:hover,
.stApp button[data-testid="stBaseButton-secondary"]:hover {
  background: #304A69 !important;
  border-color: #94DBE1 !important;
}
.block-container { max-width: 1550px; padding-top: 1.35rem; }
.aps-tag { color: #65E5D9; font-size: .78rem; font-weight: 700; letter-spacing: .1em; }
.aps-notice { background: #192B42; border: 1px solid #38516B;
 border-left: 3px solid #40DCD0; border-radius: 9px; padding: 12px 14px;
 color: #E4EFF9; }
</style>
""", unsafe_allow_html=True)

PRESETS = {
    "Normal conditions": dict(initial_sensors=6, noise_std=0.06, missing_rate=0.05, source_intensity=1.0, decay_rate=0.65),
    "Limited coverage": dict(initial_sensors=3, noise_std=0.08, missing_rate=0.10, source_intensity=1.0, decay_rate=0.65),
    "High noise": dict(initial_sensors=5, noise_std=0.26, missing_rate=0.10, source_intensity=1.0, decay_rate=0.65),
    "High missingness": dict(initial_sensors=6, noise_std=0.08, missing_rate=0.50, source_intensity=1.0, decay_rate=0.65),
    "Weak source": dict(initial_sensors=5, noise_std=0.09, missing_rate=0.10, source_intensity=0.35, decay_rate=0.65),
    "Combined stress": dict(initial_sensors=4, noise_std=0.22, missing_rate=0.35, source_intensity=0.45, decay_rate=0.95),
}
DEMO_SCENES = {
    "Normal Conditions": {"preset": "Normal conditions", "source": "zone_4", "seed": 42},
    "Missing Readings": {"preset": "High missingness", "source": "zone_2", "seed": 64},
    "Combined Stress": {"preset": "Combined stress", "source": "zone_6", "seed": 81},
}

def load_demo_scene(name):
    """Apply repeatable demo controls before Streamlit renders the widgets."""
    scene = DEMO_SCENES[name]
    st.session_state["preset"] = scene["preset"]
    for field, value in PRESETS[scene["preset"]].items():
        st.session_state[field] = value
    st.session_state["source"] = scene["source"]
    st.session_state["seed"] = scene["seed"]
    st.session_state["demo_reveal"] = False
    st.session_state.pop("auto_sample_details", None)

for key, value in {"seed": 42, "source": "Random", "preset": "Normal conditions", **PRESETS["Normal conditions"]}.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("demo_selected_zone", "zone_3")
st.session_state.setdefault("live_selected_zone", "zone_3")

def apply_preset():
    for key, value in PRESETS[st.session_state.preset].items():
        st.session_state[key] = value

def advance_seed():
    """Generate a new random-source event, retaining the current conditions.

    This callback runs before sidebar widgets are instantiated. Loading a
    fixed demonstration preset remains reproducible, while generating a
    new event deliberately switches away from that fixed source.
    """
    max_seed = 2_000_000_000
    current = int(st.session_state["seed"])
    st.session_state["seed"] = current + 1 if current < max_seed else 0
    st.session_state["source"] = "Random"
    st.session_state["demo_reveal"] = False
    st.session_state["reveal_truth"] = False

with st.sidebar:
    st.markdown('<div class="aps-tag">APS / SIMULATION CONTROLS</div>', unsafe_allow_html=True)
    st.header("Simulation settings")
    st.selectbox("Scenario preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.selectbox("Hidden source", ["Random"] + sorted(build_facility().nodes),
                 key="source", format_func=lambda zone: "Random Source" if zone == "Random" else zone_label(zone))
    st.number_input("Random seed", min_value=0, max_value=2_000_000_000, step=1, key="seed")
    st.slider("Initially sampled zones", 2, 7, step=1, key="initial_sensors")
    st.slider("Sensor noise (standard deviation)", 0.0, 0.40, step=0.01, key="noise_std")
    st.slider("Missing-reading probability", 0.0, 0.95, step=0.05, key="missing_rate")
    st.slider("Source intensity", 0.20, 1.00, step=0.05, key="source_intensity")
    st.slider("Distance decay", 0.10, 1.50, step=0.05, key="decay_rate")
    st.button(
        "Generate New Scenario (Random Source)",
        type="primary",
        use_container_width=True,
        on_click=advance_seed,
    )
    st.caption(
        "Preset demo buttons load fixed examples. Generating a new scenario keeps "
        "the current sensor conditions but randomizes the hidden source and seed. "
        "A random selection can occasionally pick the same zone again."
    )
    st.divider()
    st.caption("CSC 4240 | Synthetic facility simulation only.")

params = Scenario(
    seed=int(st.session_state.seed),
    source=None if st.session_state.source == "Random" else st.session_state.source,
    initial_sensors=int(st.session_state.initial_sensors),
    noise_std=float(st.session_state.noise_std),
    missing_rate=float(st.session_state.missing_rate),
    source_intensity=float(st.session_state.source_intensity),
    decay_rate=float(st.session_state.decay_rate),
)
if st.session_state.get("active_params") != params:
    st.session_state.active_params = params
    st.session_state.active_snapshot = simulate(params)
    st.session_state.reveal_truth = False
    st.session_state.pop("manual_zone", None)
    st.session_state.pop("batch_results", None)
    st.session_state.pop("adaptive_results", None)
    st.session_state.pop("auto_sample_details", None)
snap = st.session_state.active_snapshot

# Stable layout of the canonical facility topology, not Owen's example graph.
POS = {
    "zone_1": (-1.9, 1.1), "zone_2": (-1.9, -1.1), "zone_3": (-0.6, 0),
    "zone_4": (0.65, 1.05), "zone_5": (0.65, -1.05), "zone_6": (1.9, 0),
    "zone_7": (3.1, 0), "zone_8": (4.25, 0),
}

def facility_plot(snapshot, reveal):
    fig = go.Figure()
    graph = build_facility()
    for a, b in graph.edges:
        fig.add_trace(go.Scatter(
            x=[POS[a][0], POS[b][0]], y=[POS[a][1], POS[b][1]], mode="lines",
            line=dict(color="#34465e", width=4), hoverinfo="skip", showlegend=False,
        ))
    present = [z for z, r in snapshot.readings.items() if r is not None]
    missing = [z for z, r in snapshot.readings.items() if r is None]
    for zones, available in [(present, True), (missing, False)]:
        if not zones:
            continue
        marker = dict(size=77, line=dict(color="#0B1220", width=2))
        if available:
            marker.update(color=[snapshot.readings[z] for z in zones],
                          colorscale=[[0, "#D9EFF9"], [.5, "#52DACC"], [1, "#F9CE83"]],
                          cmin=0, cmax=1, showscale=True,
                          colorbar=dict(title="Signal", thickness=12, len=.55, x=1.02))
        else:
            marker.update(color="#33465C", line=dict(color="#A7BBCE", width=2))
        fig.add_trace(go.Scatter(
            x=[POS[z][0] for z in zones], y=[POS[z][1] for z in zones],
            mode="markers+text", text=[zone_label(z) for z in zones],
            textposition="middle center", textfont=dict(size=12, color="#071624" if available else "#F0F6FF"),
            marker=marker, name="Measured" if available else "Unavailable",
            customdata=[[zone_label(z), f"{snapshot.readings[z]:.3f}" if available else "Not Observed"] for z in zones],
            hovertemplate="%{customdata[0]}: %{customdata[1]}<extra></extra>",
        ))
    prediction = predict_baseline(snapshot.readings)
    if prediction:
        x, y = POS[prediction]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name="Argmax Baseline",
            marker=dict(symbol="circle-open", size=78, line=dict(color=GREEN, width=6)),
            hovertemplate=f"Argmax Baseline: {zone_label(prediction)}<extra></extra>",
        ))
    if snapshot.requested_zone:
        x, y = POS[snapshot.requested_zone]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name="Additional measurement",
            marker=dict(symbol="circle-open", size=89, line=dict(color=PURPLE, width=3)),
            hovertemplate=f"Extra sample: {zone_label(snapshot.requested_zone)}<extra></extra>",
        ))
    if reveal:
        x, y = POS[snapshot.true_source]
        fig.add_trace(go.Scatter(
            x=[x+.27], y=[y+.37], mode="markers", name="True source",
            marker=dict(symbol="star", size=22, color="#fff493"),
            hovertemplate=f"True source: {zone_label(snapshot.true_source)}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTED), height=490,
        margin=dict(l=15, r=75, t=20, b=5),
        xaxis=dict(visible=False, range=[-2.65, 5.0]),
        yaxis=dict(visible=False, range=[-1.85, 2.15], scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", x=0.02, y=-.07, font=dict(size=11)),
        dragmode="pan",
    )
    return fig

def reading_plot(snapshot):
    zones = sorted(snapshot.readings)
    values = [snapshot.readings[z] for z in zones]
    fig = go.Figure(go.Bar(
        x=[zone_label(z) for z in zones],
        y=[v if v is not None else 0 for v in values],
        marker_color=[CYAN if v is not None else "#34465e" for v in values],
        customdata=["unavailable" if v is None else f"{v:.3f}" for v in values],
        hovertemplate="%{x}: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTED),
        height=240, margin=dict(l=8, r=8, t=15, b=8), showlegend=False,
        yaxis=dict(range=[0, 1.05], title="Signal", gridcolor="#27394e"),
    )
    return fig

st.markdown('<div class="aps-tag">ARTIFICIAL INTELLIGENCE / INTERACTIVE SIMULATION</div>', unsafe_allow_html=True)
st.title("Adaptive Pathogen Surveillance")
st.caption("Scenario Lab | Explore how incomplete, noisy sensor data changes source localization.")

demo_tab, live, models_tab, adaptive_tab, batch_tab, about = st.tabs(["APS Demonstration", "Live Facility", "Model Comparison", "Adaptive Sampling", "Experiment Lab", "Research Method"])

with demo_tab:
    st.subheader("Source Localization: Live Simulation")
    st.caption("Explore how sensor coverage, missing readings, and noise affect source localization in an eight-zone simulated facility.")
    st.info(
        "APS simulates eight connected zones. Room shapes are illustrative: "
        "the connected graph and sensor readings come from the simulation, "
        "not an architectural floor plan or airflow model. "
        "The true source is hidden from the prediction methods."
    )

    scene_cols = st.columns(3)
    for column, name in zip(scene_cols, DEMO_SCENES):
        with column:
            st.button(
                f"Load {name}",
                key=f"demo_scene_{name.replace(' ', '_')}",
                use_container_width=True,
                on_click=load_demo_scene,
                args=(name,),
            )
    scene_name = next(
        (name for name, scene in DEMO_SCENES.items()
         if (st.session_state.source == scene["source"]
             and st.session_state.seed == scene["seed"]
             and st.session_state.preset == scene["preset"])),
        "Custom Scenario",
    )
    st.caption(f"Scenario: {scene_name} | Seed: {params.seed} | "
               f"Initial sensor budget: {params.initial_sensors} zones")

    st.markdown("#### 1. Simulated Facility")
    view_left, view_right = st.columns([1.8, 1], gap="large")
    with view_left:
        demo_view = st.radio(
            "Facility visualization", ["3D Rooms", "2D Network"],
            key="demo_facility_view", horizontal=True,
        )
        st.checkbox("Reveal True Source", key="demo_reveal")
        if demo_view == "3D Rooms":
            graph_figure = facility_3d_plot(
                snap, build_facility(), predict_baseline(snap.readings),
                show_source=st.session_state.demo_reveal,
                selected_zone=st.session_state.demo_selected_zone,
            )
        else:
            graph_figure = facility_plot(snap, st.session_state.demo_reveal)
        st.plotly_chart(
            graph_figure, use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )
        if demo_view == "3D Rooms":
            st.caption(
                "Drag to rotate; scroll to zoom. Blue = low signal, amber = high signal, "
                "gray = unavailable. Green diamond = Argmax prediction, purple = additional sample, "
                "gold = true source when revealed."
            )
        else:
            st.caption("Connections show graph edges, not actual corridors or airflow.")
    with view_right:
        available = sum(reading is not None for reading in snap.readings.values())
        st.metric("Available sensor readings", f"{available} / 8")
        st.metric(ARGMAX_LABEL, zone_label(predict_baseline(snap.readings)))
        graph_guess, graph_probs = graph_for_snapshot(snap)
        st.metric("Graph-Aware Likelihood", zone_label(graph_guess))
        st.caption("Uses known simulated decay, signal intensity and noise; not a trained ML model.")
        top_sources = sorted(graph_probs.items(), key=lambda item: item[1], reverse=True)[:3]
        st.caption(
            "Top candidate zones: " +
            " · ".join(f"{zone_label(zone)} ({prob:.1%})" for zone, prob in top_sources)
            + " — estimates under the assumed simulator model"
        )
        st.metric("Random seed", params.seed)
        room = st.selectbox(
            "Inspect a Room", sorted(snap.readings),
            key="demo_selected_zone", format_func=zone_label,
        )
        value = snap.readings[room]
        st.metric("Room Signal", "Unavailable" if value is None else f"{value:.3f}")
        linked = sorted(build_facility().neighbors(room))
        st.caption("Connected to: " + ", ".join(zone_label(z) for z in linked))
        if snap.requested_zone == room:
            st.caption("This room received the additional measurement.")
        if st.session_state.demo_reveal:
            st.metric("True source", zone_label(snap.true_source))
            error = baseline_error(snap)
            st.caption(f"Baseline graph-distance error: {error if error is not None else 'N/A'} edges")
        st.markdown("**Maximum-Sensor-Reading (Argmax) Algorithm**")
        st.caption(
            "A deterministic heuristic: predict the zone with the highest "
            "available sensor value. It does not use model training."
        )
        st.markdown(r"$\hat{s} = \underset{i \in O}{\operatorname{arg\,max}}\; r_i$")
        st.caption("O = observed zones; rᵢ = the reading at zone i.")

    st.markdown("#### 2. Source Predictions")
    st.write(
        "**Argmax** selects the strongest available sensor. **Logistic Regression** "
        "and **Random Forest** learn from observed values and missingness; their "
        "robust versions also train on deliberately sparse and noisy views. "
        "**Graph-Aware Likelihood** scores all eight candidate sources using "
        "the known simulated signal-decay and noise settings."
    )
    st.caption(
        "Graph-Aware Likelihood has additional simulator-parameter information "
        "not provided to the trained models. Compare its results with that advantage in mind."
    )
    if not st.session_state.get("ml_enabled", False):
        if st.button("Train and Compare Models", key="demo_train_models", type="primary"):
            try:
                load_ml_artifacts()
                st.session_state.ml_enabled = True
                st.rerun()
            except (FileNotFoundError, ValueError) as error:
                st.error(f"Model training unavailable: {error}")
    else:
        try:
            artifacts = load_ml_artifacts()
            predictions = all_current_predictions(artifacts, snap)
            predictions = predictions.rename(columns={
                "model": "Model", "predicted_source": "Predicted Source",
                "max_class_probability": "Top Model Probability",
            })
            predictions["Predicted Source"] = predictions["Predicted Source"].map(zone_label)
            predictions["Top Model Probability"] = predictions["Top Model Probability"].map(
                lambda p: "N/A" if pd.isna(p) else f"{p:.1%}"
            )
            st.dataframe(predictions, hide_index=True, use_container_width=True)
            st.caption(
                "Original = standard training; unmarked ML = sparse/noise-augmented training. "
                "Probabilities are model-dependent estimates, not guarantees of correctness."
            )
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Model comparison unavailable: {error}")

    st.markdown("#### 3. Adaptive Sensor Selection")
    st.caption(
        "The information-gain policy estimates which currently unobserved zone is most informative. "
        "The new reading is revealed only after the model selects the zone."
    )
    sample_left, sample_right = st.columns([1.4, 1])
    with sample_left:
        demo_selector = st.selectbox(
            "Sampling model", ["Logistic Regression", "Random Forest", GRAPH_MODEL_NAME],
            key="demo_selector",
        )
        candidates = [zone for zone, reading in snap.readings.items() if reading is None]
        if snap.requested_zone is None and candidates:
            if st.button("Select and Measure Next Zone", type="primary", key="demo_sample"):
                try:
                    prior = sampling_prior(demo_selector, snap)
                    choice = choose_next_zone(
                        build_facility(), snap.readings, prior,
                        source_intensity=params.source_intensity,
                        decay_rate=params.decay_rate,
                    )
                    if choice is not None:
                        st.session_state.active_snapshot = sample_one_more(snap, choice.zone)
                        if demo_selector != GRAPH_MODEL_NAME:
                            st.session_state.ml_enabled = True
                        st.session_state.auto_sample_details = {
                            "model": demo_selector, "zone": choice.zone,
                            "gain": choice.expected_gain_bits,
                        }
                        st.rerun()
                except (FileNotFoundError, ValueError) as error:
                    st.error(f"Adaptive sampling unavailable: {error}")
        elif snap.requested_zone is not None:
            st.success(f"Additional reading at {zone_label(snap.requested_zone)}: {snap.requested_reading:.3f}")
            if st.button("Reset this measurement", key="demo_reset_sample"):
                st.session_state.active_snapshot = simulate(params)
                st.session_state.pop("auto_sample_details", None)
                st.rerun()
        else:
            st.info("No unobserved zones remain.")
    with sample_right:
        if snap.requested_zone is not None:
            details = st.session_state.get("auto_sample_details")
            if details:
                st.metric("Selected zone", zone_label(details["zone"]))
                st.metric("Estimated uncertainty reduction", f"{details['gain']:.3f} bits")
            st.caption("A change in this individual prediction does not by itself prove better aggregate accuracy.")
        else:
            st.caption("The selector has not yet requested an extra measurement.")

    if snap.requested_zone is not None:
        try:
            previous = simulate(params)
            initial_graph, _ = graph_for_snapshot(previous)
            updated_graph, _ = graph_for_snapshot(snap)
            rows = [
                {"Model": ARGMAX_LABEL, "Before": snap.initial_prediction,
                 "After": predict_baseline(snap.readings)},
                {"Model": GRAPH_MODEL_NAME, "Before": initial_graph,
                 "After": updated_graph},
            ]
            if st.session_state.get("ml_enabled", False):
                trained_models = load_ml_artifacts()
                initial = all_current_predictions(trained_models, previous)
                updated = all_current_predictions(trained_models, snap)
                original = initial[["model", "predicted_source"]].merge(
                    updated[["model", "predicted_source"]],
                    on="model", suffixes=("_before", "_after"),
                )
                rows.extend({
                    "Model": r["model"], "Before": r["predicted_source_before"],
                    "After": r["predicted_source_after"],
                } for _, r in original.iterrows() if r["model"] != GRAPH_MODEL_NAME)
            comparison = display_zone_columns(pd.DataFrame(rows), ["Before", "After"])
            st.markdown("**Source Predictions Before and After Sampling**")
            st.dataframe(comparison, hide_index=True, use_container_width=True)
        except (FileNotFoundError, ValueError) as error:
            st.error(f"Prediction comparison unavailable: {error}")

    st.markdown("#### 4. Results Across Repeated Simulations")
    summary_path = ROOT / "Results" / "Demo" / "sampling_summary.csv"
    if summary_path.is_file():
        saved = pd.read_csv(summary_path)
        required = {"condition", "model", "strategy", "trials", "accuracy",
                    "mean_graph_error", "mean_additional_samples"}
        if not required.issubset(saved):
            st.error("Evaluation results could not be displayed because the saved summary is incomplete.")
        else:
            selected_condition = st.selectbox(
                "Evaluation condition", sorted(saved["condition"].unique()),
                key="demo_result_condition",
            )
            subset = saved[saved["condition"] == selected_condition].copy()
            methods = sorted(subset["model"].unique())
            preferred = ["Logistic Regression", "Random Forest", GRAPH_MODEL_NAME]
            shown_methods = st.multiselect(
                "Methods shown in chart and table", methods,
                default=[name for name in preferred if name in methods],
                key="demo_results_models",
            )
            if GRAPH_MODEL_NAME not in methods:
                st.info(
                    "Saved evaluations predate the new graph-aware method. "
                    "Recalculate the repeated trials to compare the upgraded system."
                )
            subset = subset[subset["model"].isin(shown_methods)].copy()
            subset["strategy"] = subset["strategy"].map({
                "static": "No Additional Sample",
                "random": "Random Extra Sample",
                "information_gain": "Information-Gain Sample",
            })
            fig = go.Figure()
            for model_name, group in subset.groupby("model", sort=False):
                fig.add_trace(go.Bar(
                    name=model_name, x=group["strategy"],
                    y=group["accuracy"] * 100,
                    text=group["accuracy"].map(lambda v: f"{v:.1%}"),
                    textposition="outside",
                    cliponaxis=False,
                    textfont=dict(size=13),
                ))
            fig.update_layout(
                title=f"Measured accuracy: {selected_condition}",
                barmode="group", height=440,
                yaxis=dict(range=[0, 115], title="Exact-source accuracy (%)",
                           gridcolor="#33475D", dtick=20),
                margin=dict(t=70, l=65, r=30, b=110),
                legend=dict(orientation="h", x=0, y=-0.32),
                paper_bgcolor=PANEL, plot_bgcolor=PANEL,
                font=dict(color=MUTED),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Both random and information-gain strategies use one additional measurement; "
                "the static strategy uses none. Results are based on repeated synthetic trials."
            )
            metrics_table = subset[[
                "model", "strategy", "trials", "accuracy",
                "mean_graph_error", "mean_additional_samples",
            ]].copy()
            metrics_table["accuracy"] = metrics_table["accuracy"].map(lambda v: f"{v:.1%}")
            metrics_table["mean_graph_error"] = metrics_table["mean_graph_error"].map(
                lambda v: "N/A" if pd.isna(v) else f"{v:.2f}"
            )
            metrics_table["mean_additional_samples"] = metrics_table["mean_additional_samples"].map(
                lambda v: f"{v:.1f}"
            )
            metrics_table.columns = [
                "Model", "Sampling Strategy", "Trials", "Accuracy",
                "Mean Graph Error (Edges)", "Additional Measurements",
            ]
            if "missing_source_accuracy" in subset:
                metrics_table["Source Unobserved Accuracy"] = subset[
                    "missing_source_accuracy"
                ].map(lambda v: "N/A" if pd.isna(v) else f"{v:.1%}")
            st.dataframe(metrics_table, hide_index=True, use_container_width=True)
            st.caption(
                "Source Unobserved Accuracy measures trials where the true source's "
                "initial sensor reading was missing. All results use seeded independent "
                "synthetic events; graph likelihood assumes known simulator parameters."
            )
    else:
        st.info("Repeated-trial evaluation results are not available for display.")


with live:
    st.markdown('<div class="aps-notice">The green ring indicates the Argmax Baseline prediction: the zone with the highest observed sensor reading. The purple ring identifies the additional sample.</div>', unsafe_allow_html=True)
    st.write("")
    prediction = predict_baseline(snap.readings)
    graph_guess, _ = graph_for_snapshot(snap)
    metrics = st.columns(5)
    metrics[0].metric("Available readings", f"{sum(x is not None for x in snap.readings.values())}/8")
    metrics[1].metric(ARGMAX_LABEL, zone_label(prediction))
    metrics[2].metric("Graph Likelihood", zone_label(graph_guess))
    metrics[3].metric("Sensor noise", f"{params.noise_std:.2f}")
    metrics[4].metric("Missing probability", f"{params.missing_rate:.0%}")
    if st.session_state.get("ml_enabled", False):
        try:
            live_ml_predictions = predict_models(load_ml_artifacts(), snap.readings)
            ml_cols = st.columns(2)
            for column, (_, result) in zip(ml_cols, live_ml_predictions.iterrows()):
                zone = result["predicted_source"]
                column.metric(result["model"], zone_label(zone))
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"ML model integration: {exc}")
    l, r = st.columns([1.75, 1], gap="large")
    with l:
        st.subheader("Eight-Zone Facility")
        live_view = st.radio(
            "Facility visualization", ["3D Rooms", "2D Network"],
            key="live_facility_view", horizontal=True,
        )
        st.toggle("Reveal hidden source", key="reveal_truth")
        if live_view == "3D Rooms":
            live_figure = facility_3d_plot(
                snap, build_facility(), predict_baseline(snap.readings),
                show_source=st.session_state.reveal_truth,
                selected_zone=st.session_state.live_selected_zone,
            )
        else:
            live_figure = facility_plot(snap, st.session_state.reveal_truth)
        st.plotly_chart(
            live_figure, use_container_width=True,
            config={"displaylogo": False, "scrollZoom": True},
        )
        if live_view == "3D Rooms":
            st.caption(
                "Rotate and zoom the facility. Blue = low, amber = high, gray = unavailable. "
                "Green = Argmax prediction, purple = additional sample, gold = revealed source."
            )
        else:
            st.caption("Graph edges show connectivity, not physical distances or airflow.")
    with r:
        st.subheader("Sensor Readings")
        room = st.selectbox(
            "Inspect a Room", sorted(snap.readings),
            key="live_selected_zone", format_func=zone_label,
        )
        room_reading = snap.readings[room]
        st.caption(
            f"{zone_label(room)}: "
            + ("No available reading" if room_reading is None else f"Signal {room_reading:.3f}")
            + " | Connected to: "
            + ", ".join(zone_label(z) for z in sorted(build_facility().neighbors(room)))
        )
        st.plotly_chart(reading_plot(snap), use_container_width=True,
                        config={"displaylogo": False})
        st.subheader("One Additional Measurement")
        st.caption("Compare a manual additional sample with model-assisted sampling that selects a currently unobserved zone.")
        candidates = sorted([z for z, value in snap.readings.items() if value is None])
        if snap.requested_zone is None and candidates:
            target = st.selectbox("Sample location", candidates, key="manual_zone", format_func=zone_label)
            if st.button("Request manual sample", use_container_width=True):
                st.session_state.pop("auto_sample_details", None)
                st.session_state.active_snapshot = sample_one_more(snap, target)
                st.rerun()
            selector_model = st.selectbox(
                "Information-gain selector", ["Logistic Regression", "Random Forest", GRAPH_MODEL_NAME],
                key="adaptive_selector_model",
            )
            if st.button("AI selects next zone", type="primary", use_container_width=True):
                try:
                    prior = sampling_prior(selector_model, snap)
                    choice = choose_next_zone(
                        build_facility(), snap.readings, prior,
                        source_intensity=params.source_intensity,
                        decay_rate=params.decay_rate,
                    )
                    if choice is not None:
                        st.session_state.active_snapshot = sample_one_more(snap, choice.zone)
                        st.session_state.auto_sample_details = {
                            "model": selector_model,
                            "zone": choice.zone,
                            "gain": choice.expected_gain_bits,
                        }
                        if selector_model != GRAPH_MODEL_NAME:
                            st.session_state.ml_enabled = True
                        st.rerun()
                except (FileNotFoundError, ValueError) as error:
                    st.error(f"Adaptive selection failed: {error}")
        elif snap.requested_zone:
            st.success(f"New reading at {zone_label(snap.requested_zone)}: {snap.requested_reading:.3f}")
            selected = st.session_state.get("auto_sample_details")
            if selected:
                st.caption(
                    f"{selected['model']} selected {zone_label(selected['zone'])} "
                    f"(estimated information gain: {selected['gain']:.3f} bits)."
                )
            if st.button("Reset sampling", use_container_width=True):
                st.session_state.active_snapshot = simulate(params)
                st.session_state.pop("auto_sample_details", None)
                st.rerun()
        else:
            st.info("No unavailable zone remains to sample.")
        if st.session_state.reveal_truth:
            before, after = baseline_error(snap, after_sample=False), baseline_error(snap)
            st.write(f"True source: {zone_label(snap.true_source)}")
            st.write(f"Initial Argmax: {zone_label(snap.initial_prediction)} | error: {before if before is not None else 'N/A'}")
            st.write(f"Current Argmax: {zone_label(prediction)} | error: {after if after is not None else 'N/A'}")
    table = pd.DataFrame([
        {"zone": z, "initially_selected": z in snap.initially_selected, "reading": snap.readings[z]}
        for z in sorted(snap.readings)
    ])
    with st.expander("View and export current sensor data"):
        shown_readings = table.rename(columns={
            "zone": "Zone", "initially_selected": "Initially Selected", "reading": "Signal Reading"
        })
        shown_readings["Zone"] = shown_readings["Zone"].map(zone_label)
        st.dataframe(shown_readings, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", table.to_csv(index=False).encode(), file_name="aps_sensor_readings.csv", mime="text/csv")

with models_tab:
    st.subheader("Machine-Learning Source Localization")
    st.write(
        "Compare original Logistic Regression and Random Forest against versions trained "
        "with additional sparse/noisy views of Justin's **training partition only**. "
        "Their accuracy is measured on the identical untouched held-out test partition."
    )
    st.caption(
        "Graph-Aware Likelihood is a parameter-aware model of the simulated signal physics, "
        "not a trained ML classifier. It knows source intensity, distance decay, and sensor "
        "noise, which the data-only classifiers do not."
    )
    if st.button("Train models and evaluate held-out test data", type="primary"):
        st.session_state.ml_enabled = True
    if not st.session_state.get("ml_enabled", False):
        st.info("Compare Logistic Regression and Random Forest against the rule-based Maximum-Sensor-Reading (Argmax) Baseline.")
    else:
        try:
            trained = load_ml_artifacts()
            st.caption(
                f"Dataset: {Path(trained.source_path).name} | "
                f"Training: {trained.train_count:,} independent scenarios | "
                f"Augmented training views: {trained.augmented_train_count:,} | "
                f"Held-out test: {trained.test_count:,} untouched scenarios"
            )
            scores = trained.scores.copy()
            shown_scores = scores.copy()
            shown_scores["model"] = shown_scores["model"].map(method_label)
            shown_scores["accuracy"] = shown_scores["accuracy"].map(lambda x: f"{x:.1%}")
            shown_scores["mean_graph_error"] = shown_scores["mean_graph_error"].map(
                lambda x: "N/A" if pd.isna(x) else f"{x:.3f}"
            )
            shown_scores.columns = ["Model", "Exact accuracy", "Mean graph error (edges)", "Test scenarios"]
            st.dataframe(shown_scores, hide_index=True, use_container_width=True)
            fig = go.Figure(go.Bar(
                x=scores["accuracy"] * 100,
                y=scores["model"].map(method_label),
                orientation="h",
                marker_color=[
                    "#7389A5" if "(Original)" in name else
                    "#EDB978" if name == GRAPH_MODEL_NAME else
                    CYAN if name in ("Logistic Regression", "Random Forest") else GREEN
                    for name in scores["model"]
                ],
                text=[f"{v:.1%}" for v in scores["accuracy"]],
                textposition="outside",
                cliponaxis=False,
                textfont=dict(size=13),
            ))
            fig.update_layout(
                title="Held-out exact-source accuracy",
                xaxis=dict(title="Accuracy (%)", range=[0, 115], gridcolor="#26384f", dtick=20),
                yaxis=dict(autorange="reversed"),
                paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTED),
                height=440, margin=dict(t=70, l=285, r=45, b=65), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Predictions on the Current Scenario")
            current_predictions = all_current_predictions(trained, snap)
            initial_predictions = all_current_predictions(trained, simulate(params))
            display_predictions = current_predictions.copy()
            display_predictions["max_class_probability"] = display_predictions["max_class_probability"].map(
                lambda p: "N/A" if pd.isna(p) else f"{p:.1%}"
            )
            display_predictions.columns = ["Model", "Predicted Source", "Top Model Probability (Not Guaranteed)"]
            display_predictions["Predicted Source"] = display_predictions["Predicted Source"].map(zone_label)
            st.dataframe(display_predictions, hide_index=True, use_container_width=True)
            if snap.requested_zone is not None:
                before_after = initial_predictions[["model", "predicted_source"]].merge(
                    current_predictions[["model", "predicted_source"]],
                    on="model", suffixes=("_before", "_after")
                )
                before_after.columns = ["Model", "Before Additional Sample", "After Additional Sample"]
                before_after = display_zone_columns(before_after, ["Before Additional Sample", "After Additional Sample"])
                st.caption(f"Prediction comparison after sampling {zone_label(snap.requested_zone)}")
                st.dataframe(before_after, hide_index=True, use_container_width=True)
            by_condition = trained.test_predictions.groupby(
                ["scenario_type", "model"]
            )["correct"].mean().reset_index()
            condition_fig = go.Figure()
            for model_name, data in by_condition.groupby("model", sort=False):
                condition_fig.add_trace(go.Scatter(
                    name=method_label(model_name),
                    mode="lines+markers",
                    x=data["scenario_type"].map(condition_label),
                    y=data["correct"] * 100,
                    marker=dict(size=8),
                ))
            condition_fig.update_layout(
                title="Held-out accuracy by synthetic condition",
                yaxis=dict(range=[0, 105], title="Accuracy (%)", gridcolor="#26384f"),
                paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTED),
                height=470, margin=dict(t=65, l=55, r=45, b=130),
                legend=dict(orientation="h", y=-0.46, x=0),
            )
            st.plotly_chart(condition_fig, use_container_width=True)
            st.download_button(
                "Download held-out prediction records",
                trained.test_predictions.to_csv(index=False).encode("utf-8"),
                file_name="aps_holdout_model_predictions.csv", mime="text/csv"
            )
            st.caption(
                "Original and augmented classifiers share the same original training IDs "
                "and are evaluated on unseen test IDs. More training views do not guarantee "
                "better generalization. Graph likelihood receives the known simulator "
                "parameter settings (not the hidden source), giving it extra information. "
                "Class probabilities are not guaranteed to be calibrated."
            )
        except (FileNotFoundError, ValueError) as exc:
            st.error(f"Unable to train or evaluate models: {exc}")

with adaptive_tab:
    st.subheader("Does Intelligent Sensor Selection Actually Help?")
    st.write(
        "Compare no additional measurement, one random measurement, and "
        "one model-selected measurement on exactly the same seeded scenarios. "
        "Both sampling strategies are allowed one extra measurement."
    )
    c_trials, c_method = st.columns([1, 2])
    with c_trials:
        experiment_trials = st.slider(
            "Matched simulated events", min_value=20, max_value=200,
            value=60, step=20, key="adaptive_trial_count",
        )
    with c_method:
        selected_models = st.multiselect(
            "Models to evaluate",
            ["Logistic Regression", "Random Forest"],
            default=["Logistic Regression", "Random Forest"],
            key="adaptive_evaluation_models",
        )
    if st.button("Run matched sampling experiment", type="primary"):
        if not selected_models:
            st.error("Select at least one model.")
        else:
            try:
                st.session_state.adaptive_results = compare_sampling_policies(
                    load_ml_artifacts(), params,
                    trials=experiment_trials,
                    model_names=tuple(selected_models),
                )
            except (FileNotFoundError, ValueError) as error:
                st.error(f"Adaptive experiment failed: {error}")
    matched = st.session_state.get("adaptive_results")
    if matched is None:
        st.info("Run the experiment to generate paired comparisons and accuracy charts.")
    else:
        summary = summarize_comparison(matched)
        results_view = summary.copy()
        results_view["accuracy"] = results_view["accuracy"].map(lambda a: f"{a:.1%}")
        results_view["mean_graph_error"] = results_view["mean_graph_error"].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.3f}"
        )
        results_view["additional_samples"] = results_view["additional_samples"].map(
            lambda x: f"{x:.1f}"
        )
        results_view.columns = [
            "Model", "Sampling strategy", "Trials", "Exact-source accuracy",
            "Mean graph error (edges)", "Extra samples per run",
        ]
        st.dataframe(results_view, hide_index=True, use_container_width=True)
        fig = go.Figure()
        label_map = {
            "static": "No additional sample",
            "random": "Random selection",
            "information_gain": "Expected information gain",
        }
        for name, group in summary.groupby("model", sort=False):
            fig.add_trace(go.Bar(
                name=name,
                x=[label_map[x] for x in group["strategy"]],
                y=group["accuracy"] * 100,
                text=[f"{a:.1%}" for a in group["accuracy"]],
                textposition="outside",
                cliponaxis=False,
                textfont=dict(size=13),
            ))
        fig.update_layout(
            barmode="group", title="Accuracy on matched simulated events",
            yaxis=dict(title="Exact-source accuracy (%)", range=[0, 115],
                       gridcolor="#27394e", dtick=20),
            paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            font=dict(color=MUTED), height=415,
            margin=dict(t=75, l=58, r=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)
        for name, group in summary.groupby("model", sort=False):
            rates = dict(zip(group["strategy"], group["accuracy"]))
            st.caption(
                f"{name}: information gain vs. random = "
                f"{100 * (rates['information_gain'] - rates['random']):+.1f} "
                f"percentage points. A negative value means random sampling did better."
            )
        st.download_button(
            "Download paired experiment CSV",
            matched.to_csv(index=False).encode("utf-8"),
            file_name="aps_sampling_policy_comparison.csv", mime="text/csv",
        )
        with st.expander("Inspect all paired predictions"):
            display_matched = matched.rename(columns={
                "run_id": "Trial", "seed": "Seed", "model": "Model", "strategy": "Sampling Strategy",
                "true_source": "True Source", "predicted_source": "Predicted Source",
                "correct": "Correct", "graph_distance": "Graph Error",
                "sampled_zone": "Sampled Zone", "extra_samples": "Additional Samples",
                "expected_gain_bits": "Expected Gain (bits)", "noise_std": "Noise SD",
                "missing_rate": "Missing Probability", "initial_sensors": "Initial Sensors",
                "source_intensity": "Source Intensity", "decay_rate": "Distance Decay",
            })
            display_matched = display_zone_columns(
                display_matched, ["True Source", "Predicted Source", "Sampled Zone"]
            )
            display_matched["Sampling Strategy"] = display_matched["Sampling Strategy"].map(condition_label)
            st.dataframe(display_matched, hide_index=True, use_container_width=True)
    st.caption(
        "This is a controlled spatial snapshot simulation. Source truth is used "
        "only for scoring. The information-gain policy uses model probabilities "
        "that are not calibrated and assumes the simulator's graph-distance "
        "signal model and sampling noise are known. It is not guaranteed to "
        "outperform random sampling."
    )


with batch_tab:
    st.subheader("Argmax Baseline: Repeated Experiments")
    st.write("Evaluate the Maximum-Sensor-Reading (Argmax) Baseline on independent simulated events. Each prediction selects the zone with the highest available reading. This rule uses no training data.")
    n = st.slider("Trials", 20, 300, 100, 20)
    if st.button("Run Argmax Trials", type="primary"):
        st.session_state.batch_results = pd.DataFrame(batch_baseline(params, n=n))
    results = st.session_state.get("batch_results")
    if results is None:
        st.info("Run an experiment to populate the charts and metrics.")
    else:
        accuracy = results["correct"].mean()
        graph_error = results["graph_distance"].mean()
        missing = results["predicted_source"].isna().sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Exact-source accuracy", f"{accuracy:.1%}")
        c2.metric("Mean graph error", f"{graph_error:.2f} edges" if pd.notna(graph_error) else "N/A")
        c3.metric("No-reading trials", f"{missing}/{len(results)}")
        summary = results.groupby("true_source").agg(accuracy=("correct", "mean"), trials=("correct", "size")).reset_index()
        fig = go.Figure(go.Bar(
            x=summary["true_source"].map(zone_label), y=summary["accuracy"] * 100, marker_color=CYAN,
            customdata=summary["trials"],
            hovertemplate="%{x}: %{y:.1f}% (n=%{customdata})<extra></extra>",
        ))
        fig.update_layout(title="Argmax Baseline Accuracy by True Source", height=345,
                          paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTED),
                          yaxis=dict(range=[0, 110], title="Accuracy (%)", gridcolor="#26384f"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("All trials count toward accuracy, including abstentions as incorrect. Graph-distance error excludes abstentions.")
        st.download_button("Download trial records", results.to_csv(index=False).encode(),
                           file_name="aps_baseline_trials.csv", mime="text/csv")
        with st.expander("Inspect trial records"):
            shown_baseline = results.rename(columns={
                "run_id": "Trial", "seed": "Seed", "true_source": "True Source",
                "predicted_source": "Predicted Source", "correct": "Correct",
                "graph_distance": "Graph Error", "observed_count": "Available Readings",
                "initial_sensors": "Initially Sampled", "noise_std": "Noise SD",
                "missing_rate": "Missing Probability",
                "source_intensity": "Source Intensity", "decay_rate": "Distance Decay",
            })
            shown_baseline = display_zone_columns(shown_baseline, ["True Source", "Predicted Source"])
            st.dataframe(shown_baseline, use_container_width=True, hide_index=True)

with about:
    st.subheader("Research Question")
    st.markdown(
        "**Can machine learning identify the hidden source of a simulated contamination "
        "event using noisy or missing sensor readings, and can one strategically selected "
        "additional measurement improve source localization?**"
    )
    st.subheader("Localization Algorithms")
    st.markdown(
        "**1. Maximum-Sensor-Reading (Argmax) Baseline:** A deterministic "
        "reference algorithm selecting the location of the highest observed signal. "
        "There is no training step.\n"
        "**2. Logistic Regression:** A supervised multiclass classifier trained "
        "to recognize source locations from eight readings and eight missingness flags.\n"
        "**3. Random Forest:** A supervised ensemble of decision trees trained "
        "on the same inputs for a fair comparison."
    )
    st.markdown(r"**Argmax prediction:** $\hat{s} = \underset{i \in O}{\operatorname{arg\,max}}\; r_i$")
    st.subheader("Experimental Method")
    st.markdown(
        "1. **Simulate:** Assign a hidden source to one of eight connected zones. "
        "Signals decrease with graph distance, with adjustable noise and missing observations.\n"
        "2. **Predict:** Compare the Maximum-Sensor-Reading (Argmax) Baseline against Logistic Regression "
        "and Random Forest trained on synthetic observations.\n"
        "3. **Sample:** Select one extra measurement at random or at the zone with "
        "the highest estimated information gain. Both strategies use the same budget.\n"
        "4. **Evaluate:** Measure exact-source accuracy and shortest-path error "
        "against the simulator's known ground truth over repeated events."
    )
    st.subheader("Interpretation and Limitations")
    st.write(
        "The network is an idealized spatial graph, not a real building or airflow simulation. "
        "Each event represents a single spatial snapshot, not contamination spreading through time."
    )
    st.write(
        "Higher expected information gain does not guarantee higher source-identification accuracy. "
        "We compare the results empirically, and the classifiers' probabilities have not been calibrated."
    )
