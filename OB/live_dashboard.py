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

st.set_page_config(page_title="APS | Simulation Lab", page_icon="🧬", layout="wide")
BG, PANEL, MUTED, CYAN, GREEN, PURPLE = "#0b1220", "#111e30", "#b7cada", "#48d4de", "#66e4ad", "#a994ff"

st.markdown("""
<style>
.stApp {background: #0b1220; color: #edf4ff;}
[data-testid="stSidebar"] {background: #101b2b;}
[data-testid="stMetric"] {background: #15243a; border: 1px solid #263b53; border-radius: 12px; padding: 14px;}
.block-container {max-width: 1550px; padding-top: 1.25rem;}
.aps-tag {color:#48d4de;font-size:.77rem;font-weight:700;letter-spacing:.14em;}
.aps-notice {background:#152238;border:1px solid #293e55;border-left:3px solid #48d4de;
 border-radius:9px;padding:12px 14px;color:#c9d9e9;}
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
for key, value in {"seed": 42, "source": "Random", "preset": "Normal conditions", **PRESETS["Normal conditions"]}.items():
    st.session_state.setdefault(key, value)

def apply_preset():
    for key, value in PRESETS[st.session_state.preset].items():
        st.session_state[key] = value

with st.sidebar:
    st.markdown('<div class="aps-tag">APS / SCENARIO CONTROLS</div>', unsafe_allow_html=True)
    st.header("Simulation settings")
    st.selectbox("Scenario preset", list(PRESETS), key="preset", on_change=apply_preset)
    st.selectbox("Hidden source", ["Random"] + sorted(build_facility().nodes), key="source")
    st.number_input("Random seed", min_value=0, max_value=2_000_000_000, step=1, key="seed")
    st.slider("Initially sampled zones", 2, 7, step=1, key="initial_sensors")
    st.slider("Sensor noise (standard deviation)", 0.0, 0.40, step=0.01, key="noise_std")
    st.slider("Missing-reading probability", 0.0, 0.95, step=0.05, key="missing_rate")
    st.slider("Source intensity", 0.20, 1.00, step=0.05, key="source_intensity")
    st.slider("Distance decay", 0.10, 1.50, step=0.05, key="decay_rate")
    if st.button("Generate new seeded scenario", type="primary", use_container_width=True):
        st.session_state.seed += 1
        st.rerun()
    st.caption("Changing controls recomputes the simulation. The seed makes each run reproducible.")
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
        marker = dict(size=60, line=dict(color="#0b1220", width=2))
        if available:
            marker.update(color=[snapshot.readings[z] for z in zones],
                          colorscale=[[0, "#bbf4f4"], [.5, "#47d3dd"], [1, "#fc8663"]],
                          cmin=0, cmax=1, showscale=True,
                          colorbar=dict(title="Signal", thickness=12, len=.55, x=1.02))
        else:
            marker.update(color="#2a3b50", line=dict(color="#778ba0", width=2))
        fig.add_trace(go.Scatter(
            x=[POS[z][0] for z in zones], y=[POS[z][1] for z in zones],
            mode="markers+text", text=[z.replace("zone_", "Z") for z in zones],
            textposition="middle center", textfont=dict(size=15, color="#061422" if available else "#e3ecf7"),
            marker=marker, name="Measured" if available else "Unavailable",
            customdata=[[z, f"{snapshot.readings[z]:.3f}" if available else "not observed"] for z in zones],
            hovertemplate="%{customdata[0]}: %{customdata[1]}<extra></extra>",
        ))
    prediction = predict_baseline(snapshot.readings)
    if prediction:
        x, y = POS[prediction]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name="Baseline prediction",
            marker=dict(symbol="circle-open", size=78, line=dict(color=GREEN, width=6)),
            hovertemplate=f"Baseline prediction: {prediction}<extra></extra>",
        ))
    if snapshot.requested_zone:
        x, y = POS[snapshot.requested_zone]
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name="Manual extra sample",
            marker=dict(symbol="circle-open", size=89, line=dict(color=PURPLE, width=3)),
            hovertemplate=f"Extra sample: {snapshot.requested_zone}<extra></extra>",
        ))
    if reveal:
        x, y = POS[snapshot.true_source]
        fig.add_trace(go.Scatter(
            x=[x+.27], y=[y+.37], mode="markers", name="True source",
            marker=dict(symbol="star", size=22, color="#fff493"),
            hovertemplate=f"True source: {snapshot.true_source}<extra></extra>",
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
        x=[z.replace("zone_", "Z") for z in zones],
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

live, batch_tab, about = st.tabs(["Live facility", "Experiment lab", "Methods & next steps"])

with live:
    st.markdown('<div class="aps-notice">This dashboard runs the existing APS simulation. The green ring represents the strongest-sensor baseline, not a trained ML prediction. A single extra measurement can be requested manually.</div>', unsafe_allow_html=True)
    st.write("")
    prediction = predict_baseline(snap.readings)
    metrics = st.columns(4)
    metrics[0].metric("Available readings", f"{sum(x is not None for x in snap.readings.values())}/8")
    metrics[1].metric("Baseline predicts", prediction or "No reading")
    metrics[2].metric("Sensor noise", f"{params.noise_std:.2f}")
    metrics[3].metric("Missing probability", f"{params.missing_rate:.0%}")
    l, r = st.columns([1.75, 1], gap="large")
    with l:
        st.subheader("Eight-zone facility")
        st.toggle("Reveal hidden source", key="reveal_truth")
        st.plotly_chart(facility_plot(snap, st.session_state.reveal_truth), use_container_width=True,
                        config={"displaylogo": False, "scrollZoom": True})
        st.caption("Hover or zoom to inspect nodes. Edges are graph connections, not airflow paths or physical distances.")
    with r:
        st.subheader("Sensor readings")
        st.plotly_chart(reading_plot(snap), use_container_width=True,
                        config={"displaylogo": False})
        st.subheader("One additional measurement")
        st.caption("Choose an unavailable zone. This manual step demonstrates updated evidence; it is not the AI adaptive policy.")
        candidates = sorted([z for z, value in snap.readings.items() if value is None])
        if snap.requested_zone is None and candidates:
            target = st.selectbox("Sample location", candidates, key="manual_zone")
            if st.button("Request manual sample", type="primary", use_container_width=True):
                st.session_state.active_snapshot = sample_one_more(snap, target)
                st.rerun()
        elif snap.requested_zone:
            st.success(f"New reading at {snap.requested_zone}: {snap.requested_reading:.3f}")
            if st.button("Reset sampling", use_container_width=True):
                st.session_state.active_snapshot = simulate(params)
                st.rerun()
        else:
            st.info("No unavailable zone remains to sample.")
        if st.session_state.reveal_truth:
            before, after = baseline_error(snap, after_sample=False), baseline_error(snap)
            st.write(f"True source: {snap.true_source}")
            st.write(f"Initial baseline: {snap.initial_prediction or 'None'} | error: {before if before is not None else 'N/A'}")
            st.write(f"Current baseline: {prediction or 'None'} | error: {after if after is not None else 'N/A'}")
    table = pd.DataFrame([
        {"zone": z, "initially_selected": z in snap.initially_selected, "reading": snap.readings[z]}
        for z in sorted(snap.readings)
    ])
    with st.expander("View and export current sensor data"):
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", table.to_csv(index=False).encode(), file_name="aps_sensor_readings.csv", mime="text/csv")

with batch_tab:
    st.subheader("Baseline-only repeated experiments")
    st.write("Run independent seeded simulations at the current settings, with a randomized hidden source each time. These are genuine synthetic APS results, not estimates of ML-model accuracy.")
    n = st.slider("Trials", 20, 300, 100, 20)
    if st.button("Run baseline trials", type="primary"):
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
            x=summary["true_source"], y=summary["accuracy"] * 100, marker_color=CYAN,
            customdata=summary["trials"],
            hovertemplate="%{x}: %{y:.1f}% (n=%{customdata})<extra></extra>",
        ))
        fig.update_layout(title="Baseline accuracy by true source", height=345,
                          paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=MUTED),
                          yaxis=dict(range=[0, 105], title="Accuracy (%)", gridcolor="#26384f"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("All trials count toward accuracy, including abstentions as incorrect. Graph-distance error excludes abstentions.")
        st.download_button("Download trial records", results.to_csv(index=False).encode(),
                           file_name="aps_baseline_trials.csv", mime="text/csv")
        with st.expander("Inspect trial records"):
            st.dataframe(results, use_container_width=True, hide_index=True)

with about:
    st.subheader("What this demonstrator actually does")
    st.write("The facility, source selection, exponential graph-distance decay, noisy or missing readings, and baseline all come directly from APS core modules. Each display is a single spatial snapshot, not time-dependent propagation.")
    st.write("Implemented: interactive graph, simulated conditions, source reveal, manual follow-up sample, deterministic seeds, baseline metrics and trial CSV export.")
    st.write("Not yet implemented: ML predictions in this dashboard, calibrated confidence, AI-selected adaptive sampling, time-dependent transport, or production biosensing.")
    st.write("Next: connect Jimmy's trained models and Justin's shared dataset, implement information-based sampling, and feed matched predictions into Owen's model evaluation plots.")
