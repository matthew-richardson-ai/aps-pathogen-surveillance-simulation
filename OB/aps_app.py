"""APS visualization demo | Run: streamlit run aps_app.py

This interface is simulation-only. Example topology/readings are illustrative.
Model results are displayed ONLY when the user provides a result CSV.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from aps_viz import (
    accuracy_comparison, adaptive_gain, distance_comparison, draw_facility,
    example_facility, read_facility_json, sensitivity_heatmap,
    simulate_illustrative_readings, strongest_sensor, validate_runs,
)

st.set_page_config(page_title="APS | Simulation dashboard", page_icon="🔬", layout="wide")
st.title("Adaptive Pathogen Surveillance")
st.caption("CSC 4240 | Simulation-only facility source-localization visualization")
st.info("Prototype notice: no real biosensing, clinical diagnosis, or physical surveillance. "
        "The built-in facility and sensor values are **illustrative**, not your team's simulation output.")

with st.sidebar:
    st.header("Connect team data")
    graph_file = st.file_uploader("Optional 8-zone graph JSON", type="json", help="See sample_graph.json for the expected format")
    reading_file = st.file_uploader("Optional sensor readings CSV", type="csv", key="readings", help="Columns: zone, signal; empty signal = missing")
    results_file = st.file_uploader("Optional model results CSV", type="csv", key="results", help="See results_schema.csv")

try:
    graph, positions = read_facility_json(graph_file) if graph_file is not None else example_facility()
except (ValueError, KeyError, TypeError) as exc:
    st.error(f"Facility graph error: {exc}")
    st.stop()

tab_map, tab_eval, tab_notes = st.tabs(["Facility / sampling", "Model evaluation", "Demo notes"])

with tab_map:
    st.subheader("Simulated facility sensor overview")
    if graph_file is None:
        st.caption("EXAMPLE TOPOLOGY: replace it with the team's actual NetworkX edges for faithful output.")
    left, right = st.columns([1, 3])
    with left:
        source = st.selectbox("Hidden source (for simulation)", list(graph.nodes), index=3)
        reveal_truth = st.checkbox("Reveal true source marker", value=True)
        if reading_file is None:
            noise_sd = st.slider("Illustrative noise SD", 0.0, 0.5, 0.08, 0.01)
            missing_fraction = st.slider("Fraction missing", 0.0, 0.75, 0.125, 0.125)
            decay = st.slider("Distance decay", 0.1, 2.0, 0.9, 0.1)
            seed = st.number_input("Random seed", min_value=0, max_value=999_999, value=12, step=1)
            readings = simulate_illustrative_readings(graph, source,
                noise_sd=noise_sd, missing_fraction=missing_fraction,
                decay=decay, seed=int(seed))
        else:
            try:
                rd = pd.read_csv(reading_file)
                if not {"zone", "signal"}.issubset(rd.columns):
                    raise ValueError("Required columns are zone, signal")
                if rd["zone"].astype(str).duplicated().any():
                    raise ValueError("Each zone should appear at most once")
                observed = {str(x): y for x, y in zip(rd["zone"], pd.to_numeric(rd["signal"], errors="coerce"))}
                readings = {node: (None if pd.isna(observed.get(str(node))) else float(observed[str(node)])) for node in graph.nodes}
                if any(value is not None and not (0 <= value <= 1) for value in readings.values()):
                    raise ValueError("Signal must be normalized between 0 and 1")
            except (ValueError, KeyError) as exc:
                st.error(f"Readings CSV error: {exc}")
                st.stop()
        predicted = strongest_sensor(readings)
        st.metric("Strongest-sensor baseline", predicted or "No available readings")
        if reveal_truth and predicted:
            distance = __import__("networkx").shortest_path_length(graph, source, predicted)
            st.metric("Baseline graph error", f"{distance} edges")
    with right:
        fig = draw_facility(graph, positions, readings,
            true_source=source, predicted_source=predicted,
            show_truth=reveal_truth,
            title="APS | network graph and sensor state")
        st.pyplot(fig, clear_figure=True, use_container_width=True)
        plt.close(fig)
    st.caption("The green outline marks a strongest-sensor baseline, not a trained ML prediction. "
               "Any next-sample location must come from the team's adaptive strategy.")

with tab_eval:
    st.subheader("Performance from repeated simulation runs")
    if results_file is None:
        st.warning("Upload actual `results.csv` to populate evaluation charts. "
                   "No model accuracy numbers are assumed or invented.")
        st.markdown("**Required columns:** `run_id`, `model`, `mode`, `true_source`, `predicted_source`. "
                    "**Recommended additional columns:** `seed`, `confidence`, `noise_sd`, "
                    "`missing_fraction`, `sample_count`, `sampled_zone`.")
    else:
        try:
            runs = validate_runs(pd.read_csv(results_file), graph)
        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Results CSV error: {exc}")
            st.stop()
        st.caption(f"{len(runs):,} prediction records across {runs['run_id'].nunique():,} runs. "
                   "Accuracy and shortest-path error are recomputed from source IDs.")
        c1, c2 = st.columns(2)
        with c1:
            fig = accuracy_comparison(runs)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        with c2:
            fig = distance_comparison(runs)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        st.divider()
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Adaptive vs. static (same outbreak runs)**")
            try:
                fig, paired_summary = adaptive_gain(runs)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
                st.dataframe(paired_summary, hide_index=True)
            except ValueError as exc:
                st.info(str(exc))
        with c4:
            st.markdown("**Noise and missing-data sensitivity**")
            selected_model = st.selectbox("Model for sensitivity", sorted(runs['model'].unique()))
            selected_mode = st.selectbox("Sampling mode", sorted(runs['mode'].unique()))
            try:
                fig = sensitivity_heatmap(runs, model=selected_model, mode=selected_mode)
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            except ValueError as exc:
                st.info(str(exc))
        st.divider()
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("**Confusion matrix**")
            choices = sorted(runs['model'].unique())
            matrix_model = st.selectbox("Model for confusion matrix", choices)
            matrix_mode = st.selectbox("Mode for confusion matrix", sorted(runs['mode'].unique()), key="cm_mode")
            subset = runs[(runs.model == matrix_model) & (runs['mode'] == matrix_mode)]
            nodes = list(map(str, graph.nodes))
            matrix = pd.crosstab(subset['true_source'], subset['predicted_source'])
            matrix = matrix.reindex(index=nodes, columns=nodes, fill_value=0)
            fig, ax = plt.subplots(figsize=(5.8, 4.9), layout="constrained")
            im = ax.imshow(matrix.to_numpy(), cmap="Blues")
            ax.set_xticks(range(len(nodes)), nodes)
            ax.set_yticks(range(len(nodes)), nodes)
            ax.set_xlabel("Predicted source")
            ax.set_ylabel("True source")
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    ax.text(j, i, matrix.iloc[i, j], ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=.8)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
        with c6:
            st.markdown("**Confidence calibration**")
            if 'confidence' not in runs.columns or runs['confidence'].isna().all():
                st.info("Provide numeric confidence in [0, 1] for models that support it.")
            else:
                calib_model = st.selectbox("Model for calibration", sorted(runs['model'].unique()))
                chosen = runs[(runs.model == calib_model) & (runs['confidence'].between(0,1))].copy()
                if chosen.empty:
                    st.info("No valid confidence values for this model.")
                else:
                    chosen['confidence_bin'] = pd.cut(chosen['confidence'], bins=np.linspace(0,1,6), include_lowest=True)
                    summary = chosen.groupby('confidence_bin', observed=True).agg(
                        predicted=('confidence', 'mean'), observed=('correct', 'mean'), n=('correct', 'size')).reset_index()
                    fig, ax = plt.subplots(figsize=(5.8, 4.9), layout="constrained")
                    ax.plot([0,1], [0,1], linestyle='--', label="Perfect calibration")
                    ax.plot(summary['predicted'], summary['observed'], 'o-', label="Observed")
                    ax.set_xlim(0,1); ax.set_ylim(0,1)
                    ax.set_xlabel("Mean predicted confidence")
                    ax.set_ylabel("Fraction exactly correct")
                    ax.legend()
                    st.pyplot(fig, clear_figure=True)
                    plt.close(fig)
                    st.caption("Calibration bins computed using only uploaded predictions with confidence scores.")
        with st.expander("Inspect computed prediction records"):
            st.dataframe(runs, use_container_width=True)
            st.download_button("Download scored CSV", data=runs.to_csv(index=False).encode("utf-8"),
                file_name="aps_scored_results.csv", mime="text/csv")

with tab_notes:
    st.markdown("""
    ### Presentation walkthrough
    1. **Explain topology.** Nodes are the 8 simulated zones; edges are connectivity, not physical meters.
    2. **Run a noisy scenario.** Pick a hidden source, add missing readings, then highlight the strongest-sensor baseline.
    3. **Compare models.** Import the team's repeated test-run CSV to show Logistic Regression and Random Forest versus baseline.
    4. **Show adaptation.** Compare the *same* seeded outbreak scenarios before and after one additional sample.
    5. **Show limitations.** Compare accuracy under higher noise/missingness. This is a course simulation only.

    ### Evaluation requirements
    - Train/test partitions should use independent simulation runs; never use the held-out true source as an input feature.
    - Use identical held-out runs for methods being compared.
    - Pair static/adaptive rows using the same `run_id` and `model`. Static has no extra query; adaptive has at most one.
    - Show both number of trials and metric values; larger datasets reduce random variation.
    - Report sampling budget in addition to accuracy so any adaptive gains are interpreted fairly.

    ### Next integration points
    Replace the example graph with exported NetworkX nodes/edges; upload one row per simulated trial and model.
    Add the team's actual low-confidence sampling-selection algorithm for a next-sample marker.
    """)
