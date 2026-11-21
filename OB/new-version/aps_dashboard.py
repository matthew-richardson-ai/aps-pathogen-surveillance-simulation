"""Run from the repository root: python -m streamlit run aps_dashboard.py"""
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
st.set_page_config(page_title="APS facility", layout="wide")
st.title("Facility surveillance")

upload = st.file_uploader("Open a simulation dataset", type="csv")
if upload is None:
    st.info("Select one generated CSV to explore its facility snapshots and baseline results.")
    st.stop()
try:
    df = pd.read_csv(upload)
    zones = [c[:-8] for c in df if c.startswith("zone_") and c.endswith("_reading")]
    if not zones or df.empty:
        raise ValueError("The CSV needs at least one snapshot and zone reading.")
    truth = df["true_source_node"]
    if not truth.isin(zones).all():
        raise ValueError("A source label does not match the zone columns.")
    readings = pd.DataFrame(index=df.index)
    for z in zones:
        flags = pd.to_numeric(df[z + "_missing"], errors="raise")
        values = pd.to_numeric(df[z + "_reading"], errors="raise")
        if not flags.isin([0, 1]).all() or not values.dropna().between(0, 1).all():
            raise ValueError("Expected readings from 0 to 1 and binary missing flags.")
        readings[z] = values.mask(flags.eq(1))
except (ValueError, KeyError, pd.errors.ParserError) as exc:
    st.error(f"Unable to load this dataset. {exc}")
    st.stop()

try:
    from Simulation.facility import build_facility
    graph = build_facility()
    if set(graph.nodes) != set(zones):
        raise ValueError("Zone mismatch")
    topology = True
except (ImportError, ValueError):
    graph = nx.empty_graph(zones)
    topology = False

default = nx.spring_layout(graph, seed=42) if topology else nx.circular_layout(graph)
coords = pd.DataFrame.from_dict(default, orient="index", columns=["x", "y"]).loc[zones]
st.caption(f"{upload.name} · {len(df):,} snapshots · {len(zones)} zones")
facility, evaluation = st.tabs(["Facility", "Evaluation"])

def present(fig, filename):
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=180, bbox_inches="tight")
    st.pyplot(fig)
    st.download_button("Download graphic", output.getvalue(), filename, "image/png")
    plt.close(fig)

with facility:
    row = int(st.number_input("Snapshot number", 1, len(df), 1)) - 1
    with st.expander("Edit layout"):
        st.caption("Change each zone’s horizontal and vertical position. Connections come from Simulation/facility.py.")
        layout = st.data_editor(coords, disabled=["_index"], key="layout_" + ",".join(zones))
    if not topology:
        st.caption("Schematic positions only. Facility connections are unavailable.")
    pos = {z: layout.loc[z].to_numpy(float) for z in zones}
    if not np.isfinite(list(pos.values())).all():
        st.error("Enter a numeric x and y for every zone.")
    else:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="#a7b9bc", width=2)
        cmap = plt.get_cmap("Blues")
        for z, value in readings.iloc[row].items():
            color = "#dddeda" if pd.isna(value) else cmap(0.2 + 0.7 * value)
            ax.scatter(*pos[z], s=3000, color=color, edgecolors="#bb581c" if z == truth.iloc[row] else "#617680",
                       linewidths=3 if z == truth.iloc[row] else 1, zorder=3)
            label = "missing" if pd.isna(value) else f"{value:.2f}"
            ax.text(*pos[z], z.replace("_", " ") + "\n" + label, ha="center", va="center",
                    color="white" if pd.notna(value) and value > 0.64 else "#152532", zorder=4)
        ax.margins(.23)
        ax.set_axis_off()
        ax.set_title(f"Snapshot {row+1}   |   Recorded source: {truth.iloc[row]}")
        fig.tight_layout()
        present(fig, "facility.png")

with evaluation:
    index = df.index
    if "split" in df:
        splits = df["split"].dropna().unique().tolist()
        if splits:
            selected = st.selectbox("Dataset split", splits, index=splits.index("test") if "test" in splits else 0)
            index = df.index[df["split"].eq(selected)]
    valid = readings.loc[index].dropna(how="all")
    predicted = valid.idxmax(axis=1)
    correct = predicted.eq(truth.loc[valid.index]).reindex(index, fill_value=False)
    st.subheader("Strongest-sensor baseline")
    st.caption("The highest available reading predicts the source. Ties use CSV zone order. Runs with all readings missing count as incorrect.")
    a, b, c = st.columns(3)
    a.metric("Exact-source accuracy", f"{correct.mean():.1%}")
    b.metric("Correct / evaluated", f"{correct.sum():,} / {len(index):,}")
    c.metric("No prediction", f"{len(index)-len(valid):,}")
    counts = readings.loc[index].isna().sum(axis=1)
    grouped = correct.groupby(counts).agg(["mean", "size"])
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    left.bar(grouped.index, grouped["mean"] * 100, color="#227979")
    left.set(xlabel="Missing readings in snapshot", ylabel="Exact-source accuracy (%)",
             ylim=(0, 115), xticks=grouped.index, title="Accuracy by missing readings")
    for n, stat in grouped.iterrows():
        left.text(n, stat["mean"]*100+2, f'n={int(stat["size"])}', ha="center", fontsize=8)
    labels = zones + ["No prediction"]
    matrix = pd.crosstab(truth.loc[index], predicted.reindex(index).fillna(labels[-1])).reindex(index=zones, columns=labels, fill_value=0)
    right.imshow(matrix, cmap="Blues", aspect="auto")
    short = [z.replace("zone_", "") for z in zones]
    right.set(xticks=range(len(labels)), xticklabels=short+["None"], yticks=range(len(zones)),
              yticklabels=short, xlabel="Predicted zone", ylabel="Recorded source zone", title="Prediction counts")
    for (i, j), value in np.ndenumerate(matrix.to_numpy()):
        if value:
            right.text(j, i, str(value), ha="center", va="center", fontsize=8,
                       color="white" if value > matrix.to_numpy().max()*.55 else "#172530")
    present(fig, "baseline_evaluation.png")
