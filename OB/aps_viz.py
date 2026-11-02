"""Graph visualization + result plots for the APS simulation course project.

This file does not train models or infer a real pathogen source. All data used
by the standalone preview are explicitly illustrative.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Suitable for saved images and Streamlit.
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd


EXAMPLE_POS = {
    "Z1": (0.2, 1.55), "Z2": (1.45, 1.55), "Z3": (2.7, 1.55),
    "Z4": (3.85, 2.55), "Z5": (3.85, 0.55),
    "Z6": (5.20, 2.55), "Z7": (5.20, 0.55), "Z8": (6.35, 1.55),
}
EXAMPLE_EDGES = [
    ("Z1", "Z2"), ("Z2", "Z3"), ("Z3", "Z4"), ("Z3", "Z5"),
    ("Z4", "Z5"), ("Z4", "Z6"), ("Z5", "Z7"),
    ("Z6", "Z8"), ("Z7", "Z8"),
]


def example_facility() -> tuple[nx.Graph, dict[str, tuple[float, float]]]:
    """A visibly labeled placeholder graph, NOT the team's actual topology."""
    graph = nx.Graph()
    graph.add_nodes_from(EXAMPLE_POS)
    graph.add_edges_from(EXAMPLE_EDGES)
    return graph, EXAMPLE_POS.copy()


def read_facility_json(file_or_path: Any) -> tuple[nx.Graph, dict]:
    """Read {nodes:[...], edges:[[u,v],...], positions:{id:[x,y]}} JSON.

    Node IDs should be strings. Uses a repeatable layout if no positions given.
    """
    if hasattr(file_or_path, "read"):
        body = file_or_path.read()
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        data = json.loads(body)
    else:
        data = json.loads(Path(file_or_path).read_text(encoding="utf-8"))
    ids = [str(node) for node in data["nodes"]]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("APS checkpoint requires exactly eight distinct zone IDs")
    graph = nx.Graph()
    graph.add_nodes_from(ids)
    for pair in data["edges"]:
        if len(pair) != 2 or str(pair[0]) not in graph or str(pair[1]) not in graph:
            raise ValueError(f"Invalid edge: {pair}")
        graph.add_edge(str(pair[0]), str(pair[1]))
    if not nx.is_connected(graph):
        raise ValueError("The facility graph should be connected")
    raw_positions = data.get("positions") or {}
    if set(raw_positions) == set(ids):
        positions = {str(node): tuple(map(float, xy)) for node, xy in raw_positions.items()}
    else:
        positions = nx.spring_layout(graph, seed=19)
    return graph, positions


def simulate_illustrative_readings(
    graph: nx.Graph, source: str, decay: float = 0.9, noise_sd: float = 0.07,
    missing_fraction: float = 0.125, seed: int = 12
) -> dict[str, float | None]:
    """Preview only: distance decay + Gaussian noise + random missingness."""
    if source not in graph:
        raise ValueError("source node must exist in graph")
    rng = np.random.default_rng(seed)
    distances = nx.single_source_shortest_path_length(graph, source)
    nodes = list(graph.nodes)
    levels = np.exp(-decay * np.array([distances[node] for node in nodes], float))
    readings = np.clip(levels + rng.normal(0, noise_sd, len(nodes)), 0, 1)
    number_missing = int(round(missing_fraction * len(nodes)))
    missing_idxs = set(rng.choice(len(nodes), number_missing, replace=False).tolist())
    return {node: (None if i in missing_idxs else float(readings[i])) for i, node in enumerate(nodes)}


def strongest_sensor(readings: dict[str, float | None]) -> str | None:
    available = {node: value for node, value in readings.items() if value is not None and np.isfinite(value)}
    return max(available, key=available.get) if available else None


def draw_facility(
    graph: nx.Graph, positions: dict, readings: dict[str, float | None], *,
    true_source: str | None = None, predicted_source: str | None = None,
    suggested_sample: str | None = None, show_truth: bool = False,
    title: str = "APS | 8-zone simulated facility",
) -> plt.Figure:
    """Draw topology with simulated normalized sensor strength and annotations."""
    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
    fig.patch.set_facecolor("#FAFBFD")
    ax.set_facecolor("#FAFBFD")
    nx.draw_networkx_edges(graph, positions, ax=ax, width=3.5, edge_color="#B4C1CF")

    present = [node for node in graph if readings.get(node) is not None and np.isfinite(readings[node])]
    missing = [node for node in graph if node not in present]
    values = [float(readings[node]) for node in present]
    if present:
        nodes = nx.draw_networkx_nodes(
            graph, positions, nodelist=present, node_color=values, cmap=plt.cm.YlOrRd,
            vmin=0, vmax=1, node_size=2150, linewidths=2.5, edgecolors="#273349", ax=ax,
        )
        cbar = fig.colorbar(nodes, ax=ax, shrink=0.68, pad=0.025)
        cbar.set_label("Normalized sensor signal (simulation)", size=10)
    if missing:
        nx.draw_networkx_nodes(
            graph, positions, nodelist=missing, node_color="#D1D5DB",
            node_size=2150, linewidths=2.5, edgecolors="#273349", ax=ax,
        )
    for node, (x, y) in positions.items():
        reading = readings.get(node)
        text_color = "white" if reading is not None and np.isfinite(reading) and reading >= 0.73 else "#111827"
        ax.text(x, y + .11, str(node), ha="center", va="center", fontsize=13,
                fontweight="bold", color=text_color)
        signal_text = "missing" if reading is None or not np.isfinite(reading) else f"{reading:.2f}"
        ax.text(x, y - .15, signal_text, ha="center", va="center", fontsize=9, color=text_color)

    legend_items = [Patch(facecolor="#D1D5DB", edgecolor="#273349", label="Missing reading")]
    if show_truth and true_source in graph:
        x, y = positions[true_source]
        ax.scatter([x], [y + .38], marker="*", s=520, c="#2563EB",
                   linewidths=1.0, edgecolors="white", zorder=8)
        legend_items.append(Line2D([0], [0], marker="*", color="w", markerfacecolor="#2563EB",
                                   markersize=14, label="True source (reveal)") )
    if predicted_source in graph:
        x, y = positions[predicted_source]
        ax.scatter([x], [y], s=2650, facecolors="none", edgecolors="#047857",
                   linewidths=4, zorder=7)
        legend_items.append(Line2D([0], [0], marker="o", color="#047857", markerfacecolor="none",
                                   markersize=13, linewidth=0, markeredgewidth=3, label="Predicted source"))
    if suggested_sample in graph:
        x, y = positions[suggested_sample]
        ax.scatter([x], [y], s=3150, facecolors="none", edgecolors="#7C3AED",
                   linewidths=2.5, linestyle="--", zorder=6)
        legend_items.append(Line2D([0], [0], marker="o", color="#7C3AED", markerfacecolor="none",
                                   markersize=14, linewidth=0, markeredgewidth=2, label="Next sample (example)"))

    ax.legend(handles=legend_items, loc="lower left", frameon=False, fontsize=9, ncol=2,
              bbox_to_anchor=(0, -0.08))
    ax.set_title(title, fontsize=17, weight="bold", color="#172336", loc="left", pad=18)
    ax.text(0, 1.015, "Edges represent zone connectivity, not physical distances.",
            transform=ax.transAxes, fontsize=9, color="#5A6473")
    xvals = [p[0] for p in positions.values()]
    yvals = [p[1] for p in positions.values()]
    ax.set_xlim(min(xvals)-0.75, max(xvals)+0.85)
    ax.set_ylim(min(yvals)-0.8, max(yvals)+0.85)
    ax.axis("off")
    return fig


def validate_runs(df: pd.DataFrame, graph: nx.Graph) -> pd.DataFrame:
    """Validate measured model predictions and calculate reproducible metrics.

    Recommended columns: run_id, seed, model, mode, true_source,
    predicted_source, confidence, noise_sd, missing_fraction, sample_count.
    """
    required = {"run_id", "model", "mode", "true_source", "predicted_source"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Results CSV is missing: {', '.join(sorted(missing_columns))}")
    if df.empty:
        raise ValueError("Results CSV has no prediction rows")
    clean = df.copy()
    for key in ("true_source", "predicted_source", "model", "mode", "run_id"):
        clean[key] = clean[key].astype(str)
    wrong = set(clean["true_source"]) | set(clean["predicted_source"])
    wrong = wrong - set(map(str, graph.nodes))
    if wrong:
        raise ValueError(f"Results contain zones absent from facility graph: {sorted(wrong)}")
    allowed = {"static", "adaptive"}
    if not set(clean["mode"]).issubset(allowed):
        raise ValueError("mode must be 'static' or 'adaptive'")
    if clean.duplicated(["run_id", "model", "mode"]).any():
        raise ValueError("Each (run_id, model, mode) must appear only once")
    for field in ("true_source", "seed", "noise_sd", "missing_fraction"):
        if field in clean and clean.groupby("run_id")[field].nunique(dropna=True).gt(1).any():
            raise ValueError(f"Inconsistent {field} for a shared run_id; use one ID per scenario")
    clean["correct"] = (clean["true_source"] == clean["predicted_source"]).astype(int)
    clean["graph_distance"] = [
        nx.shortest_path_length(graph, actual, guess)
        for actual, guess in zip(clean["true_source"], clean["predicted_source"])
    ]
    for key in ("noise_sd", "missing_fraction", "confidence", "sample_count"):
        if key in clean:
            clean[key] = pd.to_numeric(clean[key], errors="coerce")
    return clean


def accuracy_comparison(df: pd.DataFrame) -> plt.Figure:
    stats = df.groupby(["model", "mode"], sort=True).agg(
        accuracy=("correct", "mean"), n=("correct", "size")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    labels = stats["model"] + " | " + stats["mode"]
    bars = ax.bar(labels, stats["accuracy"].to_numpy() * 100)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Exact-source accuracy (%)")
    ax.set_title("Model comparison | exact-source accuracy")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.annotate(f'{row["accuracy"]:.1%}\n(n={row["n"]})',
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    return fig


def distance_comparison(df: pd.DataFrame) -> plt.Figure:
    stats = df.groupby(["model", "mode"], sort=True).agg(
        avg_distance=("graph_distance", "mean"), n=("graph_distance", "size")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    labels = stats["model"] + " | " + stats["mode"]
    bars = ax.bar(labels, stats["avg_distance"])
    ax.set_ylim(0, max(1, stats["avg_distance"].max() * 1.20))
    ax.set_ylabel("Mean shortest-path error (edges)")
    ax.set_title("Model comparison | source-localization error (lower is better)")
    ax.tick_params(axis="x", rotation=22)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    for bar, (_, row) in zip(bars, stats.iterrows()):
        ax.annotate(f'{row["avg_distance"]:.2f}\n(n={row["n"]})',
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=8)
    return fig


def sensitivity_heatmap(df: pd.DataFrame, *, model: str, mode: str = "static") -> plt.Figure:
    selected = df[(df["model"] == model) & (df["mode"] == mode)].copy()
    for col in ("noise_sd", "missing_fraction"):
        if col not in selected or selected[col].isna().all():
            raise ValueError(f"Sensitivity plot requires '{col}' in results CSV")
    selected["noise_sd"] = selected["noise_sd"].round(4)
    selected["missing_fraction"] = selected["missing_fraction"].round(4)
    matrix = selected.pivot_table(index="noise_sd", columns="missing_fraction", values="correct", aggfunc="mean")
    if matrix.empty:
        raise ValueError("No sensitivity observations found for this model/mode")
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    plotdata = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
    im = ax.imshow(plotdata, cmap="YlGnBu", origin="lower", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)), labels=[f'{x:.0%}' for x in matrix.columns])
    ax.set_yticks(np.arange(len(matrix.index)), labels=[f'{x:.2f}' for x in matrix.index])
    ax.set_xlabel("Missing sensor fraction")
    ax.set_ylabel("Gaussian noise standard deviation")
    ax.set_title(f"Sensitivity | {model} / {mode} accuracy")
    for row in range(len(matrix.index)):
        for col in range(len(matrix.columns)):
            val = matrix.iloc[row, col]
            if np.isfinite(val):
                ax.text(col, row, f"{val:.0%}", ha="center", va="center", fontsize=9,
                        color="black" if val < .7 else "white")
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("Exact-source accuracy")
    return fig


def adaptive_gain(df: pd.DataFrame) -> tuple[plt.Figure, pd.DataFrame]:
    """Pair static/adaptive predictions by exact run_id and model."""
    paired = df.pivot(index=["run_id", "model"], columns="mode", values="correct")
    if not {"static", "adaptive"}.issubset(paired.columns):
        raise ValueError("Adaptive chart requires static and adaptive records")
    paired = paired.dropna(subset=["static", "adaptive"]).reset_index()
    if paired.empty:
        raise ValueError("No runs have both static and adaptive predictions")
    paired["change_pp"] = 100 * (paired["adaptive"] - paired["static"])
    comparison = paired.groupby("model").agg(
        gain_pp=("change_pp", "mean"), paired_runs=("change_pp", "size")
    ).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    bars = ax.bar(comparison["model"], comparison["gain_pp"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_title("One-step adaptive sampling | paired accuracy gain")
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    for bar, (_, row) in zip(bars, comparison.iterrows()):
        ax.annotate(f'{row["gain_pp"]:+.1f} pp\n({row["paired_runs"]} pairs)',
                    (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 7 if row["gain_pp"] >= 0 else -26), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    return fig, comparison


def blank_evaluation_layout(output_path: str | Path) -> None:
    """Six panel *labels and axes* only; intentionally NO fabricated results."""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), layout="constrained")
    specs = [
        ("1  Exact-source accuracy", "Model / method", "Accuracy (%)"),
        ("2  Localization error", "Model / method", "Graph distance (edges)"),
        ("3  Noise & missingness", "Missing fraction", "Noise SD"),
        ("4  Adaptive improvement", "Model", "Accuracy gain (pp)"),
        ("5  Confusion matrix", "Predicted zone", "True zone"),
        ("6  Confidence calibration", "Predicted confidence", "Observed accuracy"),
    ]
    fig.patch.set_facecolor("#F7F9FC")
    for ax, (title, xlabel, ylabel) in zip(axes.flat, specs):
        ax.set_title(title, loc="left", fontsize=12, weight="bold")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.text(.5, .52, "AWAITING EXPERIMENT RESULTS", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#6B7280",
                bbox=dict(boxstyle="round,pad=.75", fc="#F0F2F6", ec="#CDD5DF"))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=.12)
        ax.set_facecolor("white")
    fig.suptitle("APS | Proposed quantitative evaluation graphics", fontsize=18, weight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent
    graph, positions = example_facility()
    readings = simulate_illustrative_readings(graph, source="Z4", missing_fraction=0.125, seed=12)
    fig = draw_facility(
        graph, positions, readings, true_source="Z4", predicted_source=strongest_sensor(readings),
        suggested_sample="Z7", show_truth=True,
        title="APS | illustrative 8-zone facility graph",
    )
    fig.savefig(out / "facility_preview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    blank_evaluation_layout(out / "evaluation_graphics_outline.png")
    print("Created facility_preview.png and evaluation_graphics_outline.png")
