"""
facility_viz.py -- facility graph drawing for the APS demo and report.

Plain-Python inputs, so it works with whatever the simulator returns:
    G             networkx.Graph, one node per zone
    readings      {zone: float}
                    float -> sensor reading
                    nan / None -> sensor exists but its reading is missing
                    zone absent -> no sensor in that zone
    true_source, predicted, extra_sample -> zone ids (or None)
    probs         {zone: probability} from the model (optional)
"""
from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.transforms import offset_copy

CMAP = plt.get_cmap("YlOrRd")
C_TRUE, C_PRED, C_EXTRA = "#D55E00", "#0072B2", "#009E73"


def layout(G, seed=7):
    """Compute positions ONCE and reuse them so the map doesn't jump between frames."""
    try:
        return nx.kamada_kawai_layout(G)
    except Exception:  # scipy missing or degenerate graph
        return nx.spring_layout(G, seed=seed)


def _off(ax, dx, dy):
    """Offset in points, so overlays stay put whatever the layout scale is."""
    return offset_copy(ax.transData, fig=ax.figure, x=dx, y=dy, units="points")


def _missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def draw_facility(G, readings=None, true_source=None, predicted=None,
                  extra_sample=None, probs=None, pos=None, ax=None,
                  title=None, vmax=None, legend=True, colorbar=True):
    readings = readings or {}
    pos = pos or layout(G)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))

    vals = [v for v in readings.values() if not _missing(v)]
    vmax = vmax or (max(vals) if vals else 1.0)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9aa0a6", width=2.5)

    for z in G.nodes:
        x, y = pos[z]
        has_sensor = z in readings
        if not has_sensor:
            face, ls, note = "white", "--", ""
        elif _missing(readings[z]):
            face, ls, note = "#d9d9d9", "-", "missing"
        else:
            face, ls, note = CMAP(min(readings[z] / vmax, 1.0)), "-", f"{readings[z]:.2f}"

        ax.scatter([x], [y], s=1700, c=[face], edgecolors="#333", linewidths=1.5,
                   linestyle=ls, zorder=3)
        ax.text(x, y, str(z), ha="center", va="center", fontsize=10,
                fontweight="bold", zorder=6)
        if probs and z in probs:
            note = (note + "\n" if note else "") + f"p={probs[z]:.2f}"
        if note:
            ax.annotate(note, (x, y), xytext=(0, -27), textcoords="offset points",
                        ha="center", va="top", fontsize=8, color="#333", zorder=6)

    # overlays: predicted ring, true-source star, extra-sample diamond
    if predicted is not None:
        ax.scatter(*pos[predicted], s=2500, facecolors="none", edgecolors=C_PRED,
                   linewidths=3.5, zorder=4)
    if true_source is not None:
        ax.scatter(*pos[true_source], marker="*", s=450, c=C_TRUE, edgecolors="k",
                   linewidths=0.8, zorder=7, transform=_off(ax, 20, 20))
    if extra_sample is not None:
        ax.scatter(*pos[extra_sample], marker="D", s=140, c=C_EXTRA, edgecolors="k",
                   linewidths=0.8, zorder=7, transform=_off(ax, -20, 20))

    if legend:
        handles = [
            Line2D([], [], marker="*", ls="", ms=14, mfc=C_TRUE, mec="k", label="true source (hidden)"),
            Line2D([], [], marker="o", ls="", ms=14, mfc="none", mec=C_PRED, mew=3, label="predicted source"),
            Line2D([], [], marker="D", ls="", ms=8, mfc=C_EXTRA, mec="k", label="adaptive sample"),
            Line2D([], [], marker="o", ls="", ms=12, mfc="#d9d9d9", mec="#333", label="sensor reading missing"),
            Line2D([], [], marker="o", ls="--", ms=12, mfc="white", mec="#333", label="no sensor"),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=2, fontsize=8, frameon=False)
    if colorbar:
        sm = ScalarMappable(norm=Normalize(0, vmax), cmap=CMAP)
        plt.colorbar(sm, ax=ax, shrink=0.7, label="sensor reading")

    ax.set_title(title or "", fontsize=12)
    ax.set_aspect("equal")
    ax.margins(0.2)
    ax.axis("off")
    return ax


def draw_adaptive_step(G, before, after, true_source, pred_before, pred_after,
                       extra_zone, probs_before=None, probs_after=None, pos=None):
    """Two panels with a shared color scale: static sensors -> one adaptive sample."""
    pos = pos or layout(G)
    allv = [v for r in (before, after) for v in r.values() if not _missing(v)]
    vmax = max(allv) if allv else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    draw_facility(G, before, true_source, pred_before, None, probs_before, pos,
                  axes[0], "Before: static sensors", vmax, legend=False, colorbar=False)
    draw_facility(G, after, true_source, pred_after, extra_zone, probs_after, pos,
                  axes[1], "After: one adaptive sample", vmax)
    fig.tight_layout()
    return fig


def save(fig, name, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


if __name__ == "__main__":
    # PLACEHOLDER graph + readings so the file runs standalone.
    # Swap in the simulator's real 8-zone graph and readings.
    import numpy as np

    rng = np.random.default_rng(1)
    G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0), (1, 5), (2, 6)])
    src = 2
    dist = nx.single_source_shortest_path_length(G, src)
    reads = {z: float(np.exp(-0.7 * dist[z]) + rng.normal(0, 0.08)) for z in G.nodes}
    reads[6] = float("nan")                       # simulate a dropped sensor
    before = {z: v for z, v in reads.items() if z not in (3, 4)}   # fewer sensors
    after = {**before, 3: reads[3]}               # adaptive sample at zone 3
    probs_b = {0: .10, 1: .22, 2: .20, 5: .18, 7: .12}
    probs_a = {0: .05, 1: .20, 2: .45, 5: .10, 7: .05, 3: .15}

    pos = layout(G)
    fig = draw_adaptive_step(G, before, after, src, 1, 2, 3, probs_b, probs_a, pos)
    print(save(fig, "facility_before_after"))
    fig1 = plt.figure(figsize=(6.5, 5.5))
    draw_facility(G, reads, src, 2, None, None, pos, fig1.gca(), "Single experiment snapshot")
    print(save(fig1, "facility_snapshot"))
