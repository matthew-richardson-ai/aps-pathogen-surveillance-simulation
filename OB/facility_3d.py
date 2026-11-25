"""Interactive 3D schematic of the eight-zone APS facility.

Each extruded room corresponds to exactly one node in the canonical NetworkX
facility graph. Raised outlines and connectors are a visual abstraction; they
do not model physical walls, room geometry, airflow, or time-dependent spread.
"""
from __future__ import annotations

import plotly.graph_objects as go

# A presentation layout, not measured room coordinates. Edges are drawn
# from Simulation.facility.build_facility() rather than reconstructed here.
ROOM_CENTERS = {
    "zone_1": (-3.7, 1.8),
    "zone_2": (-3.7, -1.8),
    "zone_3": (-1.45, 0.0),
    "zone_4": (0.75, 1.8),
    "zone_5": (0.75, -1.8),
    "zone_6": (3.0, 0.0),
    "zone_7": (5.15, 0.0),
    "zone_8": (7.3, 0.0),
}
ROOM_HALF_WIDTH = 0.75
ROOM_HALF_DEPTH = 0.62
FLOOR_HEIGHT = 0.12
WALL_HEIGHT = 0.64


def room_color(reading: float | None) -> str:
    """Color is based on the actual observed reading; missing is not zero."""
    if reading is None:
        return "#394B63"
    level = max(0.0, min(1.0, float(reading)))
    if level < 0.20:
        return "#5098BC"
    if level < 0.40:
        return "#40C2C4"
    if level < 0.65:
        return "#75D6AF"
    if level < 0.85:
        return "#F0C171"
    return "#F08A70"


def _box_vertices(x: float, y: float):
    left, right = x - ROOM_HALF_WIDTH, x + ROOM_HALF_WIDTH
    near, far = y - ROOM_HALF_DEPTH, y + ROOM_HALF_DEPTH
    return (
        [left, right, right, left, left, right, right, left],
        [near, near, far, far, near, near, far, far],
        [0.0] * 4 + [FLOOR_HEIGHT] * 4,
    )


def facility_3d_plot(snapshot, graph, baseline_zone: str | None,
                     show_source: bool = False, selected_zone: str | None = None) -> go.Figure:
    """Return a rotatable, zoomable Plotly room map linked to APS sensor data."""
    zones = set(graph.nodes)
    if zones != set(ROOM_CENTERS) or zones != set(snapshot.readings):
        raise ValueError("3D facility rooms must match the APS eight-zone graph")

    fig = go.Figure()
    # Graph connections are schematic links, not purported physical passageways.
    paths_x, paths_y, paths_z = [], [], []
    for a, b in graph.edges:
        paths_x += [ROOM_CENTERS[a][0], ROOM_CENTERS[b][0], None]
        paths_y += [ROOM_CENTERS[a][1], ROOM_CENTERS[b][1], None]
        paths_z += [0.02, 0.02, None]
    fig.add_trace(go.Scatter3d(
        x=paths_x, y=paths_y, z=paths_z, mode="lines",
        line=dict(color="#587997", width=9),
        name="Connected zones", hoverinfo="skip", showlegend=False,
    ))

    for zone in sorted(ROOM_CENTERS):
        x, y = ROOM_CENTERS[zone]
        reading = snapshot.readings[zone]
        selected = zone == selected_zone
        base_color = room_color(reading)
        floor_x, floor_y, floor_z = _box_vertices(x, y)
        fig.add_trace(go.Mesh3d(
            x=floor_x, y=floor_y, z=floor_z,
            i=[0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
            j=[1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
            k=[2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
            color=base_color, opacity=1.0, flatshading=True,
            lighting=dict(ambient=0.82, diffuse=0.66, specular=0.14, roughness=0.8),
            hoverinfo="skip", showlegend=False, name=zone,
        ))

        x0, x1 = x - ROOM_HALF_WIDTH, x + ROOM_HALF_WIDTH
        y0, y1 = y - ROOM_HALF_DEPTH, y + ROOM_HALF_DEPTH
        # Room-frame outlines make it read as a cutaway building with walls.
        segments = [
            ((x0, y0, FLOOR_HEIGHT), (x0, y0, WALL_HEIGHT)),
            ((x1, y0, FLOOR_HEIGHT), (x1, y0, WALL_HEIGHT)),
            ((x1, y1, FLOOR_HEIGHT), (x1, y1, WALL_HEIGHT)),
            ((x0, y1, FLOOR_HEIGHT), (x0, y1, WALL_HEIGHT)),
            ((x0, y0, WALL_HEIGHT), (x1, y0, WALL_HEIGHT)),
            ((x1, y0, WALL_HEIGHT), (x1, y1, WALL_HEIGHT)),
            ((x1, y1, WALL_HEIGHT), (x0, y1, WALL_HEIGHT)),
            ((x0, y1, WALL_HEIGHT), (x0, y0, WALL_HEIGHT)),
        ]
        wall_x, wall_y, wall_z = [], [], []
        for start, end in segments:
            wall_x.extend((start[0], end[0], None))
            wall_y.extend((start[1], end[1], None))
            wall_z.extend((start[2], end[2], None))
        fig.add_trace(go.Scatter3d(
            x=wall_x, y=wall_y, z=wall_z, mode="lines",
            line=dict(color="#F4F8FD" if selected else "#A1C0D7", width=8 if selected else 4),
            name="Selected room" if selected else zone,
            showlegend=False, hoverinfo="skip",
        ))

        observed = "No reading" if reading is None else f"Signal: {reading:.3f}"
        status = "Additional sample" if snapshot.requested_zone == zone else (
            "Observed" if reading is not None else "Unavailable"
        )
        label = f"Zone {int(zone.split('_')[-1])}"
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[FLOOR_HEIGHT + 0.18],
            mode="markers+text", marker=dict(size=10 if selected else 8, color="#111E2F"),
            text=[label], textposition="top center",
            textfont=dict(size=13, color="#FFFFFF"),
            name=label, showlegend=False,
            customdata=[[label, observed, status]],
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                          "<br>%{customdata[2]}<extra></extra>",
        ))

    for zone, label, marker_color, altitude, symbol in [
        (baseline_zone, "Argmax Baseline", "#7FF4C7", 1.05, "diamond"),
        (snapshot.requested_zone, "Additional measurement", "#D1ADFF", 1.33, "diamond"),
        (snapshot.true_source if show_source else None, "True source", "#FFD76C", 1.62, "diamond"),
    ]:
        if zone is None:
            continue
        x, y = ROOM_CENTERS[zone]
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[altitude], mode="markers",
            marker=dict(size=13, symbol=symbol, color=marker_color,
                        line=dict(color="#0A1525", width=2)),
            name=label, showlegend=True,
            hovertemplate=f"{label}: Zone {int(zone.split('_')[-1])}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="#142238", plot_bgcolor="#142238",
        font=dict(color="#EDF5FF", size=12),
        height=530, margin=dict(l=0, r=0, t=5, b=0),
        scene=dict(
            xaxis=dict(visible=False, range=[-5.0, 8.7]),
            yaxis=dict(visible=False, range=[-3.2, 3.2]),
            zaxis=dict(visible=False, range=[-0.2, 2.05]),
            bgcolor="#142238", aspectmode="manual",
            aspectratio=dict(x=2.35, y=1.12, z=0.52),
            camera=dict(eye=dict(x=1.45, y=-1.75, z=1.25),
                        center=dict(x=0, y=0, z=-0.16)),
            dragmode="orbit",
        ),
        legend=dict(orientation="h", x=0.02, y=0.02,
                    bgcolor="rgba(20,34,56,0.7)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#F0F7FF", font=dict(color="#102034", size=13)),
        uirevision="aps-room-layout",
    )
    return fig
