# APS Visualization Starter

This package supports the **visualization/evaluation** responsibility for the simulation-only CSC 4240 Adaptive Pathogen Surveillance project. It does not train machine-learning models, implement the adaptive policy, or represent real sensor data.

## Included files

- `facility_preview.png` — sample eight-zone facility visualization; topology, readings, and next-sample ring are **illustrative only**.
- `evaluation_graphics_outline.png` — six planned evaluation plot panels; **no model results**.
- `aps_viz.py` — plotting and CSV validation functions, plus a standalone preview generator.
- `aps_app.py` — Streamlit starter interface with facility and evaluation tabs.
- `sample_graph.json` — illustrative topology in an importable graph format.
- `results_schema.csv`, `sensor_schema.csv` — blank CSV headers for data integration.
- `APS_Evaluation_Plan.md` — chart specifications, metrics, collaboration tasks, and demo walkthrough.

## Setup

```bash
python -m pip install -r requirements.txt
python aps_viz.py
streamlit run aps_app.py
```

`aps_viz.py` runs without Streamlit and saves the preview images. Streamlit is needed to launch the interactive interface. The app was syntax-checked, but could not be launched in the build environment because Streamlit was not installed there.

## Connect the team's graph

`sample_graph.json` shows the import format: eight node ID strings, edges as node pairs, and optional coordinates. Coordinates position nodes in the picture; edges alone determine shortest-path distance. To generate an export from your existing NetworkX graph:

```python
import json

# G = the team's already-built eight-node NetworkX graph.
facility = {
    "nodes": [str(n) for n in G.nodes],
    "edges": [[str(a), str(b)] for a, b in G.edges],
}
with open("team_facility.json", "w", encoding="utf-8") as file:
    json.dump(facility, file, indent=2)
```

Upload the exported JSON to the Streamlit sidebar. The app will use a repeatable layout when `positions` is omitted.

## Connect the team's sensor data

Upload a CSV with `zone,signal` columns, for example zone ID `Z1` or the actual IDs in `team_facility.json`; leave `signal` blank for missing data. Values should be normalized to `[0, 1]`. If you do not upload readings, the interface makes **illustrative** noise/decay/missingness readings for an interactive example.

## Connect repeated model results

Use one row per model per trial per sampling mode. `results_schema.csv` supplies the column headers (but deliberately contains no synthetic model accuracy data).

Required columns:

- `run_id`: a unique test outbreak scenario ID, reused across matched methods/modes.
- `model`: e.g. `strongest_sensor`, `logistic_regression`, `random_forest`.
- `mode`: exactly `static` or `adaptive`.
- `true_source`: true hidden zone ID, recorded for evaluation only.
- `predicted_source`: model's predicted zone ID.

Recommended columns:

- `seed` (experiment seed), `noise_sd`, `missing_fraction` (experimental conditions).
- `confidence` (0–1 when available), `sample_count` (extra samples used, normally 0 or 1), `sampled_zone`.

Run IDs must uniquely describe a scenario (including noise/missing settings); the same run ID should refer to the same hidden source across all compared models and sampling modes. The plots recompute correctness and graph distance against the uploaded graph and identify paired adaptive/static rows by `(run_id, model)`.

**Caution:** To compare an ML model's adaptive mode, the team must first implement its model update after one extra reading. Merely labeling rows `adaptive` does not make them adaptive evaluations.

## Scope

All built-in data are illustrative. This is strictly a simulation/demonstration tool. No clinical diagnosis, physical sensors, wet-lab procedures, genomic analysis, or automated response recommendations are included.
