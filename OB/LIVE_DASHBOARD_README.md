# APS Live Simulation Dashboard

This is a first interactive UI that uses the **real eight-zone APS simulation modules**. It is added alongside Owen's existing visualization/evaluation starter, not as a replacement.

## Run from the VS Code terminal (repository root)

```powershell
python -m pip install -r requirements.txt
python -m pip install -r OB/requirements.txt
python -m streamlit run OB/live_dashboard.py
```

Streamlit should open a browser window, typically at http://localhost:8501.

## Functionality

- Six condition presets: normal, limited coverage, noisy sensors, missing sensors, weak source, and combined stress.
- Change the hidden source (random or fixed), random seed, initially observed zones, measurement noise, missing-reading probability, source strength, and graph-distance decay.
- Interactive Plotly facility visualization: hover/zoom to inspect sensor values. Display or hide the hidden source.
- Run the **Maximum-Sensor-Reading (Argmax) Baseline** from `Localization/baselines.py`. The deterministic rule is `max(available, key=available.get)`: it chooses the zone with the highest available reading; it does not train on data. If every reading is missing, the dashboard reports no prediction.
- Request **one manually chosen additional observation** at a zone without a reading, and inspect the changed baseline prediction.
- Run independent baseline-only trials, graph exact-source accuracy by source zone, and export CSV.

This code directly imports Simulation/facility.py, Simulation/outbreak.py, Simulation/sensors.py, and Localization/baselines.py; the graph is not the illustrative one in Owen's original example.

## Scope and limitations

- Each run is a **spatial snapshot** with exponential graph-distance signal decay, not actual airflow, pathogen transport, or a time animation.
- The extra sample is selected **manually**, not by the AI. It assumes an independent measurement can be obtained at a missing/unobserved zone, with noise SD 0.04. A maximum of one extra measurement is allowed.
- The dashboard has an optional Model comparison tab that trains Logistic Regression and Random Forest from Justin's saved synthetic dataset and compares both against the baseline on its held-out test split.
- Repeated-trial statistics are **new synthetic results computed by this app**, not claims about real-world performance. No-reading cases count as incorrect for accuracy and are excluded from mean graph-distance error.

Owen's original OB/aps_app.py remains available to visualize actual model evaluation CSVs as integration progresses.

## Model comparison

The new tab adds a three-way held-out accuracy table, graph-distance error, accuracy by synthetic condition, and downloadable individual predictions. All-missing initial observations result in abstention rather than an unsupported model guess. The app uses the preassigned train/test labels and checks persistent group IDs do not overlap between train and test. These results concern simulated scenarios only.

## Validation

```powershell
python -m pytest Tests/test_live_engine.py -q
```

The engine was exercised with six deterministic unit tests locally using the same canonical graph/sensor functions. The full Streamlit browser application still needs to be launched and checked on a machine with Streamlit installed.

## Next integration

1. Open **Model comparison** and click **Train models and evaluate held-out test data**. The dashboard reads the most recent timestamped CSV in `Team/Justin/Data`; only initial readings and missingness flags are used as features. Results are computed on the dataset's preassigned test partition.
2. Change the live scenario settings and observe the updated Logistic Regression and Random Forest predictions. Their largest class probabilities are not calibrated confidence measures.
3. Run `python -m pytest Tests/test_ml_pipeline.py -q` for the model integration tests.
4. Implement and validate a genuine confidence-/information-based one-step sampling policy and budget-matched random comparator.
5. Export paired static versus one-sample predictions to Owen's evaluation dashboard.


## Interactive 3D facility (presentation view)

The **APS Demonstration** and **Live Facility** tabs now include a **Facility visualization** choice:

- **3D Rooms** (default): rotate with mouse drag, zoom with the wheel and hover over individual rooms for their observed sensor readings. The **Inspect a Room** selector highlights a room, gives its signal value and lists the connected zones.
- **2D Network**: retains the original APS connected graph for explaining graph topology.

The 3D scene is a *schematic cutaway*, not a surveyed building floor plan. All eight modeled rooms correspond exactly to the canonical NetworkX nodes; connecting lines are derived from the simulation graph. Floor colors use the current measured signal, with gray for missing/unobserved values. Green identifies the Argmax Baseline prediction, purple identifies the extra measurement, and gold reveals the true source only when the reveal toggle is enabled.

Changing the scenario, number of available sensors, noise, missingness or chosen sample updates the same facility view. This is interactive spatial visualization **of a static simulation snapshot**; no real-time biological sensors or time-dependent transport are implied.

The dashboard also uses a high-contrast theme designed for classroom projector visibility, including bright explanatory text and dark lettering on teal primary buttons.

### 3D smoke test

```powershell
python -m pytest Tests/test_facility_3d.py -q
python -m streamlit run OB/live_dashboard.py
```

The 3D renderer uses the existing `plotly` dependency; no additional 3D engine or graphics packages are required.
