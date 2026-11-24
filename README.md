# Adaptive Pathogen Surveillance (APS)

**CSC 4240 — Artificial Intelligence | Tennessee Technological University**

**Team:** Matthew Richardson, Emmanuel Zuniga, Justin Hortopanu, Jimmy Finnegan, Owen Brown.

APS is an undergraduate AI project investigating whether AI methods can localize the source of a simulated environmental contamination event from noisy or missing sensor readings, and whether selecting one additional observation using expected information gain improves localization over random selection.

**Project scope:** APS is an eight-zone spatial snapshot simulation. It is not a clinical tool, a real-world biosensor deployment, an airflow or disease-transmission model, or a pathogen-detection product. Genomic analysis and automated response recommendations are deferred extensions, not completed CSC 4240 functionality.

## Implemented system

1. **Facility and synthetic events:** an eight-zone NetworkX graph, hidden source selection, exponential signal decay over graph distance, sensor noise, missing readings, and reproducible random seeds.
2. **Baseline localization:** strongest-observed-sensor (Argmax) prediction, with abstention if all readings are missing.
3. **AI-based source inference:** Logistic Regression and Random Forest classifiers, original and sparsity/noise-augmented training variants, plus Graph-Aware Bayesian Inference that marginalizes unknown signal parameters rather than receiving the true event settings.
4. **One-step additional sampling:** static/no-extra-sample control, random sampling, and model-assisted expected-information-gain sampling. Random and information-gain strategies each receive one additional measurement.
5. **Evaluation:** held-out exact-source accuracy and graph-distance error; matched seeded sampling trials under normal, high-noise, missing-reading, and combined-stress conditions.
6. **Interactive dashboard:** Streamlit application at `OB/live_dashboard.py`, with 3D facility visualization, editable 2D schematic layout, model comparisons, adaptive sampling, experiment evaluation, and CSV exports.
7. **Tests:** 66 automated tests were passing in the team's latest local verification. The test suite is in `Tests/`.

## Requirements

- **Python 3.11** (the repository's GitHub Actions test workflow uses Python 3.11)
- Git
- Internet access during initial dependency installation

Run all commands below from the repository root folder. The full implementation and current experiment results are on the `feature/demo-ready-aps` branch. Until that branch is merged into `main`, use the branch-specific clone command below.

## Install and run on Windows (PowerShell)

### 1. Clone the completed project branch

```powershell
git clone --branch feature/demo-ready-aps --single-branch https://github.com/matthew-richardson-ai/aps-pathogen-surveillance-simulation.git
cd aps-pathogen-surveillance-simulation
```

### 2. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the `py` launcher is unavailable, use `python -m venv .venv` instead. If PowerShell blocks virtual-environment activation, allow it for the current terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install all project and dashboard dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r OB/requirements.txt
```

The root requirements file installs the simulation, data-science, machine-learning, and testing packages. `OB/requirements.txt` adds the dashboard and visualization dependencies, including Streamlit and Plotly.

### 4. Run the automated tests

```powershell
python -m pytest -q
```

The project team most recently verified **66 passing tests**. Test output may include a different count if tests are added or changed after this README was written.

### 5. Launch the interactive dashboard

```powershell
python -m streamlit run OB/live_dashboard.py
```

Streamlit prints a local address in the terminal. Open **http://localhost:8501** if the browser does not open automatically. Keep the terminal running while using the dashboard; press `Ctrl+C` in that terminal to stop it.

The dashboard trains/loads the source-localization models from the committed synthetic dataset when required. The dataset must remain in the repository. The current branch includes CSV files under `Justin/Data/`; the model loader also supports the newer `Team/Justin/Data/` folder layout.

### macOS / Linux equivalent

From a terminal, clone the same branch and create the virtual environment:

```bash
git clone --branch feature/demo-ready-aps --single-branch https://github.com/matthew-richardson-ai/aps-pathogen-surveillance-simulation.git
cd aps-pathogen-surveillance-simulation
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r OB/requirements.txt
python -m pytest -q
python -m streamlit run OB/live_dashboard.py
```

Open the local Streamlit URL printed in the terminal (normally `http://localhost:8501`). Keep that terminal open while the dashboard is running; press `Ctrl+C` to stop it.

## Run the simulation and reproduce the evaluation

These commands are optional for someone who only wants to open the dashboard.

### Run one end-to-end baseline simulation

From the repository root:

```powershell
python Scripts/run_pipeline.py
```

This runs one seeded simulation and prints the hidden source, sensor readings, baseline prediction, and graph-distance error for inspection.

### Generate the full experimental results

```powershell
python Scripts/prepare_demo_results.py
```

This trains the models and runs the held-out and matched adaptive-sampling experiments. It creates:

- `Results/Demo/heldout_model_summary.csv`
- `Results/Demo/heldout_model_predictions.csv`
- `Results/Demo/sampling_summary.csv`
- `Results/Demo/sampling_runs.csv`
- `Results/Demo/experiment_metadata.json`

The experiment evaluates 1,500 untouched held-out examples and 100 independently seeded events in each of four simulation conditions. The matched sampling comparison evaluates five model variants under three strategies: no additional sample, one random additional sample, and one information-gain additional sample.

**Important:** `Results/` is Git-ignored, so these generated CSV and JSON files remain local unless deliberately copied into a submission archive. Preserve the files used for the final report if the instructor requests the experimental outputs.

## Dataset

The model pipeline discovers the committed file matching `outbreak_dataset_*.csv` under either `Team/Justin/Data/` or `Justin/Data/`. Do not remove or rename the dataset folder when copying or zipping the project. The project can use the already-committed dataset; regenerating it is not required to launch the dashboard or reproduce the current evaluation.

## Current measured results (synthetic experiments only)

Held-out exact-source accuracy:

- Argmax baseline: **86.20%**
- Logistic Regression, original / robust: **92.00% / 92.47%**
- Random Forest, original / robust: **92.93% / 93.00%**
- Graph-Aware Bayesian Inference (blind event parameters): **91.47%**

Selected matched 100-trial results:

- Missing Readings, robust Logistic Regression: **73% static**, **94% random**, **100% information gain**.
- Combined Stress, blind Graph-Aware Bayesian Inference: **36% static**, **56% random**, **57% information gain**.

All outcomes are specific to this controlled synthetic setup. Follow-up measurements use a fixed noise standard deviation of 0.04. A gain over the no-extra-sample condition does not by itself show that information-gain selection is better than a sample-budget-matched random policy. Single-condition results are exploratory and should not be generalized without additional replications and uncertainty analysis.

See `Docs/APS_Final_Results_2026-11-23.md` for the current blind-inference results and limitations. `Docs/APS_Benchmark_Findings.md` contains a historical benchmark using an earlier Known Params graph method; **do not quote its obsolete graph-model percentages as current results**.

## Additional documentation

- `OB/LIVE_DASHBOARD_README.md` — dashboard controls and evaluation caveats
- `OB/ADAPTIVE_SAMPLING_README.md` — adaptive sampling overview
- `Docs/APS_Robust_Localization_Methods.md` — estimator assumptions and methods
- `Docs/APS_Demo_Runbook.md` — demonstration sequence
- `Docs/APS_Final_Results_2026-11-23.md` — current experiment findings
- `Docs/Project_Proposal_Revised.md` — original project scope

The CSC 4240 report must not exceed **four pages**. Follow the instructor's final iLearn instructions for the upload format and submission location.
