# Adaptive Pathogen Surveillance (APS)

**CSC 4240 — Artificial Intelligence | Tennessee Technological University**

Team: Matthew Richardson, Emmanuel Zuniga, Justin Hortopanu, Jimmy Finnegan, Owen Brown.

APS is an undergraduate AI research simulation investigating whether source-localization methods can identify the origin of a synthetic environmental contamination event from incomplete/noisy sensor readings, and whether selecting one additional sensor observation by expected information gain improves localization over random selection.

**Scope:** This is an eight-zone *spatial snapshot simulation*, not a clinical tool, real-world biosensor deployment, airflow or disease-transmission model. Genomic analysis and response automation remain explicitly deferred under `Future/`.

## Implemented system

1. **Facility & synthetic events:** canonical eight-zone NetworkX graph; randomly selected hidden event source; exponential signal decay over graph edges; configured source intensity, sensor noise, missingness, and reproducible seeds.
2. **Baseline localization:** strongest observed sensor (Argmax), abstaining if every reading is missing.
3. **Source inference:** original and sparsity/noise-augmented Logistic Regression and Random Forest classifiers, plus Graph-Aware Bayesian Inference that marginalizes unknown signal parameters rather than accessing the true event settings.
4. **One-step sampling:** a budget-matched random selector and a model-assisted expected-information-gain selector each request one follow-up observation; the static control requests none.
5. **Evaluation:** held-out exact-source accuracy and graph-distance error; model and condition comparisons; matched seeded sampling trials; abstention-aware confusion matrices and accuracy versus actual missing-reading count.
6. **Dashboard:** Streamlit `OB/live_dashboard.py`, including 3D facility schematic, editable **2D presentation-only** layout, model comparison, adaptive sampling, experiment lab, and CSV exports.
7. **Regression tests:** `Tests/` includes simulation, localization, ML, sampling, plotting/dashboard smoke, and Owen-inspired diagnostic checks.

## Quick start (Windows PowerShell)

Requires a supported Python 3 environment, Git, and a shell in the repository root.

```powershell
git clone https://github.com/matthew-richardson-ai/aps-pathogen-surveillance-simulation.git
cd aps-pathogen-surveillance-simulation
git switch feature/demo-ready-aps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r OB/requirements.txt
python -m pytest -q
python -m streamlit run OB/live_dashboard.py
```

The Streamlit terminal prints a local URL (typically `http://localhost:8501`). If Python is installed as `py`, use `py -m venv .venv` first. No real sensors or external clinical data are required.

**Synthetic training data:** `Localization/ml_pipeline.py` locates Justin's committed dataset and uses an established train/test partition. Keep the `Justin/` data directory intact when preparing a submission archive.

## Reproduce the final evaluation

From the repository root, with the virtual environment activated:

```powershell
python Scripts/prepare_demo_results.py
```

The script regenerates `Results/Demo/` with:
- `heldout_model_summary.csv` and `heldout_model_predictions.csv`
- `sampling_summary.csv` and `sampling_runs.csv`
- `experiment_metadata.json`

The evaluator trains on 7,000 independent training rows (28,000 original-plus-augmented views) and uses 1,500 untouched held-out test rows. Its adaptive benchmark uses 100 independently seeded events in each of four simulation conditions, with five model variants and three sampling strategies on matched events. Regeneration may take time.

**Important:** `Results/` is Git-ignored intentionally, so generated CSVs are local artifacts, not automatically uploaded to GitHub. Preserve the outputs when packaging the final code/report; do not commit large experiment data by accident.

## Final November 23, 2026 results (synthetic only)

Held-out source accuracy:
- Argmax baseline: **86.20%**
- Logistic Regression, original / robust: **92.00% / 92.47%**
- Random Forest, original / robust: **92.93% / 93.00%**
- Graph-Aware Bayesian Inference (blind event parameters): **91.47%**

Selected matched 100-trial findings:
- Missing Readings, robust Logistic Regression: **73% static**, **94% random**, **100% information gain**.
- Combined Stress, blind Graph-Aware Bayesian Inference: **36% static**, **56% random**, **57% information gain**.

These outcomes are specific to a controlled synthetic experiment. In particular, all extra samples use a follow-up measurement noise SD of 0.04; the gain over no extra sample is not evidence that information gain is better than a sample-budget-matched random policy. Exploratory single-condition results should not be generalized without replication, uncertainty analysis, and correction for multiple comparisons.

Current findings: see `Docs/APS_Final_Results_2026-11-23.md`.
Prior **Known Params** benchmark is retained for historical comparison in `Docs/APS_Benchmark_Findings.md` and **must not** be cited as the current blind Bayesian result.

## Additional documentation

- `OB/LIVE_DASHBOARD_README.md` — dashboard controls and evaluation caveats
- `OB/ADAPTIVE_SAMPLING_README.md` — adaptive policy overview
- `Docs/APS_Robust_Localization_Methods.md` — estimator assumptions
- `Docs/APS_Demo_Runbook.md` — presentation steps
- `Docs/Project_Proposal_Revised.md` — original project scope

The CSC 4240 final report has a strict **four-page maximum**; its editable draft is being reviewed separately from the code release. Submission location and packaging should follow the professor's final iLearn instructions.
