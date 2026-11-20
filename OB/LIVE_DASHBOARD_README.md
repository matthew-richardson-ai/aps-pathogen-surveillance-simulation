# APS Dashboard — Classroom Demonstration and Model Comparisons

The dashboard uses the real eight-zone APS NetworkX facility, synthetic source
signals, noise/missingness and one-step sampling engine. **It is an idealized
spatial snapshot, not a live biosensor system, airflow model, or clinical tool.**

## Start and verify on Windows

From the repository root, with the local virtual environment activated:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r OB/requirements.txt
python -m pytest -q
python Scripts/prepare_demo_results.py
python -m streamlit run OB/live_dashboard.py
```

Run the experiment-preparation script after changing models or sampling logic,
**not on every Streamlit startup**. Results are saved under `Results/Demo`.
The dashboard uses these measured results; numbers are not hard-coded.

## Demonstration controls

- **APS Demonstration:** Fixed, repeatable Normal / Missing Readings / Combined
  Stress examples. Combined Stress fixes the source at Zone 6 and seed 81.
- **Generate New Scenario (Random Source):** Advances the seed and randomizes
  the source while retaining the chosen noise, coverage and decay settings.
  A random event can pick the same zone twice by chance.
- **3D Rooms / 2D Network:** Rotatable schematic eight-room view or graph view,
  built from the canonical NetworkX topology. Gray means no reading, not zero.
  Hover for observations, or use **Inspect a Room** to highlight a location.
- **Reveal True Source:** Hidden during prediction and displayed only for
  evaluation. The true source is never an input to the prediction algorithms.
- **Adaptive Sensor Selection:** Choose a learned classifier or graph likelihood
  as the probability source. The entropy-based selector chooses one currently
  unobserved room. Random and information-gain policies each get one sample.

## Models and fair comparisons

| Method | Predictive evidence |
|---|---|
| **Argmax Baseline** | Picks the highest available observed sensor value; no fitting |
| **Logistic Regression (Original)** | Original Justin training data, eight readings + eight missingness flags |
| **Random Forest (Original)** | Original Justin training data and same 16 features |
| **Logistic Regression** | Original training rows plus three sparse/noisy *training-only* views per row |
| **Random Forest** | Same augmented training views, same 16 inputs |
| **Graph-Aware Likelihood (Known Params)** | Current observed readings plus **known simulation intensity, graph decay, and measurement noise** |

The graph-aware method has a **different information budget** from the ML
classifiers; do not present it as an equally informed head-to-head comparison.
It uses censored Gaussian likelihood to model the simulator's clipping at zero
and one, and separately accounts for follow-up measurement noise SD 0.04.
Missing observations provide no evidence. An all-missing scenario abstains.

Robust training is not guaranteed to help: both the original and augmented
classifiers are evaluated on the same held-out partition and remain visible
to reveal wins **and losses**. Test/validation examples are never augmented or
used to train the classifiers. See `Docs/APS_Robust_Localization_Methods.md`
for assumptions, pseudocode and evaluation constraints.

## Experimental results

- **Model Comparison:** Held-out exact-source accuracy, graph error, and
  condition-specific performance across original/augmented models. The graph
  method appears on held-out data only if simulation parameter metadata exists.
- **Adaptive Sampling:** Matched static/no-sample, one-random-sample, and
  one-information-gain-sample runs on identical seeded events. Compare
  exact-source accuracy, graph error, additional sample budget, and results when
  the true source's own initial sensor is missing.
- **Experiment Lab:** Independent Argmax-only repeated trials.
- **Research Method:** Algorithm names, training procedures and interpretation
  caveats, suitable for a student audience rather than a presenter checklist.

Saved final evaluation covers 100 independent events per condition: Normal,
High Noise, Missing Readings and Combined Stress. Check actual computed
metrics before describing gains. A single successful seed is not evidence of
aggregate superiority. Probability scores assume a model and are not universal
confidence guarantees.

## Test suites

```powershell
python -m pytest -q
python -m pytest Tests/test_robust_training.py Tests/test_graph_likelihood.py Tests/test_graph_sampling_integration.py Tests/test_ml_pipeline.py -q
```

Open `Docs/APS_Demo_Runbook.md` for the off-screen presentation sequence.
Keep the repository's original APS simulation modules and source datasets
intact; model improvements are layered on top, not substitutions for evidence.
