# APS Final Synthetic Evaluation — November 23, 2026

**CSC 4240 / APS — Reproducible run of `python Scripts/prepare_demo_results.py`**

This document records the **current blind graph-inference implementation**, not the historical Known Params benchmark in `APS_Benchmark_Findings.md`. Values below were transcribed from the completed local experiment output; raw generated files are deliberately ignored by Git and must be regenerated or packaged separately.

## Experimental design

- Model training: 7,000 independent synthetic training events, including 28,000 original-plus-augmented training views.
- Held-out testing: 1,500 untouched examples, with identical held-out cases across methods.
- Adaptive comparison: four conditions × 100 independent seeded synthetic events = **400 events**. Five model variants use the same underlying event per condition, and each is evaluated under **static**, **one random sample**, and **one information-gain sample**.
- Four conditions: Normal Conditions, High Noise, Missing Readings, Combined Stress.
- Measurements: follow-up sample uses a clean fixed noise standard deviation **0.04**, even when initial readings were generated under noisier conditions. Treat comparisons as *an extra higher-quality measurement*, not simply another identical sensor measurement.
- Source location and actual event-specific intensity/decay/noise parameters are not provided to the inference and blind sampling policies. The true source is used only after inference for evaluation.
- All numbers are from simulated graphs and readings, not physical pathogen measurements.

## Held-out performance (1,500 cases per method)

| Method | Exact-source accuracy | Mean shortest-path graph error (edges) |
|---|---:|---:|
| Strongest-sensor (Argmax) baseline | 86.200% | 0.1853 |
| Logistic Regression (Original) | 92.000% | 0.1307 |
| Random Forest (Original) | 92.933% | 0.1180 |
| Logistic Regression (Robust) | 92.467% | 0.1280 |
| Random Forest (Robust) | **93.000%** | 0.1187 |
| Graph-Aware Bayesian Inference (blind) | 91.467% | 0.1320 |

Robust RF's advantage over original RF is **0.067 percentage points (one extra correct held-out case)**, not a large general gain. RF robust graph-distance error is also fractionally worse than original.

## Selected adaptive sampling results (exact-source accuracy, n = 100 per row)

| Condition | Model | Static | Random +1 | Information gain +1 |
|---|---|---:|---:|---:|
| Normal | Logistic Regression (Robust) | 97% | 99% | 100% |
| High Noise | Logistic Regression (Robust) | 73% | 81% | 83% |
| Missing Readings | Logistic Regression (Robust) | 73% | 94% | **100%** |
| Combined Stress | Logistic Regression (Robust) | 20% | 23% | 25% |
| Missing Readings | Random Forest (Robust) | 77% | 91% | 90% |
| Combined Stress | Random Forest (Robust) | 16% | 23% | 27% |
| Combined Stress | Graph-Aware Bayesian Inference | 36% | 56% | **57%** |

The single most favorable condition for robust Logistic Regression was Missing Readings: +27 percentage points over static, **but only +6 points over the budget-matched random sample**. Among those paired 100 trials, the information-gain policy corrected six examples that random got wrong and did not lose any that random got right; exact two-sided McNemar p = 0.03125. This is **exploratory and unadjusted for the number of model-condition comparisons**.

Aggregating the four conditions for each robust model (400 events/model), information-gain gains over random sampling were:
- Logistic Regression: +2.75 percentage points (paired exact test p ≈ 0.099).
- Random Forest: +1.25 percentage points (p ≈ 0.522).
- Graph-Aware Bayesian Inference: +4.00 percentage points (p ≈ 0.068).

These aggregated differences do **not** reach p < 0.05 on the quoted tests. Overall evidence that information gain consistently beats budget-matched random sampling is **inconclusive** at this sample size.

## Main findings and limitations

1. The strongest held-out method here is robust Random Forest (93.0% exact-source accuracy), substantially above the 86.2% simple Argmax baseline on the same held-out cases.
2. The largest apparent adaptive gains relative to static sampling often come from obtaining **one more observation**. For assessing sample-location intelligence, compare information-gain against **random with the same measurement budget**.
3. Blind graph-aware inference is competitive in the specified difficult Combined Stress experiment (57% with information gain vs. 27% robust RF) but still produces many incorrect predictions. It assumes the simulated spatial graph and an idealized signal family.
4. Accuracy is limited to synthetic spatial snapshots; no clinical utility, pathogen detection, temporal transmission prediction, or real building airflow is established.
5. Trial count is 100 per condition; selected perfect results can be seed-specific. Follow-up samples have a fixed cleaner instrument model. A larger independent seed sweep, bootstrap/paired uncertainty intervals, and model calibration are future work.
6. The historical benchmark has an **obsolete graph method** that knew true event parameters; never mix its percentages with the current blind approach.

## Reproduce

```powershell
python -m pip install -r requirements.txt
python -m pip install -r OB/requirements.txt
python -m pytest -q
python Scripts/prepare_demo_results.py
python -m streamlit run OB/live_dashboard.py
```

The generator creates `Results/Demo/heldout_model_summary.csv`, `heldout_model_predictions.csv`, `sampling_runs.csv`, `sampling_summary.csv`, and `experiment_metadata.json`. Those raw outputs remain excluded by `.gitignore` and should be preserved alongside the final report/code submission if requested.
