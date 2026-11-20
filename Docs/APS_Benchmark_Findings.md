# APS Benchmark Findings — Measured Synthetic Results

> **HISTORICAL BENCHMARK — PREVIOUS ALGORITHM (November 19, 2026)**
>
> The Graph-Aware Likelihood numbers in this document came from the old
> **Known Params** algorithm. That method was provided the true simulation
> intensity, decay and noise parameters. It has now been replaced in the
> Streamlit demonstration by **Graph-Aware Bayesian Inference**, which does
> *not* receive those parameters and instead marginalizes over a predefined
> parameter grid. **Do not quote the old graph-model accuracy percentages
> as current results.** Rerun `python Scripts/prepare_demo_results.py` and
> use the newly generated `Results/Demo/` CSV files to evaluate the revised
> blind system. The old measurements are retained solely as historical records.
>

**Evaluation source:** GitHub Actions `APS Robustness Benchmark` run
`44646424256`, commit `61a7eabe129286685ea4d6abebf68163ae09561f`,
executed November 19, 2026 (US Central) / November 20 UTC.

**Data:** Justin's saved synthetic dataset, 7,000 original training rows,
28,000 total original-plus-augmented training views, **1,500 untouched
held-out test events**. The sampling-policy experiment evaluates **100 fresh
seeded events in each of four conditions**, with five localization methods
and three strategies. These are **simulated experiments**, not clinical data.

The generated CSV and JSON metadata are archived under GitHub Actions artifact
`aps-robustness-evaluation` for the run above. Dashboard output files are
generated at `Results/Demo/` when the same script is executed locally.

## Exact-source accuracy on the unchanged held-out set

| Method | Correct-source accuracy | Mean graph-distance error |
|---|---:|---:|
| Argmax Baseline | 86.20% | 0.185 edges |
| Logistic Regression (Original) | 92.00% | 0.131 edges |
| Logistic Regression (Robust) | 92.47% | 0.128 edges |
| Random Forest (Original) | 92.93% | 0.118 edges |
| Random Forest (Robust) | 93.00% | 0.119 edges |
| Graph-Aware Likelihood (Known Params) | 94.27% | 0.097 edges |

**Interpretation:** On this broad held-out mix, augmenting the ML training
views improves exact-source accuracy only modestly. Random Forest's mean graph
error actually worsens slightly. The graph-aware reference has an **explicit
information advantage**: it receives simulator intensity, decay and noise
parameters that neither ML classifier receives.

## Repeated-trial sampling: 100 events per condition

Each value below is exact-source accuracy (%) at the stated initial sensor
budget. The **Random** and **Information Gain** columns each use exactly one
additional measurement; **Static** uses none.

| Condition | Method | Static | Random | Information Gain |
|---|---|---:|---:|---:|
| Normal | Logistic Regression (Original) | 95% | 99% | 100% |
| Normal | Logistic Regression (Robust) | 97% | 99% | 100% |
| Normal | Random Forest (Original) | 97% | 98% | 100% |
| Normal | Random Forest (Robust) | 99% | 99% | 100% |
| Normal | Graph-Aware Likelihood | 96% | 99% | 100% |
| High Noise | Logistic Regression (Original) | 71% | 80% | 80% |
| High Noise | Logistic Regression (Robust) | 73% | 81% | 84% |
| High Noise | Random Forest (Original) | 70% | 79% | 77% |
| High Noise | Random Forest (Robust) | 67% | 78% | 79% |
| High Noise | Graph-Aware Likelihood | 72% | 94% | 97% |
| Missing Readings | Logistic Regression (Original) | 67% | 79% | 82% |
| Missing Readings | Logistic Regression (Robust) | 73% | 94% | 93% |
| Missing Readings | Random Forest (Original) | 65% | 83% | 86% |
| Missing Readings | Random Forest (Robust) | 77% | 91% | 92% |
| Missing Readings | Graph-Aware Likelihood | 76% | 94% | 97% |
| Combined Stress | Logistic Regression (Original) | 22% | 27% | 33% |
| Combined Stress | Logistic Regression (Robust) | 20% | 23% | 27% |
| Combined Stress | Random Forest (Original) | 21% | 28% | 29% |
| Combined Stress | Random Forest (Robust) | 16% | 23% | 27% |
| Combined Stress | Graph-Aware Likelihood | 36% | 64% | 68% |

## Results for the difficult scenarios

**Missing Readings:** Robust training helps both learned classifiers.
Without extra samples, Logistic Regression improves from **67% to 73%** (+6
percentage points) and Random Forest from **65% to 77%** (+12 points).
With information-gain sampling, robust Logistic Regression reaches 93% and
robust Random Forest reaches 92%, versus 82% and 86% for their original
counterparts.

**Combined Stress:** Robustly augmented ML models perform *worse* than
their original variants in this run: static Logistic Regression 20% versus
22%, Random Forest 16% versus 21%. Graph-Aware Likelihood achieves 36% static,
64% with a random additional measurement, and **68% with an information-gain
measurement**. It benefits from knowing the simulator's true parameter
settings; this is **not** evidence that a data-only ML approach reaches 68%.

**High Noise:** Graph-aware information gain achieves 97% versus 94% random.
Robust Logistic Regression improves slightly, while robust Random Forest
slightly loses relative to its original control. These effects are not all
positive, and no statistical significance is claimed from 100 trials.

**Seed 82:** The true source is Zone 3 with only two initially available
readings. Graph likelihood ranks Zone 4 first and Zone 3 tied for second
(alongside Zone 6) before an additional measurement; the new method does
**not** magically recover an unidentifiable source from two noisy readings.

## How to reproduce and interpret

```powershell
python -m pytest -q
python Scripts/prepare_demo_results.py
```

The source code retains the exact preassigned train/test split and runs all
sampling strategies on matched seeded events. Result CSVs contain the individual
truth labels, predictions and graph errors for auditing. In the app, select
the evaluation condition and compare methods, including negative outcomes.
Do not state a gain for AI sampling if random is tied with or ahead of it.

A scientific extension would evaluate more seeds, uncertainty intervals,
additional domain shifts, probability calibration and the model misspecification
risk of giving the graph estimator perfect simulation parameters.
