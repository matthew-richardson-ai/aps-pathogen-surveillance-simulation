# APS Final Results

**CSC 4240, Adaptive Pathogen Surveillance**

These are the measured results from our completed APS simulation and the current graph-aware Bayesian model. They come from running `python Scripts/prepare_demo_results.py` on the `feature/demo-ready-aps` branch. This document is intended to accompany the four-page project report and help others reproduce our evaluation.

The earlier [APS Benchmark Findings](APS_Benchmark_Findings.md) document is a historical comparison. Its original graph model was given the simulation's actual signal parameters. Our current graph-aware Bayesian implementation does not receive those hidden parameters, so the older graph-model results should not be compared as if they came from the same method.

## 1. What we evaluated

APS models a facility as eight connected zones. Each simulated event has one hidden source, with signals that weaken as graph distance increases. Sensors can have noisy or missing readings. The task is to predict which zone contains the source.

We compared six approaches on the same held-out dataset:

- Strongest-sensor (Argmax) baseline, which selects the largest available reading.
- Logistic Regression and Random Forest trained on the original dataset.
- Logistic Regression and Random Forest trained with additional noisy and incomplete **training-only** views. We call these the robust variants.
- Graph-Aware Bayesian Inference, which evaluates possible sources using the known facility graph and plausible signal settings.

For adaptive sampling, we compared **five model variants** (the two original classifiers, two robust classifiers, and the Bayesian method) under three strategies. The static strategy takes no extra measurement. Random sampling and information-gain sampling each receive exactly **one** additional measurement. The same simulated event is used across these strategies, allowing paired comparisons.

The model-training dataset includes 7,000 original training events and 28,000 training views after augmentation. The held-out evaluation uses **1,500 test events** that were not used for training. The adaptive experiment uses **100 new events in each of four conditions**, or 400 distinct simulated events in total. Because models and strategies are compared on these same events, the 6,000 adaptive records are not 6,000 independent outbreaks.

### Conditions used in the adaptive experiments

| Condition | Initially selected sensors | Initial noise SD | Missing rate | Source intensity | Decay rate |
|---|---:|---:|---:|---:|---:|
| Normal Conditions | 6 | 0.06 | 0.05 | 1.00 | 0.65 |
| High Noise | 5 | 0.26 | 0.10 | 1.00 | 0.65 |
| Missing Readings | 6 | 0.08 | 0.50 | 1.00 | 0.65 |
| Combined Stress | 4 | 0.22 | 0.35 | 0.45 | 0.95 |

These are controlled simulation settings, not measured rates from physical monitoring equipment. The missing rate is applied probabilistically, so the number of available sensor readings varies across events.

**Follow-up measurement assumption:** The extra manual measurement always succeeds and uses a fixed noise standard deviation of **0.04**. This can be cleaner than the initial readings. Any improvement over the static strategy therefore reflects the value of obtaining an additional, potentially higher-quality measurement. To evaluate the choice of sampling location, the fair comparison is information gain versus random sampling, since both use the same extra-sample budget.

The current source-localization and sampling methods do not receive the hidden source label or actual event-specific intensity, decay, or initial-noise parameters. The true source is used afterward to score the predictions.

## 2. Held-out source-localization results

All methods were tested on the same 1,500 held-out examples.

| Method | Exact-source accuracy | Mean graph error (edges) |
|---|---:|---:|
| Strongest-sensor baseline | 86.20% | 0.1853 |
| Logistic Regression (Original) | 92.00% | 0.1307 |
| Logistic Regression (Robust) | 92.47% | 0.1280 |
| Random Forest (Original) | 92.93% | 0.1180 |
| Random Forest (Robust) | **93.00%** | 0.1187 |
| Graph-Aware Bayesian Inference | 91.47% | 0.1320 |

Random Forest with robust training had the highest exact-source accuracy at **93.0%**, compared with **86.2%** for the strongest-sensor baseline. The difference between the original and robust Random Forest was only **one correct prediction out of 1,500**. Its mean graph error also increased slightly, from 0.1180 to 0.1187 edges. We therefore cannot claim that augmentation consistently improves Random Forest performance.

Exact-source accuracy is the fraction of events for which the predicted zone matches the true source. Graph error is the shortest-path distance between the predicted and actual source zones. An abstention counts as incorrect for accuracy; graph error averages exclude cases where there was no predicted zone.

## 3. Adaptive sampling results

Each row below uses 100 matched simulated events. Values show exact-source accuracy. The static strategy has no extra sample; the other two strategies each take one.

| Condition | Model | Static | Random +1 | Information gain +1 |
|---|---|---:|---:|---:|
| Normal Conditions | Logistic Regression (Original) | 95% | 99% | 100% |
| Normal Conditions | Logistic Regression (Robust) | 97% | 99% | 100% |
| Normal Conditions | Random Forest (Original) | 97% | 98% | 100% |
| Normal Conditions | Random Forest (Robust) | 99% | 99% | 100% |
| Normal Conditions | Graph-Aware Bayesian Inference | 90% | 96% | 99% |
| High Noise | Logistic Regression (Original) | 71% | 80% | 80% |
| High Noise | Logistic Regression (Robust) | 73% | 81% | 83% |
| High Noise | Random Forest (Original) | 70% | 79% | 81% |
| High Noise | Random Forest (Robust) | 67% | 78% | 79% |
| High Noise | Graph-Aware Bayesian Inference | 67% | 76% | 80% |
| Missing Readings | Logistic Regression (Original) | 67% | 79% | 83% |
| Missing Readings | Logistic Regression (Robust) | 73% | 94% | 100% |
| Missing Readings | Random Forest (Original) | 65% | 83% | 83% |
| Missing Readings | Random Forest (Robust) | 77% | 91% | 90% |
| Missing Readings | Graph-Aware Bayesian Inference | 55% | 69% | 77% |
| Combined Stress | Logistic Regression (Original) | 22% | 27% | 36% |
| Combined Stress | Logistic Regression (Robust) | 20% | 23% | 25% |
| Combined Stress | Random Forest (Original) | 21% | 28% | 30% |
| Combined Stress | Random Forest (Robust) | 16% | 23% | 27% |
| Combined Stress | Graph-Aware Bayesian Inference | 36% | 56% | 57% |

We found three notable patterns:

1. **One more measurement often helps.** Under Missing Readings, robust Logistic Regression increased from 73% without a follow-up measurement to 94% with a random follow-up and 100% with information-gain sampling. Its information-gain advantage over random sampling was **6 percentage points**, not 27.
2. **Information gain is not always the better choice.** For robust Random Forest under Missing Readings, random sampling reached 91% and information-gain sampling reached 90%. We should report such cases rather than treating every improvement over the static strategy as evidence for intelligent selection.
3. **The graph-aware method performed best among these methods under Combined Stress.** It reached 57% accuracy with information-gain sampling, versus 27% for robust Random Forest. However, 57% still means 43 incorrect predictions in 100 trials. This is a result for the chosen synthetic scenario, not evidence of reliable real-world localization.

### Paired statistical checks

Because the strategies were evaluated on the same events, we can compare their predictions directly.

In the 100 Missing Readings trials for robust Logistic Regression, information-gain sampling correctly classified six events that random sampling missed, and random sampling did not correctly classify any event that information gain missed. An **exact two-sided paired McNemar test** gives **p = 0.03125** for that particular comparison.

Across all four conditions together (400 matched events per model), the advantage of information gain over random was:

| Model | Accuracy gain over random | Exact paired test p-value |
|---|---:|---:|
| Logistic Regression (Robust) | +2.75 percentage points | 0.099 |
| Random Forest (Robust) | +1.25 percentage points | 0.522 |
| Graph-Aware Bayesian Inference | +4.00 percentage points | 0.068 |

The aggregate comparisons do not meet a conventional 0.05 significance threshold. The single-condition p-value is exploratory and has not been adjusted for testing multiple combinations of models and conditions. We therefore cannot conclude that information-gain sampling consistently outperforms a budget-matched random sample.

## 4. Limitations and interpretation

APS is an **eight-zone synthetic spatial snapshot model**. It does not model time-dependent pathogen transmission, ventilation or airflow in a real building, physical sensor failure and recovery, real clinical observations, or outbreak-free events. All simulated events contain one source.

The Bayesian method is blind to each event's actual hidden signal parameters, but it still benefits from knowing the correct graph topology and assuming a family of exponential signal patterns. Its probabilities are model-dependent and have not been calibrated as real-world confidence scores.

The experiment uses 100 seeded events per condition. A perfect score in one condition should not be treated as a general guarantee. Larger independent replications, uncertainty intervals, different facility layouts, and realistic sensor measurements would be needed to establish how well the approach generalizes.

## 5. Reproduce these results

Clone the completed project branch or use the submitted project folder. From the repository root, with **Python 3.11** and an active virtual environment:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python Scripts/prepare_demo_results.py
python -m streamlit run OB/live_dashboard.py
```

The evaluation script regenerates the following files in `Results/Demo/`:

- `heldout_model_summary.csv` and `heldout_model_predictions.csv`
- `sampling_summary.csv` and `sampling_runs.csv`
- `experiment_metadata.json`

The CSVs contain the underlying truth labels, model predictions, graph-distance errors, and matched trial records. They are intentionally excluded from Git by `.gitignore`; regenerate them using the command above or include them separately in the course submission package.

See the root [README](../README.md) for complete setup instructions, the dashboard guide in [OB/LIVE_DASHBOARD_README.md](../OB/LIVE_DASHBOARD_README.md), and [APS Robust Localization Methods](APS_Robust_Localization_Methods.md) for technical assumptions.
