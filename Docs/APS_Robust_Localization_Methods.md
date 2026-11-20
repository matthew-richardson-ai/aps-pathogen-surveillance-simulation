# APS Robust Localization: Methods, Evaluation and Honest Limitations

**CSC 4240 | Simulation-only study | No physical pathogen detection**

This note describes the upgraded APS source-localization pipeline in a way
students can reproduce and reviewers can audit. The eight-zone NetworkX graph
and the existing source, noise and missing-reading simulator are unchanged.

## What was weak in the original classifiers?

Justin's saved synthetic training cases contain a mix of source locations,
noise and missing readings, but the live Combined Stress demonstration may
start with four sampled rooms and end up with only one or two usable readings.
This creates a **training/inference observation-coverage shift**. Fitting a
supervised model only to the original training views may not adequately prepare
it for very sparse observations.

An isolated wrong prediction, including seed 82, is not proof of a software
bug. Some observed patterns genuinely support multiple source locations.

## Methods being compared

| Method | Training data | Input information | Intended role |
|---|---|---|---|
| Argmax Baseline | None | Available readings | Simple deterministic control |
| Logistic Regression (Original) | Original training rows | Available readings + missingness flags | Historical supervised control |
| Random Forest (Original) | Original training rows | Same | Historical supervised control |
| Logistic Regression | Original + augmented *training-only* views | Same | Robust supervised candidate |
| Random Forest | Original + augmented *training-only* views | Same | Robust supervised candidate |
| Graph-Aware Likelihood (Known Params) | None | Available readings **and configured intensity, distance decay, noise** | Parameter-aware inference reference |

Do **not** describe graph likelihood as having the same information budget as
the learned models. It is purposefully a more informed physics-based reference,
not a fair apples-to-apples ML input comparison.

## Robust training-only augmentation

Implemented in `Localization/robust_training.py`.

1. Load Justin's preassigned `train` and `test` partitions. Preserve
   `group_id` separation to prevent persistent-sensor group leakage.
2. Fit the original Logistic Regression and Random Forest on the untouched
   training rows.
3. For the robust versions only, keep every original training row and create
   three reproducible, randomly masked training views per row.
4. Each added view samples a **potential initial sensor budget of 2–7 rooms**,
   retains readings only if those were genuinely present in the original row,
   and sometimes adds a bounded perturbation (noise SD 0, 0.04, 0.12 or 0.24).
5. Recalculate all eight missingness indicators from the altered values.
6. Fit new classifiers on these views, with the same 16-feature input schema.

Augmentation is an approximation of sparse/noisy observations; altered views
of a single event are **not independent events**. No validation or held-out
test rows are augmented, copied, or fitted. No source truth, scenario metadata,
ideal signals, or test labels are inference features of the ML classifiers.

The dataset itself is **not overwritten**. The standard and robust models
remain separately scored on identical held-out observations, so any performance
regression is visible.

## Graph-Aware Likelihood (Known Params)

Implemented in `Localization/graph_likelihood.py`.

For each proposed source zone `s`, compute the expected signal at every
observed room `i`:

```text
mean(i | s) = intensity * exp(-decay_rate * graph_distance(s, i))
```

Score that hypothesis with independent Gaussian observation noise, ignoring
unavailable sensor readings rather than treating them as zero. Because the
simulator clips readings to [0,1], the method uses tail probabilities for
observed zeros and ones (censored Gaussian likelihood) and a regular Gaussian
density for readings strictly between those limits. The model assumes a flat
prior over eight possible sources and normalizes candidate likelihoods. It
returns the most probable candidate and a full eight-zone ranking.

For one follow-up observation, its configured noise SD is **0.04**, distinct
from the initial sensor noise. If all sensors are missing, the displayed
prediction abstains; it never fabricates a source estimate.

**Assumptions:** The graph and exponential signal model are exactly the
simulator's, and `source_intensity`, `decay_rate` and `noise_std` are
provided from the scenario controls. No true source ID or ideal signals are
provided for prediction. On a real facility these physical parameters would
need to be estimated; calibration and correlated sensor errors are not modeled.
The normalized scores are *conditional on these assumptions* and should not
be claimed as real-world confidence.

## One additional measurement

`OB/adaptive_experiments.py` runs three matched policies for every source
localization method:

- **Static:** Use no extra measurement.
- **Random:** Draw one currently unobserved room uniformly.
- **Information gain:** Choose one currently unobserved room with the highest
  estimated entropy reduction from the method's source probabilities.

The random and informed policies both receive exactly one measurement with
the simulator's same `sample_one_more` function. The true source is not used
to choose a candidate location or produce an initial prediction. Source truth
is consulted only *afterward*, for scoring.

Information gain is not the same objective as exact-source classification
accuracy. Intelligent selection can therefore lose to random selection.

## Evaluation and interpretation

Run from the project root:

```powershell
python -m pytest -q
python Scripts/prepare_demo_results.py
python -m streamlit run OB/live_dashboard.py
```

The preparation script evaluates **100 independent seeds per condition** for
Normal Conditions, High Noise, Missing Readings and Combined Stress. All
methods and sampling strategies are assessed on the same event seeds and
measurement budgets. Original and robust models are compared on the same
untouched held-out Justin test partition as well.

The output folder is `Results/Demo/`. Inspect actual measured values rather
than reporting an assumed gain; regenerate the files after changing models.
The results record:

- Exact-source accuracy (all trials count, abstention is incorrect).
- Graph-distance error (defined only for predictions, not abstentions).
- Number of extra readings per event (zero or one).
- Accuracy when the true source's **initial** sensor reading was unavailable,
  using the true label **only for retrospective evaluation**.

The 3D cutaway is a visualization of the eight graph nodes, not an architectural
floor plan, airflow calculation, time simulation or clinical surveillance
system.

## Interpreting changes

A larger training set or a parameter-aware likelihood can improve difficult
cases but does not guarantee it. Report the differences as signed percentage
points, compare all conditions and emphasize *where* models fail. The demo
does not use one handpicked random seed as evidence of improvement. Keep the
original model controls in the final results rather than silently replacing
weaker outcomes.

The graph-aware method and robust classifiers are **comparators**, not evidence
of real-world pathogen source localization beyond this synthetic setup.
