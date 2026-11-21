# Adaptive Pathogen Surveillance (APS): Visualization and Evaluation Plan

**CSC 4240 | Assigned deliverable: performance charts, model comparison, final demo interface**  
**Status:** Visual prototypes and plotting code prepared; model statistics require the team's repeated simulation results.

## 1. Facility visualization

The project currently uses a fixed eight-zone NetworkX graph. Draw one node per zone, and one edge per connection. Color each node by a **normalized simulated signal reading** (0–1); gray means a reading is missing. Use a green ring to identify the predicted source, a blue star to reveal the simulator's true source **after a prediction**, and an optional purple dashed ring for the next sample selected by the adaptive algorithm. Label the figure with a scenario seed and with the noise and missing-data settings. Graph distances are numbers of *edges*, not measurements of actual physical length.

The included `facility_preview.png` uses **illustrative topology and sensor values**, because the proposal does not supply the team's current graph edge list or an experiment file. It must not be presented as actual APS experimental output. The `aps_viz.py` renderer accepts a real NetworkX graph; `aps_app.py` accepts a JSON export of the actual graph and optional normalized sensor readings. The example next-sample marker is a visual placeholder, **not** the product of an adaptive method.

## 2. Evaluation graphics

| Priority | Figure | Chart type | X-axis | Y-axis / meaning | Comparison |
|---|---|---|---|---|---|
| Essential | Exact-source accuracy | Bar chart | Strongest sensor, Logistic Regression, Random Forest; static/adaptive where appropriate | % of predictions matching the true source | Higher is better; include number of test runs |
| Essential | Localization error | Bar chart | Same methods | Mean shortest-path distance between predicted and true zone (edges) | Lower is better, even when a guess is not exactly right |
| Essential | Noise and missingness | Heatmap | Missing-data fraction | Noise standard deviation; cell color = exact-source accuracy | Examine robustness at several combinations, keeping conditions comparable |
| Essential | Adaptive benefit | Paired bar chart | Model or strategy | Accuracy difference in *percentage points* (adaptive minus static) | Pair each static/adaptive prediction on the same outbreak trial ID |
| Helpful | Zone confusion | 8×8 confusion matrix | Predicted zone | True source zone; cell = count | See which sources get mixed up |
| Helpful | Confidence calibration | Reliability curve | Mean predicted confidence | Observed exact accuracy | See whether stated confidence is informative |

**Calculations:**

- `exact_accuracy = number of correct predictions / number of predictions`.
- `distance_error = shortest_path_length(G, true_source, predicted_source)`; average over trials.
- `adaptive_gain_pp = 100 × (accuracy_adaptive − accuracy_static)`, comparing matched trials.
- When reporting accuracy, include `n` for each series. Where possible, calculate uncertainty across independent runs (e.g. bootstrap intervals) for the final report.

**Important fairness rules:** Every model should be evaluated on the same held-out outbreak scenarios, with independent train and test random seeds and no use of the hidden source as an input feature. For each matched outbreak scenario, record a static prediction, and record the prediction after the adaptive method has the option to request **at most one** additional measurement. Adaptive methods spend an extra sample when triggered; report the fraction of trials that triggered a sample and mean additional samples per run. If a budget-matched random sampling comparator is available, include it in the final report to test whether adaptive *targeting*, not just extra information, caused the gain. Do not present the before/after delta as a controlled benefit of targeting unless sampling budgets are comparable.

## 3. Interface structure

**Tab 1 – Facility / sampling:** Show 8-zone network, sensor values, missing sensors, strongest-sensor prediction, optional true-source reveal, and eventual algorithm-selected next sample. Expose a random seed and noise/missingness sliders when preview data are used. Clearly label a preview as *illustrative*.

**Tab 2 – Model evaluation:** Read a single CSV of held-out results, recompute exact accuracy and graph distance from source labels, plot comparison graphics, allow model selection for the heatmap and confusion matrix, and export a scored results table.

**Tab 3 – Demo notes:** Provide a short presentation sequence and explicit statement that this is a simulation-only AI class project.

## 4. Team data needed

1. **Facility edges and node labels:** export the team's NetworkX graph to the schema in `sample_graph.json`; the example JSON is not necessarily the real topology.
2. **Sensor output:** optionally export `zone,signal` with blank readings for missing sensors; normalize readings to 0–1 for this interface.
3. **Repeated experiment results:** one row per `(run_id, model, mode)` containing `run_id`, `model`, `mode`, `true_source`, `predicted_source`. Recommended additional columns: `seed`, `confidence`, `noise_sd`, `missing_fraction`, `sample_count`, `sampled_zone`.
4. **Model names and test settings:** agree on source IDs (e.g. `Z1` … `Z8`), model labels, noise grid, missingness grid, held-out test seeds, and confidence definitions.

Example meaning of the schema: the **same `run_id`** should appear once for each model and sampling mode being compared; `mode` is `static` or `adaptive`. `sample_count` is the number of additional targeted samples, **not** the number of initial sensors. Logistic Regression and Random Forest confidence can come from `predict_proba` as the maximum class probability, but these raw probabilities are not necessarily well calibrated.

## 5. Recommended final demonstration (about 3–4 minutes)

1. **Explain the facility:** "This graph shows eight connected simulated zones, not a real building or clinical measurements."
2. **Show imperfect evidence:** reveal missing signals and noisy readings. Show the strongest-sensor baseline.
3. **Compare AI to baseline:** display held-out Logistic Regression and Random Forest accuracy and mean graph error.
4. **Show adaptive action:** demonstrate a low-confidence case, let the team's adaptive method choose one new sample, then show the revised prediction and before/after aggregate evaluation.
5. **Conclude with robustness:** show the noise/missingness heatmap and explain where performance deteriorates.

## 6. Completion checklist

- [ ] Replace the illustrated 8-zone graph with the real team graph.
- [ ] Obtain results for the strongest-sensor baseline, Logistic Regression, and Random Forest on identical test runs.
- [ ] Gather the static and one-step adaptive results with matching `run_id` values.
- [ ] Verify graph-distance correctness against NetworkX shortest-path calculations.
- [ ] Generate the four essential graphics from real experiments.
- [ ] Optionally add confusion matrix and confidence calibration plots.
- [ ] Make the displayed simulation/demo state traceable to a seed.
- [ ] Verify no future-scope genomic/response functionality appears in the class demo.
