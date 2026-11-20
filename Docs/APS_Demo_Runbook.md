# APS Classroom Demonstration: Presenter Runbook

CSC 4240 | Adaptive Pathogen Surveillance | Simulation-only research project

## Purpose

Demonstrate the approved question: Can a machine-learning model infer the hidden source of a simulated contamination event from noisy, missing sensor readings, and can one strategically chosen additional sample improve the prediction?

Show an actual eight-zone NetworkX facility. No physical sensors, genomics, clinical conclusions, transport simulation, or real pathogen identification is implemented.

## Prepare the machine (at least one day before presenting)

From the repository root with the final demo branch checked out:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r OB/requirements.txt
python -m pytest -q
python Scripts/prepare_demo_results.py
python -m streamlit run OB/live_dashboard.py
```

The experiment script creates real computed files under `Results/Demo`:
- `heldout_model_summary.csv` and `heldout_model_predictions.csv`
- `sampling_summary.csv` and `sampling_runs.csv`
- `experiment_metadata.json`

These output files are *not automatically part of GitHub*. Review their values and submit/commit deliberately. Never state results from prior exploratory runs as though they came from these final experiments.

For a presentation on another computer, bring the exact code branch, Python environment instructions, and CSV files. Test the projector resolution and browser zoom, and launch Streamlit before the audience arrives. Keep the terminal minimized but running.

## Five-minute presentation sequence

### 0:00–0:45 | Why APS exists — Matthew or designated opening speaker

Suggested wording:

> Our question is whether a machine-learning system can identify a hidden contamination source from incomplete environmental readings. We model an eight-zone connected facility and run simulated events where sensors are noisy or missing. Then we test whether one additional targeted measurement can improve the prediction.

Show the **APS Demonstration** view and point to the facility network.

### 0:45–1:40 | How the simulation works — Emmanuel

Suggested wording:

> One zone is assigned as the hidden source. We generate signals that decrease with graph distance, then apply noise and missing observations. The lines connect zones; they do not represent real airflow or transmission. A fixed random seed lets us reproduce the same experiment.

Click **Normal Conditions**, then **Missing Readings** or **Combined Stress**. Do not reveal the source until explaining the prediction.

### 1:40–2:35 | Localization models — Jimmy and Justin

Suggested wording:

> Our baseline is the **Maximum-Sensor-Reading (Argmax) algorithm**. Argmax selects the zone with the highest available sensor value. It is a deterministic rule, not a trained model. Logistic Regression and Random Forest are trained on Justin's synthetic dataset using readings and missingness indicators. We evaluate them on held-out simulated events rather than reporting training accuracy.

Open **Model Comparison** and reference the actual calculated test metrics. If the interface is slow, use precomputed `heldout_model_summary.csv`.

### 2:35–3:40 | One additional intelligent measurement — Matthew / Jimmy

Suggested wording:

> The model looks at the evidence it already has and estimates which unobserved zone would reduce uncertainty the most. It requests only one additional reading, then updates the prediction. We compare that with randomly selecting a zone, giving both policies the same sampling budget.

In the **Live Facility** tab, click **AI Selects Next Zone**. Show the selected node and compare before/after. Reveal the hidden source after discussing the model's prediction. **Do not claim the single demo sample proves an accuracy improvement.**

### 3:40–4:40 | Evaluation and what it means — Owen / Matthew

Suggested wording:

> We repeated the experiment across normal, noisy, missing-reading, and combined-stress conditions. Here are the exact-source accuracy and graph-distance results for no additional sample, one random sample, and one information-gain-selected sample. A positive difference means the targeted policy helped in that setting; a negative difference means random sampling worked better.

Use the **APS Demonstration** summary (from saved experiment results), or the **Adaptive Sampling** tab. Never promise a positive result for both classifiers. Read actual values from the saved experiment.

### 4:40–5:00 | Conclusion and limitations

Suggested wording:

> We demonstrated a functioning simulation, source localization, and an experimentally testable adaptive sampling policy. The findings apply to this synthetic graph-distance model, not to physical pathogen detection. The next research step would be stronger validation and more realistic transport modeling.

Do not use the phrase "real-time pathogen surveillance" for this implementation.


### Updated localization explanation

The dashboard compares the **Argmax Baseline**, robust and original
**Logistic Regression** and **Random Forest**, and **Graph-Aware Bayesian
Inference**. The trained models receive sixteen sensor reading/missingness
features, while the graph method assumes the known graph topology and a
family of exponential signal-decay relationships. It **does not** receive
the event's actual intensity, decay, noise, true source or ideal signals.

The graph estimator sums over plausible nuisance settings instead of being
told which settings the simulator actually used. For adaptive selection,
all methods rely on the observed readings and a predefined, parameter-blind
information-gain policy. The true source is only provided to the scoring
layer **after Reveal True Source**.

Once revealed, explain **Correct?** (exact match), **Graph Distance** (shortest
number of graph edges to the source), and **Error Reduction** (before minus
after; positive means closer). Never claim every individual extra reading
improves the result.

**Important historical-results warning:** Previous benchmarks for
"Graph-Aware Likelihood (Known Params)" used actual event signal/noise
settings. Their published percentages, including 36%→68% Combined Stress,
are valid only for that earlier method. Regenerate the Results/Demo outputs
for this blind version and report its *new* measured outcomes, whether
positive or negative.

## Presenter checklist

- [ ] Confirm the working local branch contains the reviewed dashboard and source data.
- [ ] All tests pass on the presentation machine.
- [ ] Train both original and robust ML variants at least once; confirm graph likelihood works without training.
- [ ] Run and inspect the four-condition experiment; save the CSV outputs.
- [ ] Check that labels use "Zone 1" rather than code identifiers and that all controls are legible on a projector.
- [ ] Prepare two deterministic seeds: one normal and one difficult. Confirm actual outcomes in advance and note the seeds.
- [ ] Avoid live dependency installation or training during the five-minute presentation.
- [ ] Open Streamlit before presenting and close unrelated windows/notifications.
- [ ] Have PDF/screenshot backups of the facility, model metrics, and adaptive comparison.
- [ ] Final report and demo use the same experiment definitions and numeric results.
- [ ] Verify all presenters can describe their own contribution and the full pipeline.

## If something fails live

- If model training is slow: use precomputed `heldout_model_summary.csv` and briefly describe the training inputs.
- If an adaptive run stalls: show the saved `sampling_summary.csv` and the explicit method instead of inventing an on-screen outcome.
- If the browser is unavailable: use prepared screenshots/PDFs and the CSVs, explaining that they were generated in advance by the same code.
- If any score is weaker than expected: report it as a limitation and discuss noise, missingness, and the mismatch between entropy reduction and classification accuracy.

## Questions the professor may ask

**What is the baseline algorithm?** The Maximum-Sensor-Reading (Argmax) heuristic: choose the observed zone with the largest sensor value. It is also known as a maximum-reading or winner-take-all decision rule; it is not a machine-learning classifier.

**Why use a baseline?** It tells us whether trained classifiers outperform a simple deterministic maximum-reading rule.

**What are the ML model inputs?** Eight readings plus eight missingness indicators; no true source or simulation metadata is an inference feature. Graph-aware Bayesian inference instead uses the observed readings and public graph, integrating over unknown intensity, decay, and noise rather than reading scenario settings.

**How is adaptive selection chosen?** Expected reduction in uncertainty about source location, estimated across possible next measurements using the assumed graph-distance signal model.

**Does adaptation always improve prediction accuracy?** No. Information gain and classification accuracy optimize different objectives; compare against random sampling with the same budget.

**Is this pathogen identification or disease detection?** No. The simulator models a hypothetical source and imperfect abstract readings, not a microorganism, diagnosis or measured outbreak.

**Does the map show contamination spreading over time?** No. Each event is a static spatial snapshot. Updating the controls generates a new snapshot.

**Why use the same random seed?** To make a particular scenario reproducible for evaluation and presentation. Aggregate results use many distinct seeds.
