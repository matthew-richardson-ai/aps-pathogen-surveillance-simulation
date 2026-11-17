# APS: One-Step Adaptive Sensor Selection

This feature implements and tests the central course research question: Can choosing one additional simulated sensor measurement based on available evidence improve source localization compared with an equally expensive random measurement?

## Run locally

From the repository root, after switching to this feature branch:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r OB/requirements.txt
python -m pytest Tests/test_adaptive_policy.py Tests/test_ml_pipeline.py -q
python -m streamlit run OB/live_dashboard.py
```

Open `http://localhost:8501`.

The **Live facility** tab now offers both a manual sample and an **AI selects next zone** button. Choose Logistic Regression or Random Forest as the selector. The chosen zone is outlined on the map after sampling; the initial and updated predictions can be inspected in the Model comparison tab.

The **Adaptive sampling** tab runs a matched comparison of three strategies:
- `static`: no additional measurement
- `random`: one randomly chosen unobserved zone
- `information_gain`: one unobserved zone selected to maximize expected reduction in uncertainty about the source

All three use the same initial scenario, same trained model, and one post-sample classifier update for the two one-sample strategies. Results include exact-source accuracy, mean shortest-path error, sample budget, per-scenario records, and downloadable CSV.

## How the strategy works

1. Start from the observed eight-zone sensor readings and a trained model's class probabilities for the possible source zones.
2. Apply a small uniform smoothing term to avoid ruling out source locations solely because Random Forest assigned an exact zero probability.
3. For each currently unobserved zone, compute the probability distribution of a new reading under each possible source using the same graph-distance decay model as APS.
4. Estimate expected entropy after the reading by numerically integrating over possible clipped sensor readings.
5. Choose the zone with the largest expected reduction in source entropy.
6. Let the simulator reveal one noisy measurement at that zone, then run the same model again on the updated readings.

No part of step 1-5 sees `true_source`, `ideal_signals`, or the extra sample's actual value.

## Evaluation methodology

The matched experiment runs independently seeded synthetic scenarios with randomized hidden sources. Its seeds are offset from the default Justin dataset seeds. The initial model is trained using Justin's designated training partition; the existing test partition is used for a separate conventional classifier evaluation.

The same synthetic scenario is reused for the static, random, and information-gain comparison; however, the one-sample strategies can select different zones. Both random and information-gain strategies receive exactly one extra measurement, while the static comparator receives none.

Because all observations come from the same simplified family of synthetic simulators, the experiment does not establish generalization to a physical facility, clinical scenario, real airflow, or a different data-generating process.

## Critical limitations

- This method is model-assisted expected information gain; it does not prove the classifier's probabilities are calibrated.
- It assumes that the graph-distance signal model, the scenario's decay and source intensity, and the additional-sample noise are available to the selection policy.
- It is a **single spatial snapshot** and does not simulate contamination spreading through time.
- The method is not guaranteed to beat random sampling. Report negative as well as positive gains and evaluate across seeds and difficulty levels.
- The follow-up sensor is modeled as an independent manual instrument which succeeds even when an installed sensor is missing. Only one extra observation is allowed.
- The evaluation is synthetic-only, and no biological diagnostics are performed.

## Suggested next validation

Run at least 100-200 matched scenarios at several noise/missingness settings. Examine accuracy gains and graph-distance error by condition, and report sample budgets. Then validate the complete Streamlit interface in VS Code, review with Owen, and prepare the November 25 report figures.
