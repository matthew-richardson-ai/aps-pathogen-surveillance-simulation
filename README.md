# Adaptive Pathogen Surveillance (APS)

**CSC 4240 Artificial Intelligence | Tennessee Technological University**

**Team:** Emmanuel Zuniga, Justin Hortopanu, Jimmy Finnegan, Owen Brown, and Matthew Richardson.

APS is a Python project that tests how well existing AI methods can locate the source of a simulated contamination event when sensor readings are incomplete or noisy. It also asks whether choosing a useful place to take one more measurement improves the prediction.

The project uses an eight-zone facility simulation. It does not connect to physical sensors or diagnose real pathogens.

**For the instructor or TA:** The fastest way to review the project is to follow **Run the project** below, then open the Streamlit dashboard. The full evaluation script generates the results displayed in the repeated-trial comparison. You can also run the automated tests to check the code.

## Get the project

**Completed project branch:** [feature/demo-ready-aps](https://github.com/matthew-richardson-ai/aps-pathogen-surveillance-simulation/tree/feature/demo-ready-aps)

The repository is public. The finished dashboard and evaluation code are on this branch, which is not yet merged into the default `main` branch.

If you're using **the course submission ZIP**, extract it and open a terminal in the folder containing this `README.md` and `requirements.txt`. You can skip cloning.

If you're downloading directly from GitHub, clone the completed branch:

```bash
git clone --branch feature/demo-ready-aps --single-branch https://github.com/matthew-richardson-ai/aps-pathogen-surveillance-simulation.git
cd aps-pathogen-surveillance-simulation
```

All remaining commands should be run from that project folder.

## Run the project

You'll need **Python 3.11**. Git is only needed if you are cloning the repository. An internet connection is needed to install Python packages.

### Windows (PowerShell)

**1. Create and activate a virtual environment.**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the `py` launcher is unavailable, use `python -m venv .venv` after checking that `python --version` reports Python 3.11. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and try the activation command again. This changes the policy only for that terminal session.

**2. Install the dependencies.**

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This one requirements file installs everything needed for the simulator, AI models, dashboard, experiment scripts, and tests. The separate `OB/requirements.txt` is kept for older dashboard-only workflows, but you do not need to install it again.

**3. Run the tests.**

```powershell
python -m pytest -q
```

The team most recently verified 66 passing tests. The exact count may change if additional tests are added.

**4. Generate the saved experiment results.**

```powershell
python Scripts/prepare_demo_results.py
```

This step trains and evaluates the models, and creates the CSV files used for the dashboard's repeated-trial charts. It may take longer than the other steps. The dashboard can still launch without these files, but that saved-results section will be empty until this command finishes.

**5. Start the dashboard.**

```powershell
python -m streamlit run OB/live_dashboard.py
```

Open the local URL shown by Streamlit, normally [http://localhost:8501](http://localhost:8501). The first load may take a moment while the models train. Leave the terminal open while using the dashboard. Press `Ctrl+C` in that terminal to stop it.

### macOS or Linux

Use a terminal in the project folder after cloning or extracting the ZIP:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q
python Scripts/prepare_demo_results.py
python -m streamlit run OB/live_dashboard.py
```

Open the Streamlit URL printed in the terminal. Press `Ctrl+C` to stop the application.

## What to explore in the dashboard

The dashboard contains six tabs:

1. **APS Demonstration:** Try preset scenarios, compare predictions, collect a follow-up reading, and reveal the true source after the predictions have been made.
2. **Live Facility:** View the eight connected zones, sensor readings, and missing observations. The 2D layout can be rearranged for display without changing the simulation.
3. **Model Comparison:** Compare the strongest-sensor baseline, Logistic Regression, Random Forest, and graph-aware Bayesian inference.
4. **Adaptive Sampling:** Compare random sampling with information-gain sampling using the same extra-measurement budget.
5. **Experiment Lab:** Run repeated baseline trials and inspect accuracy, a confusion matrix, and performance by number of missing readings.
6. **Research Method:** Read about the selected AI methods and how the experiments were designed.

The dashboard uses synthetic observations generated by the project. Some comparisons are interactive, and others use the saved outputs from the evaluation script.

## How the AI methods work

The strongest-sensor rule is our simple baseline. Logistic Regression learns weighted relationships between sensor readings and possible source zones. The integrated Random Forest combines **120 decision trees** to capture more complex patterns. Both classifiers are evaluated with their original training data and with extra noisy or incomplete training views.

Graph-Aware Bayesian Inference uses the facility connections and plausible signal patterns to compare possible source locations. It does not receive the true source or the hidden signal settings for an event. For follow-up sampling, we compare a random location with a location selected to reduce expected uncertainty.

The project tests these methods rather than assuming one approach must always win. We measure exact-source accuracy and graph-distance error, including under high noise, missing readings, and combined stress.

## Data and experiment outputs

The project includes a synthetic training dataset under `Justin/Data/`. The model loader also recognizes `Team/Justin/Data/` if the team folders are reorganized. **Keep the dataset folder in place** when extracting or packaging the source code. No real pathogen data or physical sensors are needed.

To generate the results again:

```bash
python Scripts/prepare_demo_results.py
```

The script writes five files to `Results/Demo/`:

- `heldout_model_summary.csv`: accuracy and graph error on the untouched test split
- `heldout_model_predictions.csv`: held-out predictions for inspection
- `sampling_summary.csv`: accuracy and sample budget by model, policy, and condition
- `sampling_runs.csv`: individual matched trial records
- `experiment_metadata.json`: settings and evaluation details

These files are generated locally and are intentionally excluded from Git by `.gitignore`. If reviewing a submitted ZIP, use the included copies if present or run the script above to recreate them.

The current evaluation uses 1,500 held-out test cases and 100 fresh simulated events in each of four conditions. Random sampling and information-gain sampling are given one extra measurement each. Static sampling gets none.

### Selected results from the final evaluation

| Method | Held-out exact-source accuracy |
|---|---:|
| Strongest-sensor baseline | 86.20% |
| Logistic Regression (Original) | 92.00% |
| Logistic Regression (Robust) | 92.47% |
| Random Forest (Original) | 92.93% |
| Random Forest (Robust) | 93.00% |
| Graph-Aware Bayesian Inference | 91.47% |

With missing readings, robust Logistic Regression reached 73% accuracy without another sample, 94% with a random sample, and 100% with an information-gain sample in that 100-trial condition. In combined stress, the graph-aware model reached 57% with information-gain sampling.

These are results from a controlled simulation, not a real-world detection system. The additional measurement uses a fixed, cleaner noise setting (standard deviation 0.04). A better result than static sampling does not, by itself, establish that choosing a location intelligently is consistently better than choosing one at random. The comparisons and limitations are discussed in [the final results document](Docs/APS_Final_Results_2026-11-23.md).

## Other ways to check the code

Run a single baseline simulation and inspect the printed source, readings, prediction, and graph error:

```bash
python Scripts/run_pipeline.py
```

Run the full test suite again:

```bash
python -m pytest -q
```

If something does not start:

- **"No APS training dataset found":** Check that the CSV files in `Justin/Data/` or `Team/Justin/Data/` were included.
- **"No module named streamlit" or another missing package:** Activate the virtual environment and rerun `python -m pip install -r requirements.txt`.
- **Repeated-trial results are unavailable in the dashboard:** Run `python Scripts/prepare_demo_results.py` from the project root and reload the page.
- **The browser does not open automatically:** Use the local URL printed by Streamlit.

## Project files and further reading

| Location | Purpose |
|---|---|
| `Simulation/` | Facility, source signals, and sensor behavior |
| `Localization/` | Baseline, supervised ML, and Bayesian source estimation |
| `Adaptive/` | Information-gain sampling |
| `OB/` | Streamlit dashboard and sampling experiments |
| `Evaluation/` | Evaluation metrics |
| `Scripts/` | Simulation and experiment entry points |
| `Tests/` | Automated tests |
| `Justin/Data/` | Committed synthetic training data |
| `Docs/` | Methods, results, and presentation notes |

For dashboard controls, see [the dashboard guide](OB/LIVE_DASHBOARD_README.md). For model assumptions, see [the methods document](Docs/APS_Robust_Localization_Methods.md). The older [benchmark findings](Docs/APS_Benchmark_Findings.md) describe a previous graph model that used true event parameters; those older graph-model figures should not be confused with the current blind Bayesian evaluation.

APS was built for CSC 4240 to explore how existing AI techniques can help with a source-localization problem. Its scope is limited to reproducible spatial simulations; a practical monitoring system would require real sensors, validated data, and a much more detailed physical model.
