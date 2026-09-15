# APS Project Quick Reference Guide

## Autonomous Pathogen Surveillance — Simulation Project

This guide explains the organization of the APS repository, the purpose of each major component, common project terminology, and basic development conventions.

APS is an **undergraduate artificial intelligence simulation project**. It does not perform real-world pathogen surveillance, clinical diagnosis, or medical decision-making.

---

# 1. Project Concept

APS simulates an environmental pathogen-surveillance system.

The basic idea is:

```text
Simulated Facility
        ↓
Simulated Outbreak
        ↓
Simulated Sensor Observations
        ↓
Genomic Analysis
        ↓
Outbreak Source Localization
        ↓
AI Response Recommendation
        ↓
Evaluation
```

The system is intentionally modular.

Each major part of the project performs one responsibility and passes its results to the next part.

---

# 2. Repository Structure

```text
aps-pathogen-surveillance-simulation/
│
├── Config/
│
├── Data/
│   ├── Raw/
│   ├── Processed/
│   ├── Synthetic/
│   ├── ReferenceGenomes/
│   └── Samples/
│
├── Docs/
│   ├── Architecture/
│   ├── Diagrams/
│   ├── MeetingNotes/
│   └── Research/
│
├── Simulation/
│   ├── facility.py
│   ├── outbreak.py
│   └── sensors.py
│
├── Genomics/
│   ├── sequence_loader.py
│   ├── mutation_detector.py
│   └── risk_scorer.py
│
├── Localization/
│   ├── graph_builder.py
│   ├── source_localizer.py
│   └── baselines.py
│
├── Response/
│   └── response_agent.py
│
├── Evaluation/
│   ├── metrics.py
│   └── experiments.py
│
├── Models/
├── Notebooks/
├── Results/
├── Scripts/
├── Tests/
│
├── .env.example
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

# 3. Major Folder Reference

## `Config/`

Contains settings that control how simulations and experiments run.

Example:

```text
Config/default.yaml
```

Possible settings include:

* number of simulation steps
* sensor sensitivity
* sensor false-positive rate
* number of experiment runs
* random seed
* localization settings

The goal is to change experiment settings here instead of hard-coding values throughout the program.

---

# 4. `Data/`

Contains project datasets and generated data.

## `Data/Raw/`

Original external data before modification.

Examples:

* downloaded public datasets
* original CSV files
* source datasets

Raw data should generally remain unchanged.

Large raw datasets should normally **not be committed to GitHub**.

---

## `Data/Processed/`

Contains data that has been cleaned, transformed, filtered, or prepared for analysis.

Example:

```text
Raw genome metadata
        ↓
remove unused fields
        ↓
normalize names
        ↓
Processed dataset
```

---

## `Data/Synthetic/`

Contains data created by our own simulation.

Examples:

```text
sensor_readings.csv
outbreak_run_001.csv
simulated_mutations.csv
facility_events.csv
```

This will likely become one of the most important data folders.

---

## `Data/ReferenceGenomes/`

Contains locally downloaded reference genome files.

Typical format:

```text
.fasta
.fna
.gb
.gbff
```

Example reference:

```text
SARS-CoV-2
NCBI RefSeq: NC_045512.2
```

Large genome files should generally not be committed directly to GitHub.

Instead, accession numbers and download instructions should be documented.

---

## `Data/Samples/`

Contains small example datasets that are safe to include in GitHub.

These files can help developers understand expected formats.

Examples:

```text
sample_sensor_data.csv
sample_mutations.csv
sample_pathogens.csv
```

---

# 5. `Simulation/`

Creates the simulated world APS operates within.

This is where the project's ground truth originates.

---

## `facility.py`

Defines the simulated facility.

The facility may contain:

```text
Rooms
Hallways
Entrances
HVAC connections
Plumbing connections
Sensor locations
```

The facility may eventually be represented as a graph.

Example:

```text
Room A ─── Hallway ─── Room B
   │                       │
Sensor 1                Sensor 2
```

---

## `outbreak.py`

Controls how the simulated outbreak begins and spreads.

Possible responsibilities:

* choose outbreak origin
* determine initial infected location
* simulate spread between connected locations
* track outbreak over time
* preserve the true outbreak origin for evaluation

Example:

```text
Ground Truth Origin:
Room_203

Time 0:
Room_203 infected

Time 5:
Hallway_B exposed

Time 10:
Room_204 exposed
```

The AI does not receive the hidden ground truth.

That information is used later by `Evaluation/`.

---

## `sensors.py`

Generates simulated sensor observations.

Sensors may include realistic imperfections such as:

* false positives
* false negatives
* delayed detections
* missing readings
* imperfect sensitivity

Example output:

```csv
timestamp,sensor_id,node,pathogen_detected
08:10,S03,Room_203,True
08:24,S02,Hallway_B,True
08:41,S05,Room_204,True
```

---

# 6. `Genomics/`

Handles genomic sequence information and simulated mutations.

---

## `sequence_loader.py`

Loads genomic sequence files.

Most complete genomes will use FASTA rather than CSV.

Example:

```text
reference.fasta
      ↓
sequence_loader.py
      ↓
DNA sequence available to Python
```

Biopython will likely be used here.

---

## `mutation_detector.py`

Compares genomic sequences.

Example:

```text
Reference Genome
       +
Observed Variant
       ↓
Mutation Detector
       ↓
Differences / Mutations
```

Example result:

```text
Position: 23403
Reference: A
Observed: G
Mutation: A → G
```

Because this is a simulation, we can intentionally introduce mutations and therefore know the correct answer.

---

## `risk_scorer.py`

Assigns some form of experimental anomaly or risk score to genomic differences.

For the undergraduate simulation, this should not be interpreted as a clinically validated prediction of disease severity.

Possible output:

```text
LOW
MODERATE
HIGH
```

or:

```text
Risk Score: 0.72
```

---

# 7. `Localization/`

Attempts to determine where the simulated outbreak most likely originated.

This is one of the project's main AI components.

---

## `graph_builder.py`

Transforms the facility into a graph structure.

In graph terminology:

```text
Node = location

Edge = connection between locations
```

Example:

```text
Room A ── Hallway ── Room B
             │
           Stairs
             │
           Room C
```

NetworkX will likely be used here.

---

## `source_localizer.py`

Uses sensor observations, timestamps, and facility connections to estimate the most likely outbreak origin.

Example:

```text
Actual origin:
Room_203

Model predictions:

1. Room_203    0.68
2. Hallway_B   0.17
3. Room_204    0.09
```

The preferred terminology is:

**Outbreak Source**

**Source Node**

**Probable Origin Node**

rather than "Patient Zero."

APS detects a probable location, not a specific person.

---

## `baselines.py`

Contains simple algorithms used for comparison.

A baseline is a simpler method that provides something to compare the proposed AI method against.

Example:

```text
Baseline:
Choose the location closest to the first positive sensor.

AI Method:
Use graph structure + time + multiple sensor observations.
```

If the AI performs better than the baseline, we have evidence that the more sophisticated method adds value.

---

# 8. `Response/`

Contains the final response/recommendation component.

## `response_agent.py`

Receives information from previous stages and generates simulated recommendations.

Possible inputs:

```text
Detected pathogen
Mutation/anomaly information
Probable outbreak origin
Facility structure
Spread estimate
```

Possible output:

```text
Probable origin:
Room 203

Recommended simulated response:
- Inspect Room 203
- Monitor adjacent hallway sensors
- Increase simulated sampling frequency
- Evaluate connected rooms
```

These are simulation outputs only.

---

# 9. `Evaluation/`

Determines how well APS performs.

This section is extremely important because it turns the project from:

> "We built something."

into:

> "We built something and measured whether it works."

---

## `metrics.py`

Contains functions for calculating performance.

Possible metrics include:

### Top-1 Accuracy

Was the model's first choice correct?

```text
Actual: Room_203
Prediction #1: Room_203

Correct
```

---

### Top-3 Accuracy

Was the actual location somewhere in the model's three highest predictions?

---

### Graph Distance Error

How far was the prediction from the actual source?

Example:

```text
Actual:
Room A

Predicted:
Room C

Graph Distance:
2 edges
```

---

### Detection Lead Time

How quickly did the system detect a developing outbreak?

---

### False Positive Performance

How often did APS respond to something that was not actually present?

---

### Missing Sensor Robustness

How well does localization work if some sensors fail?

---

## `experiments.py`

Runs repeated controlled experiments.

Example:

```text
Run 1
Run 2
Run 3
...
Run 100
```

Each experiment may randomly change:

* outbreak origin
* pathogen
* mutation
* sensor failures
* sensor noise
* spread probability

Results can then be aggregated.

---

# 10. `Models/`

Stores trained or downloaded machine-learning model artifacts locally.

Examples:

```text
model.pt
classifier.pkl
model.ckpt
```

These files can become extremely large.

Most model files should therefore **not be committed to GitHub**.

---

# 11. `Notebooks/`

Contains Jupyter notebooks used for:

* experimentation
* visualization
* dataset exploration
* model testing
* research demonstrations

Example:

```text
Notebooks/
└── genome_exploration.ipynb
```

Notebooks should generally be used for exploration.

Finished project logic should eventually move into normal `.py` files.

---

# 12. `Results/`

Contains generated experiment results.

Examples:

```text
accuracy_results.csv
source_localization_results.csv
experiment_summary.csv
```

Charts may also eventually be generated here.

Most automatically generated results should not need to be committed.

---

# 13. `Scripts/`

Contains executable project utilities.

The main script will eventually be:

```text
Scripts/run_pipeline.py
```

Its job will be to connect the project components.

Example:

```text
Simulation
    ↓
Genomics
    ↓
Localization
    ↓
Response
    ↓
Evaluation
```

Eventually, running something similar to:

```bash
python Scripts/run_pipeline.py
```

should execute an APS simulation.

---

# 14. `Tests/`

Contains automated tests.

Example:

```text
test_simulation.py
test_genomics.py
test_localization.py
test_evaluation.py
```

Tests verify that individual components behave as expected.

Example:

```python
def test_sensor_false_positive_rate(): ...
```

As features are added, corresponding tests should also be added.

---

# 15. `Docs/`

Contains project documentation that does not belong in source code.

---

## `Docs/Architecture/`

System architecture and technical-design documents.

---

## `Docs/Diagrams/`

Flowcharts, UML diagrams, architecture diagrams, graphs, and other visuals.

---

## `Docs/MeetingNotes/`

Important team decisions and meeting notes.

Useful things to record include:

```text
What was decided?
Why was it decided?
Who is responsible?
What changed?
```

---

## `Docs/Research/`

Research papers, citations, references, notes, and links relevant to APS.

Examples:

* genomic foundation models
* outbreak source localization
* wastewater surveillance
* graph neural networks
* environmental pathogen sensing

Do not upload copyrighted research papers unless redistribution is permitted.

Links, citations, and research notes are usually preferable.

---

# 16. Important Root Files

## `README.md`

Main introduction to the project.

Someone visiting the GitHub repository should be able to understand the project from this file.

---

## `requirements.txt`

Lists Python packages required by APS.

Example:

```text
numpy
pandas
networkx
biopython
scikit-learn
matplotlib
pytest
```

Install with:

```bash
pip install -r requirements.txt
```

---

## `.gitignore`

Tells Git which files should not be uploaded.

Examples:

```text
Virtual environments
Large datasets
Reference genomes
Model weights
Generated results
Secret API keys
Temporary files
```

---

## `.env.example`

Shows which environment variables the project expects without exposing actual secrets.

Never commit a real `.env` file containing API keys.

---

## `CONTRIBUTING.md`

Documents team development practices.

This can eventually include:

* branch naming
* commit conventions
* pull-request process
* coding style
* testing expectations

---

# 17. Project Data Flow

The intended architecture is:

```text
                ┌─────────────────────┐
                │      Config/        │
                └─────────┬───────────┘
                          │
                          ▼
                ┌─────────────────────┐
                │    Simulation/      │
                │                     │
                │ facility.py         │
                │ outbreak.py         │
                │ sensors.py          │
                └─────────┬───────────┘
                          │
                          ▼
                    Sensor Data
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   ┌────────────────┐          ┌────────────────┐
   │   Genomics/    │          │ Localization/  │
   └───────┬────────┘          └───────┬────────┘
           │                           │
           ▼                           ▼
    Genomic Analysis             Probable Origin
           │                           │
           └─────────────┬─────────────┘
                         ▼
                ┌────────────────┐
                │   Response/    │
                └───────┬────────┘
                        │
                        ▼
                 Recommendations
                        │
                        ▼
                ┌────────────────┐
                │  Evaluation/   │
                └────────────────┘
```

---

# 18. Glossary

## APS

**Autonomous Pathogen Surveillance**

The name of the simulated AI system.

---

## FASTA

A common text format for storing biological sequences.

Example:

```text
>sequence_name
ATTAAAGGTTTATACCTTCCCAGGTA...
```

---

## Reference Genome

A known genome sequence used as a comparison point.

Example:

```text
SARS-CoV-2 Wuhan-Hu-1
NCBI RefSeq NC_045512.2
```

---

## Accession Number

A unique identifier assigned to a biological sequence in a database.

Example:

```text
NC_045512.2
```

---

## Mutation

A difference between a reference genetic sequence and another sequence.

Example:

```text
Reference: A
Variant:   G

A → G
```

---

## SNP / SNV

A difference involving a single nucleotide.

Example:

```text
A → G
```

---

## Synthetic Data

Artificially generated data designed to imitate characteristics of real data.

APS primarily uses synthetic data.

---

## Ground Truth

The correct answer known by the simulator.

Example:

```text
True source:
Room_203
```

The AI does not receive this answer.

It is used afterward for evaluation.

---

## Node

A location or object in a graph.

Example:

```text
Room_203
```

---

## Edge

A connection between two nodes.

Example:

```text
Room_203 ─── Hallway_B
```

---

## Graph

A mathematical structure consisting of nodes and edges.

APS uses graphs to represent the simulated facility.

---

## Source Localization

The task of estimating where an outbreak began based on available observations.

---

## Source Node

The simulated location where an outbreak originated.

Preferred over the term "Patient Zero."

---

## Baseline

A simpler method used for comparison against a more advanced technique.

---

## Model

An algorithm that learns patterns from data or uses learned parameters to make predictions.

---

## Training Data

Data used to teach a machine-learning model.

---

## Test Data

Data kept separate from training and used to measure how well a model performs.

---

## Inference

Using a trained model to generate a prediction.

---

## Sensor Sensitivity

The probability that a simulated sensor detects something that is actually present.

---

## False Positive

The sensor reports a pathogen when the simulated pathogen is not present.

---

## False Negative

The sensor fails to report a pathogen that actually is present.

---

## Pipeline

A sequence of processing stages where the output of one stage becomes input to another.

APS is designed as a pipeline.

---

## MVP

**Minimum Viable Product**

The smallest complete version of APS that demonstrates the project's major concept.

The MVP does not need to contain every possible future capability.

---

# 19. Naming Conventions

Top-level organizational directories use capital letters:

```text
Simulation/
Genomics/
Localization/
Evaluation/
```

Python files remain lowercase:

```text
source_localizer.py
mutation_detector.py
graph_builder.py
```

Python variables and functions should use `snake_case`:

```python
sensor_readings
source_node
detect_mutations()
calculate_accuracy()
```

Python classes should use `PascalCase`:

```python
FacilityGraph
SensorReading
OutbreakSimulation
```

Constants should generally use uppercase:

```python
DEFAULT_SEED = 42
MAX_SIMULATION_STEPS = 100
```

---

# 20. Where Does My File Go?

If you are unsure where something belongs:

| Item                          | Location                    |
| ----------------------------- | --------------------------- |
| Facility simulation code      | `Simulation/`               |
| Outbreak spread code          | `Simulation/`               |
| Sensor simulation             | `Simulation/`               |
| Complete genome FASTA         | `Data/ReferenceGenomes/`    |
| Small example CSV             | `Data/Samples/`             |
| Generated simulation CSV      | `Data/Synthetic/`           |
| Genome loading code           | `Genomics/`                 |
| Mutation detection            | `Genomics/`                 |
| Facility graph code           | `Localization/`             |
| Source prediction model       | `Localization/`             |
| Simple comparison algorithm   | `Localization/baselines.py` |
| Response recommendation logic | `Response/`                 |
| Accuracy calculations         | `Evaluation/`               |
| Repeated experiments          | `Evaluation/`               |
| Jupyter experiments           | `Notebooks/`                |
| Research notes                | `Docs/Research/`            |
| Architecture diagram          | `Docs/Diagrams/`            |
| Team meeting notes            | `Docs/MeetingNotes/`        |
| Unit tests                    | `Tests/`                    |

---

# 21. Git Workflow

Before beginning work:

```bash
git pull
```

Check repository status:

```bash
git status
```

After completing a meaningful change:

```bash
git add .
```

Then commit:

```bash
git commit -m "feat: add facility graph model"
```

Push:

```bash
git push
```

Avoid commits such as:

```text
stuff
changes
update
fixed things
final
final2
```

Prefer descriptive commits:

```text
feat: add synthetic sensor generator

feat: implement FASTA sequence loader

fix: correct sensor timestamp ordering

test: add mutation detector tests

docs: update architecture guide
```

---

# 22. Common Commit Prefixes

## `feat:`

New functionality.

```text
feat: add outbreak simulator
```

## `fix:`

Bug fix.

```text
fix: correct graph edge weights
```

## `docs:`

Documentation only.

```text
docs: add genomic dataset notes
```

## `test:`

Testing changes.

```text
test: add localization unit tests
```

## `refactor:`

Code restructuring without changing intended behavior.

```text
refactor: reorganize sensor processing
```

## `chore:`

Repository or development maintenance.

```text
chore: update project dependencies
```

---

# 23. Important Team Rule

Avoid writing two different components that secretly expect different data formats.

Before connecting modules, agree on the exact structure of the data being passed.

For example:

```text
Simulation/
      ↓
sensor_readings.csv
      ↓
Localization/
```

Everyone should agree on fields such as:

```csv
timestamp,sensor_id,node,pathogen_id,detected
```

rather than one person expecting:

```text
room
time
positive
```

while another produces:

```text
location
timestamp
detection
```

Clear interfaces will save a significant amount of integration work later.

---

# 24. Current Scope

APS is currently intended to investigate:

1. Simulated environmental pathogen observations.
2. Genomic sequence and mutation analysis.
3. Facility graph modeling.
4. Outbreak-source localization.
5. AI-assisted simulated response recommendations.
6. Quantitative evaluation against known simulation ground truth.

APS is **not** currently intended to:

* build a physical biosensor
* diagnose patients
* identify individual infected people
* provide real-world medical recommendations
* provide clinically validated virulence predictions
* operate as an actual public-health surveillance platform

---

# 25. The Main Question to Remember

Every component should ultimately contribute to answering:

> Given simulated pathogen observations across a facility, can an AI-based system analyze the available evidence, estimate the probable outbreak origin, and generate an appropriate simulated response?

If a proposed feature does not help answer that question or evaluate the answer, it may be outside the current MVP scope.
