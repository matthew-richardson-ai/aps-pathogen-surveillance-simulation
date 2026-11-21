# Simulation and Sensor Validation

**Author:** Emmanuel Zuniga  
**Project:** Adaptive Pathogen Surveillance (APS)  
**Course:** CSC 4240/5240 Artificial Intelligence

## My Role

My responsibility is simulation and environmental modeling. I build on
the existing facility and sensor code, document its assumptions, and
check that it behaves consistently under different sensor conditions.

## Simulation Assumptions

The facility consists of eight connected zones represented as a
NetworkX graph. Connections are undirected, and distance is measured
by the number of connections along the shortest path.

One zone is selected as the hidden source. Ideal signal strength
decreases with graph distance using:

`signal = source_intensity * exp(-decay_rate * distance)`

The demonstrations use a source intensity of 1.0 and a decay rate
of 0.65. Sensor readings can include random noise and missing values.
Available readings are clipped to the range 0.0 to 1.0.

A fixed random seed makes runs repeatable with the same settings
and software environment.

## My Contributions

### Simulation Demonstration

I created `simulation_demo.py` to run four scenarios automatically
using seed 43 and the same source location.

| Scenario | Noise Standard Deviation | Missing Rate |
|---|---:|---:|
| Ideal conditions | 0.00 | 0.00 |
| Noise only | 0.08 | 0.00 |
| Missing readings only | 0.00 | 0.30 |
| Noise and missing readings | 0.08 | 0.30 |

Each scenario displays the source, graph distances, ideal signals,
observed readings, and number of missing sensors.

### Automated Validation

I created `validate_simulation.py` to check all eight possible source
zones. For each source, the script checks that:

- The source signal is 1.0 and strictly stronger than other signals.
- Ideal signals decrease as graph distance increases.
- Zero noise and zero missingness preserve ideal readings.
- Reusing the same seed reproduces noisy and missing readings.
- Available readings remain between 0.0 and 1.0.
- A missing rate of 100% removes every reading.

One additional check confirms that source selection repeats when
the same seed is used.

## Results

The validation script completed successfully:

```text
Results: 49/49 checks passed
All validation checks passed.
```

In the demonstration with seed 43, the selected source was `zone_1`.
Under ideal conditions, all observed readings matched their ideal
values. Adding noise changed sensor readings. With a missing rate
of 30%, three of the eight sensors were missing in this example.
Combining noise and missingness produced both effects.

The missing rate is a probability for each sensor, so an individual
run does not need to have exactly 30% of its readings missing.

These results confirm the checked simulator behavior. They do not
measure source-prediction accuracy.

## How to Run

Run these commands from the project's main folder.

Install the libraries needed by these scripts:

```bash
python -m pip install networkx numpy
```

Run the demonstration:

```bash
python Team/Emmanuel/simulation_demo.py
```

Run the validation checks:

```bash
python Team/Emmanuel/validate_simulation.py
```

Save the outputs:

```bash
python Team/Emmanuel/simulation_demo.py > Team/Emmanuel/simulation_results.txt
python Team/Emmanuel/validate_simulation.py > Team/Emmanuel/validation_results.txt
```

## Data Handling

The demonstration displays the true source for inspection.
The true source must remain separate from prediction inputs
when preparing data for machine-learning models.

## Limitations

This is a controlled classroom simulation. Graph distance does not
represent actual airflow or biological transmission. The model
does not account for room size, ventilation, doors, or movement.

The validation checks cover defined rules and selected sensor
settings. They do not establish real-world accuracy or evaluate
prediction performance across many random seeds.

## Next Steps

- Run additional seeds and sensor settings.
- Coordinate simulation outputs with the dataset developer.
- Support team evaluation of source-location methods under
  different noise and missing-data conditions.