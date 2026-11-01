# Simulator Assumptions and Validation

**Author:** Emmanuel Zuniga  
**Project:** Adaptive Pathogen Surveillance (APS)  
**Course:** CSC 4240/5240 Artificial Intelligence

## Purpose

This document explains the assumptions used by the APS simulator and proposes
ways to validate its behavior. The simulator creates controlled sensor data so
the team can evaluate methods for locating the source of a simulated outbreak.

APS is a class project simulation. It is not a model of real pathogen
transmission and is not intended for clinical or real-world surveillance use.

## Simulator Assumptions

### Facility

The simulated facility is a fixed, connected graph containing eight zones.
Connections between zones are treated as undirected. The simulator does not
currently account for room size, airflow, walls, doors, or other physical
features.

### Outbreak source and signal

One zone is selected as the hidden outbreak source. The ideal signal in each
zone depends on its shortest-path distance from that source:

`signal = source_intensity * exp(-decay_rate * distance)`

This means the ideal signal is strongest at the source and decreases with
distance. This is a simplified spatial rule, not a biological transmission
model.

### Sensor readings

The simulator adds random noise to readings and can mark readings as missing.
Readings are clipped to the configured minimum and maximum values. The current
default settings include a source intensity of `1.0`, a decay rate of `0.65`,
a noise standard deviation of `0.08`, and a missing-data rate of `0.10`.

### Current localization baseline

The baseline predicts that the source is the zone with the strongest available
sensor reading. It ignores missing readings. If every reading is missing, it
cannot make a prediction.

## Validation

### Checks already covered by project tests

The existing tests check that:

- the facility contains eight connected zones;
- selecting a source with the same random seed is reproducible;
- the ideal signal is strongest at the source;
- signal strength decreases as graph distance increases; and
- zero noise and zero missingness preserve the ideal readings.

Tests also check baseline behavior with missing readings and the source
accuracy and graph-distance metrics.

### Additional validation proposed

1. Use every zone as the source and confirm that its ideal signal is strongest.
2. Run simulations with different noise and missing-data settings to see how
   sensor readings and localization performance change.
3. Run many simulations with different random seeds and report exact-source
   accuracy and average graph distance from the true source.
4. Repeat simulations with the same seed and settings to check that results are
   reproducible.
5. Compare the baseline with future machine-learning and adaptive-sampling
   methods using the same simulated cases.

These are proposed checks; results should be added after the experiments are
run.

## Limitations

The simulator uses a small, fixed facility and simplified signal decay. Its
results can help compare methods under controlled conditions, but they do not
show how real pathogens or physical sensors behave.