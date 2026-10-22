# APS Halfway Progress Report

## Current Status

The project foundation and first functional simulation checkpoint are complete. APS now contains a connected eight-zone facility model, hidden outbreak-source generation, distance-based signal propagation, simulated sensor noise and missing data, a strongest-sensor localization baseline, initial evaluation metrics, and automated tests for expected simulator behavior.

## Scope Control

To address concerns about project size, genomic analysis and automated response recommendations were moved to future work. The CSC 4240 implementation is limited to simulation, source localization, one-step adaptive sampling, and quantitative evaluation.

## Simulator Credibility

The simulator is treated as a controlled abstraction rather than a clinical pathogen-transmission model. Expected-behavior tests confirm that the source produces the strongest ideal signal, signals decrease with graph distance, and seeded runs are reproducible. The next evaluation phase will vary noise, missing-data rates, source locations, and model settings to determine whether conclusions are robust rather than dependent on one parameter configuration.

## Work Remaining

The second half will implement Logistic Regression and Random Forest localization, prediction confidence, one-step adaptive sampling, repeated experiment sweeps, comparison against the baseline, and a final visualization/demo interface.
