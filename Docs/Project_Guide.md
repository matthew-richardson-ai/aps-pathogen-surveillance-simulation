# APS Project Quick Reference Guide

## Adaptive Pathogen Surveillance â€” Simulation Project

APS studies source localization from imperfect spatial sensor observations in a small simulated facility.

## Halfway Architecture

```text
Simulation/
  facility.py   -> 8-zone NetworkX graph
  outbreak.py   -> hidden source + spatial signal decay
  sensors.py    -> noise + missingness

Localization/
  baselines.py         -> strongest-sensor baseline
  source_localizer.py  -> ML planned for second half

Adaptive/
  confidence.py -> planned
  sampler.py    -> planned

Evaluation/
  metrics.py      -> exact match + graph distance
  experiments.py  -> planned experiment sweeps
```

## Halfway Checkpoint Goal

The checkpoint demonstrates that APS can construct a facility, select a hidden source, generate spatially consistent signals, corrupt observations with controlled imperfections, produce a baseline prediction, and measure its error.

This is intentionally not the completed project. ML localization, adaptive sampling, full sensitivity analysis, and the final interface remain future milestones.

## Simulator Validation Strategy

- source zone has the strongest expected signal when noise is zero
- expected signal decreases as graph distance increases
- fixed random seeds reproduce the same run
- increasing noise should make observations less reliable
- increasing missingness should remove information
- moving the hidden source should move the signal pattern
