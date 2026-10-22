# Adaptive Pathogen Surveillance (APS)

## Pathogen Surveillance Simulation

APS is an undergraduate artificial intelligence project investigating whether AI can identify the likely source of a simulated outbreak from imperfect environmental sensor data, and whether targeted additional sampling can improve that prediction.

The class project is intentionally simulation-only. It does not perform clinical diagnosis, wet-lab testing, physical biosensing, or production pathogen surveillance.

## Approved Course Pipeline

```text
Simulated Facility
        |
        v
Hidden Outbreak Source
        |
        v
Spatial Signal Propagation
        |
        v
Noisy / Missing Sensor Readings
        |
        v
Source Localization
        |
        v
Adaptive Additional Sampling
        |
        v
Quantitative Evaluation
```

## Halfway Checkpoint

Implemented:
- fixed 8-zone NetworkX facility graph
- hidden outbreak-source selection
- exponential signal decay by graph distance
- configurable sensor noise and missing data
- strongest-sensor baseline localization
- exact-source and graph-distance metrics
- automated tests for core simulator behavior
- reproducible random seeds

Planned for the second half:
- dataset generation for repeated experiments
- Logistic Regression localization
- Random Forest localization
- prediction confidence
- one-step adaptive sampling
- noise/missing-data sensitivity experiments
- comparison of static versus adaptive sampling
- final visualization/demo interface

## Scope Management

Earlier APS concepts included genomic analysis and automated response recommendations. Those modules have been moved under `Future/` and are explicitly outside the CSC 4240 class implementation. They remain only as possible post-course extensions.
