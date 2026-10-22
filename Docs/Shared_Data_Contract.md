# APS Shared Data Contract

## Adaptive Pathogen Surveillance â€” CSC 4240 Simulation Project

### Purpose

This document defines shared data exchanged by the approved course components of APS. The project is a controlled simulation for source localization and adaptive sampling, not a clinical or biological-fidelity pathogen model.

## Course Data Flow

```text
Simulation -> Sensor Readings -> Localization -> Adaptive Sampling -> Evaluation
```

## Simulation Run Record

```json
{
  "run_id": "run_001",
  "sensor_readings": {
    "zone_1": 0.18,
    "zone_2": 0.41,
    "zone_3": null,
    "zone_4": 0.77
  }
}
```

## Hidden Ground Truth

```json
{
  "run_id": "run_001",
  "true_source_node": "zone_4"
}
```

`true_source_node` must never be model input. It is reserved for evaluation to prevent target leakage.

## Localization Output

```json
{
  "run_id": "run_001",
  "predicted_source_node": "zone_4",
  "confidence": 0.73,
  "method": "random_forest"
}
```

Confidence is planned for the second-half ML models. The current baseline returns only a predicted source.

## Adaptive Sampling Output

Planned second-half format:

```json
{
  "run_id": "run_001",
  "additional_sample_node": "zone_6",
  "reason": "low_prediction_confidence"
}
```

Only one additional sample is allowed in the approved course scope.

## Evaluation

Primary measures:
- exact source-location accuracy
- graph-distance error
- performance under increasing noise
- performance under increasing missing data
- improvement after one adaptive sample

## Deferred Work

Genomics, mutation analysis, automated response recommendations, physical sensors, wet-lab validation, clinical diagnosis, and production deployment are future work only.
