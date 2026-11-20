"""Training-only perturbations for sparse/noisy APS source localization.

This module *never* augments validation or test observations, and it does not
read source labels to decide where to place or mask sensors. It reuses each
training label for altered views of that *same training snapshot*.

The augmentation is not a replacement for new independently simulated cases:
a masked or more-noisy view of a snapshot remains statistically related to its
original. The held-out test partition must stay completely untouched.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ZONES = tuple(f"zone_{i}" for i in range(1, 9))
FEATURE_COLUMNS = tuple(
    field for zone in ZONES
    for field in (f"{zone}_reading", f"{zone}_missing")
)


def augment_sparse_training(
    features: pd.DataFrame,
    labels: pd.Series,
    *,
    copies: int = 3,
    seed: int = 4240,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return original training examples plus independent, reproducible masks.

    For every augmented view, randomly select 2..7 potentially observable
    zones, then retain only readings that were actually available in the
    original training example. Adds mild or severe measurement perturbation
    to a subset of the retained readings, clipping to valid sensor bounds.

    This reflects the dashboard's initial sampling budget and stressed noise.
    The missingness flags are *recomputed* and always agree with NaN values.
    """
    if copies < 0 or copies > 10:
        raise ValueError("copies must be between zero and ten")
    if len(features) != len(labels):
        raise ValueError("Features and source labels must have equal row counts")
    if not set(FEATURE_COLUMNS).issubset(features.columns):
        raise ValueError("Missing APS training features")

    rng = np.random.default_rng(seed)
    original = features.loc[:, FEATURE_COLUMNS].reset_index(drop=True).copy()
    truth = labels.astype(str).reset_index(drop=True)
    augmented_x = [original]
    augmented_y = [truth]
    readings = original.loc[:, [f"{z}_reading" for z in ZONES]].to_numpy(dtype=float)
    count = len(readings)

    for _ in range(copies):
        values = readings.copy()
        for row in range(count):
            budget = int(rng.integers(2, 8))  # Same 2..7 range as the UI
            selected = rng.choice(8, size=budget, replace=False)
            keep = np.zeros(8, dtype=bool)
            keep[selected] = True
            # Never 'restore' a genuinely missing original measurement.
            values[row, ~keep] = np.nan
            observed = np.isfinite(values[row])
            if np.any(observed):
                extra_noise = float(rng.choice([0.0, 0.04, 0.12, 0.24],
                                               p=[0.35, 0.20, 0.20, 0.25]))
                if extra_noise > 0:
                    values[row, observed] = np.clip(
                        values[row, observed] + rng.normal(
                            0.0, extra_noise, size=int(observed.sum())
                        ), 0.0, 1.0
                    )

        frame = pd.DataFrame(index=range(count))
        for index, zone in enumerate(ZONES):
            frame[f"{zone}_reading"] = values[:, index]
            frame[f"{zone}_missing"] = np.isnan(values[:, index]).astype(int)
        augmented_x.append(frame.loc[:, FEATURE_COLUMNS])
        augmented_y.append(truth)

    return (
        pd.concat(augmented_x, ignore_index=True),
        pd.concat(augmented_y, ignore_index=True),
    )
