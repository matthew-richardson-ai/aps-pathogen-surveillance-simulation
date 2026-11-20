"""Train-only augmentation invariants for APS missing-data robustness."""
import numpy as np
import pandas as pd
import pytest

from Localization.robust_training import augment_sparse_training, ZONES, FEATURE_COLUMNS


def features_and_labels():
    rows = []
    for i in range(24):
        record = {}
        for k, zone in enumerate(ZONES):
            value = None if k == i % 8 else float((k + 1) / 10)
            record[f"{zone}_reading"] = np.nan if value is None else value
            record[f"{zone}_missing"] = int(value is None)
        rows.append(record)
    return (pd.DataFrame(rows, columns=FEATURE_COLUMNS),
            pd.Series([ZONES[i % 8] for i in range(24)]))


def test_reproducible_augmented_training_views_preserve_labels_and_schema():
    X, y = features_and_labels()
    output, labels = augment_sparse_training(X, y, copies=3, seed=42)
    again, second_labels = augment_sparse_training(X, y, copies=3, seed=42)
    assert len(output) == 4 * len(X)
    assert list(output.columns) == list(FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(output, again)
    pd.testing.assert_series_equal(labels, second_labels)
    pd.testing.assert_frame_equal(output.iloc[:len(X)].reset_index(drop=True), X)
    assert labels.tolist() == 4 * y.tolist()


def test_augmentation_never_recovers_missing_data_or_leaks_masked_measurements():
    X, y = features_and_labels()
    output, _ = augment_sparse_training(X, y, copies=2, seed=7)
    for repeat in range(3):
        data = output.iloc[repeat * len(X):(repeat + 1) * len(X)].reset_index(drop=True)
        for zone in ZONES:
            original_missing = X[f"{zone}_reading"].isna()
            assert data.loc[original_missing, f"{zone}_reading"].isna().all()
            assert (
                data[f"{zone}_reading"].isna().astype(int).to_numpy()
                == data[f"{zone}_missing"].to_numpy()
            ).all()
            present = data[f"{zone}_reading"].dropna()
            assert present.between(0.0, 1.0).all()


def test_invalid_augmentation_arguments_are_rejected():
    X, y = features_and_labels()
    with pytest.raises(ValueError, match="copies"):
        augment_sparse_training(X, y, copies=-1)
    with pytest.raises(ValueError, match="row counts"):
        augment_sparse_training(X, y.iloc[:-1])
