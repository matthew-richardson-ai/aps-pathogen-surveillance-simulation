"""Regression tests for the two historical APS dataset locations."""
from pathlib import Path

import pytest

import Localization.ml_pipeline as pipeline


def use_project_root(monkeypatch, root):
    """Redirect module-relative project root to an isolated test folder."""
    monkeypatch.setattr(pipeline, "__file__", str(root / "Localization" / "ml_pipeline.py"))


def test_finds_legacy_justin_dataset(tmp_path, monkeypatch):
    use_project_root(monkeypatch, tmp_path)
    folder = tmp_path / "Justin" / "Data"
    folder.mkdir(parents=True)
    older = folder / "outbreak_dataset_20261031_001.csv"
    newer = folder / "outbreak_dataset_20261101_001.csv"
    older.write_text("dummy", encoding="utf-8")
    newer.write_text("dummy", encoding="utf-8")

    assert pipeline.default_dataset_path() == newer


def test_prefers_team_folder_when_both_exist(tmp_path, monkeypatch):
    use_project_root(monkeypatch, tmp_path)
    for subpath in ("Justin/Data", "Team/Justin/Data"):
        folder = tmp_path / subpath
        folder.mkdir(parents=True)
        (folder / "outbreak_dataset_20261101_001.csv").write_text(
            "dummy", encoding="utf-8"
        )

    assert pipeline.default_dataset_path() == (
        tmp_path / "Team" / "Justin" / "Data" / "outbreak_dataset_20261101_001.csv"
    )


def test_missing_datasets_reports_both_locations(tmp_path, monkeypatch):
    use_project_root(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError, match="Justin"):
        pipeline.default_dataset_path()
