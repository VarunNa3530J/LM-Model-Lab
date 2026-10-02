"""Tests for data loading, schema validation, and data quality reporting."""

import numpy as np
import pandas as pd
import pytest

from mlmodellab.config import FEATURE_NAMES, TARGET_NAME
from mlmodellab.data import validate_housing_data, load_housing_data, DatasetDownloadError


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Create synthetic housing DataFrame for offline testing."""
    rng = np.random.default_rng(42)
    n_samples = 50
    data = {
        "MedInc": rng.uniform(0.5, 15.0, n_samples),
        "HouseAge": rng.uniform(1.0, 52.0, n_samples),
        "AveRooms": rng.uniform(1.0, 10.0, n_samples),
        "AveBedrms": rng.uniform(0.5, 4.0, n_samples),
        "Population": rng.uniform(50.0, 5000.0, n_samples),
        "AveOccup": rng.uniform(1.0, 5.0, n_samples),
        "Latitude": rng.uniform(32.0, 42.0, n_samples),
        "Longitude": rng.uniform(-124.0, -114.0, n_samples),
        "MedHouseVal": rng.uniform(0.15, 5.0, n_samples),
    }
    return pd.DataFrame(data)


def test_validate_housing_data_valid(synthetic_df: pd.DataFrame):
    """Test data quality report on valid synthetic dataframe."""
    report = validate_housing_data(synthetic_df)
    assert report["n_rows"] == 50
    assert report["n_cols"] == 9
    assert report["duplicate_rows"] == 0
    assert report["total_missing"] == 0
    assert TARGET_NAME in report["feature_ranges"]
    for feat in FEATURE_NAMES:
        assert feat in report["feature_ranges"]


def test_validate_housing_data_missing_column(synthetic_df: pd.DataFrame):
    """Test validation raises ValueError when required column is missing."""
    bad_df = synthetic_df.drop(columns=["MedInc"])
    with pytest.raises(ValueError, match="DataFrame is missing required columns"):
        validate_housing_data(bad_df)


def test_validate_housing_data_detects_nans(synthetic_df: pd.DataFrame):
    """Test validation accurately reports missing values."""
    synthetic_df.loc[0, "MedInc"] = np.nan
    synthetic_df.loc[1, "HouseAge"] = np.nan
    report = validate_housing_data(synthetic_df)
    assert report["total_missing"] == 2
    assert report["missing_per_column"]["MedInc"] == 1
    assert report["missing_per_column"]["HouseAge"] == 1


def test_validate_housing_data_detects_duplicates(synthetic_df: pd.DataFrame):
    """Test validation detects duplicate rows."""
    dup_df = pd.concat([synthetic_df, synthetic_df.iloc[[0]]], ignore_index=True)
    report = validate_housing_data(dup_df)
    assert report["duplicate_rows"] == 1


def test_load_housing_data_failure_handling(monkeypatch):
    """Test that download failure raises clear DatasetDownloadError."""
    def mock_fetch(*args, **kwargs):
        raise ConnectionError("No network connection")

    monkeypatch.setattr("mlmodellab.data.fetch_california_housing", mock_fetch)
    with pytest.raises(DatasetDownloadError, match="Failed to fetch California Housing dataset"):
        load_housing_data()
