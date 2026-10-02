"""Tests for single row prediction and input validation."""

import numpy as np
import pandas as pd
import pytest

from mlmodellab.models import get_model_pipeline
from mlmodellab.predict import predict_one, validate_input_row, InputValidationError


@pytest.fixture
def mock_pipeline_and_metadata():
    """Create a fitted pipeline and metadata dictionary for isolated testing."""
    X = pd.DataFrame({
        "MedInc": [2.0, 5.0, 8.0],
        "HouseAge": [15.0, 30.0, 45.0],
        "AveRooms": [4.0, 6.0, 8.0],
        "AveBedrms": [1.0, 1.2, 1.5],
        "Population": [500.0, 1200.0, 2000.0],
        "AveOccup": [2.5, 3.0, 3.5],
        "Latitude": [34.0, 36.0, 38.0],
        "Longitude": [-120.0, -118.0, -116.0],
    })
    y = pd.Series([1.5, 2.5, 3.5])
    pipeline = get_model_pipeline("Linear Regression")
    pipeline.fit(X, y)

    metadata = {
        "model_name": "Linear Regression",
        "feature_names": list(X.columns),
        "feature_ranges": {
            feat: {"min": float(X[feat].min()), "max": float(X[feat].max())}
            for feat in X.columns
        },
    }
    return pipeline, metadata


def test_predict_one_valid(mock_pipeline_and_metadata):
    """Test valid prediction returns correct structure and price."""
    pipeline, metadata = mock_pipeline_and_metadata
    sample_input = {
        "MedInc": 5.0,
        "HouseAge": 30.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.2,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 36.0,
        "Longitude": -118.0,
    }

    result = predict_one(sample_input, pipeline=pipeline, metadata=metadata)
    assert "raw_prediction" in result
    assert "price_usd" in result
    assert isinstance(result["price_usd"], float)
    assert result["price_usd"] == pytest.approx(result["raw_prediction"] * 100000.0)
    assert len(result["warnings"]) == 0


def test_predict_one_out_of_range_warning(mock_pipeline_and_metadata):
    """Test that input outside training range produces warning but succeeds."""
    pipeline, metadata = mock_pipeline_and_metadata
    sample_input = {
        "MedInc": 25.0,  # Max in fixture is 8.0
        "HouseAge": 30.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.2,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 36.0,
        "Longitude": -118.0,
    }

    result = predict_one(sample_input, pipeline=pipeline, metadata=metadata)
    assert len(result["warnings"]) == 1
    assert "above training maximum" in result["warnings"][0]


def test_validate_input_missing_feature(mock_pipeline_and_metadata):
    """Test validation catches missing features."""
    _, metadata = mock_pipeline_and_metadata
    bad_input = {"MedInc": 3.0}
    with pytest.raises(InputValidationError, match="Missing required feature"):
        validate_input_row(bad_input, metadata)


def test_validate_input_nan_or_non_numeric(mock_pipeline_and_metadata):
    """Test validation catches NaN and string non-numeric values."""
    _, metadata = mock_pipeline_and_metadata
    full_input = {
        "MedInc": np.nan,
        "HouseAge": 30.0,
        "AveRooms": 6.0,
        "AveBedrms": 1.2,
        "Population": 1200.0,
        "AveOccup": 3.0,
        "Latitude": 36.0,
        "Longitude": -118.0,
    }
    with pytest.raises(InputValidationError, match="cannot be null or NaN"):
        validate_input_row(full_input, metadata)

    full_input["MedInc"] = "abc"
    with pytest.raises(InputValidationError, match="must be numeric"):
        validate_input_row(full_input, metadata)
