"""Tests for model pipelines, train/test split, and data leakage prevention."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from mlmodellab.config import FEATURE_NAMES, SEED, TEST_SIZE
from mlmodellab.models import get_all_model_pipelines, get_model_pipeline


@pytest.fixture
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic regression data for offline testing."""
    rng = np.random.default_rng(SEED)
    n_samples = 100
    data = {
        "MedInc": rng.uniform(0.5, 15.0, n_samples),
        "HouseAge": rng.uniform(1.0, 52.0, n_samples),
        "AveRooms": rng.uniform(1.0, 10.0, n_samples),
        "AveBedrms": rng.uniform(0.5, 4.0, n_samples),
        "Population": rng.uniform(50.0, 5000.0, n_samples),
        "AveOccup": rng.uniform(1.0, 5.0, n_samples),
        "Latitude": rng.uniform(32.0, 42.0, n_samples),
        "Longitude": rng.uniform(-124.0, -114.0, n_samples),
    }
    X = pd.DataFrame(data)
    y = pd.Series(rng.uniform(0.15, 5.0, n_samples), name="MedHouseVal")
    return X, y


def test_split_reproducibility(synthetic_data):
    """Test that train_test_split produces identical splits with the fixed seed."""
    X, y = synthetic_data
    X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)

    # Ensure train and test index do not overlap
    assert len(set(X_train_1.index).intersection(set(X_test_1.index))) == 0


def test_models_instantiation():
    """Verify all 4 models are created correctly as Pipelines."""
    models = get_all_model_pipelines()
    expected_names = ["Baseline", "Linear Regression", "Random Forest", "Gradient Boosting"]
    for name in expected_names:
        assert name in models
        pipeline = models[name]
        assert "preprocessor" in pipeline.named_steps
        assert "regressor" in pipeline.named_steps

    with pytest.raises(ValueError, match="Unknown model name"):
        get_model_pipeline("NonExistentModel")


def test_no_data_leakage(synthetic_data):
    """Verify preprocessing is strictly fit on train data.

    Adding extreme outliers to the test set must not affect the scaler means/variances
    computed on the training set.
    """
    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    pipeline = get_model_pipeline("Linear Regression")
    pipeline.fit(X_train, y_train)

    scaler = pipeline.named_steps["preprocessor"].named_steps["scaler"]
    train_mean = np.copy(scaler.mean_)
    train_var = np.copy(scaler.var_)

    # Create an altered test set with extreme numbers
    X_test_extreme = X_test.copy()
    X_test_extreme["MedInc"] = 99999.0

    # Predictions on test set
    _ = pipeline.predict(X_test)
    _ = pipeline.predict(X_test_extreme)

    # Scaler internal statistics must be unchanged
    np.testing.assert_array_equal(scaler.mean_, train_mean)
    np.testing.assert_array_equal(scaler.var_, train_var)
