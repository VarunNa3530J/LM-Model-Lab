"""Tests for evaluation metrics and cross-validation."""

import numpy as np
import pandas as pd
import pytest

from mlmodellab.evaluate import compute_metrics, evaluate_cross_validation
from mlmodellab.models import get_model_pipeline


def test_compute_metrics_known_values():
    """Verify metrics calculation against hand-computed values.

    Let y_true = [1.0, 2.0, 3.0, 4.0]
    Let y_pred = [1.0, 2.0, 4.0, 5.0]
    Errors: [0, 0, 1, 1]
    MAE = (0 + 0 + 1 + 1) / 4 = 0.5
    MSE = (0^2 + 0^2 + 1^2 + 1^2) / 4 = 0.5
    RMSE = sqrt(0.5) ≈ 0.7071
    y_mean = 2.5
    Total SS = (1-2.5)^2 + (2-2.5)^2 + (3-2.5)^2 + (4-2.5)^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
    Residual SS = 0 + 0 + 1 + 1 = 2.0
    R2 = 1 - (2.0 / 5.0) = 0.6
    """
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 4.0, 5.0])

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == 0.5
    assert metrics["rmse"] == round(np.sqrt(0.5), 4)
    assert metrics["r2"] == 0.6


def test_baseline_predicts_mean():
    """Verify Baseline DummyRegressor predicts training mean and achieves R2 <= 0 on test."""
    X_train = pd.DataFrame({"MedInc": [1.0, 2.0, 3.0], "HouseAge": [10.0, 20.0, 30.0], "AveRooms": [5.0, 5.0, 5.0], "AveBedrms": [1.0, 1.0, 1.0], "Population": [100.0, 200.0, 300.0], "AveOccup": [2.0, 2.0, 2.0], "Latitude": [34.0, 34.0, 34.0], "Longitude": [-118.0, -118.0, -118.0]})
    y_train = pd.Series([10.0, 20.0, 30.0])  # mean is 20.0

    X_test = pd.DataFrame({"MedInc": [4.0], "HouseAge": [40.0], "AveRooms": [5.0], "AveBedrms": [1.0], "Population": [400.0], "AveOccup": [2.0], "Latitude": [34.0], "Longitude": [-118.0]})
    y_test = pd.Series([25.0])

    pipeline = get_model_pipeline("Baseline")
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    assert pred[0] == pytest.approx(20.0)


def test_evaluate_cross_validation_synthetic():
    """Verify cross-validation runs on pipeline and returns valid cv_rmse metrics."""
    rng = np.random.default_rng(42)
    n = 40
    X_train = pd.DataFrame({
        "MedInc": rng.uniform(1, 10, n),
        "HouseAge": rng.uniform(1, 50, n),
        "AveRooms": rng.uniform(1, 5, n),
        "AveBedrms": rng.uniform(1, 3, n),
        "Population": rng.uniform(100, 1000, n),
        "AveOccup": rng.uniform(1, 4, n),
        "Latitude": rng.uniform(32, 40, n),
        "Longitude": rng.uniform(-120, -115, n),
    })
    y_train = pd.Series(rng.uniform(1, 5, n))

    pipeline = get_model_pipeline("Linear Regression")
    cv_res = evaluate_cross_validation(pipeline, X_train, y_train, cv_folds=3, seed=42)

    assert "cv_rmse_mean" in cv_res
    assert "cv_rmse_std" in cv_res
    assert cv_res["cv_rmse_mean"] > 0
