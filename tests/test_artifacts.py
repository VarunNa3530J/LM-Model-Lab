"""Tests for tracking logs and artifact persistence."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from mlmodellab.artifacts import load_artifacts, save_artifacts, MissingArtifactError
from mlmodellab.models import get_model_pipeline
from mlmodellab.tracking import log_experiment_run, REQUIRED_COLUMNS


def test_tracking_logging(tmp_path: Path):
    """Test that tracking logs record to CSV and JSON accurately."""
    csv_path = tmp_path / "experiments.csv"
    json_path = tmp_path / "experiments.json"

    record_1 = {
        "timestamp": "2026-10-02T12:00:00",
        "run_id": "test-run-1",
        "model_name": "Linear Regression",
        "parameters": "{}",
        "seed": 42,
        "test_size": 0.2,
        "cv_rmse_mean": 0.72,
        "cv_rmse_std": 0.02,
        "test_mae": 0.53,
        "test_rmse": 0.71,
        "test_r2": 0.60,
        "baseline_rmse": 1.15,
        "beats_baseline": True,
        "improvement_over_baseline_pct": 38.26,
    }

    log_experiment_run(record_1, csv_path=csv_path, json_path=json_path)

    assert csv_path.exists()
    assert json_path.exists()

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.loc[0, "model_name"] == "Linear Regression"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["run_id"] == "test-run-1"

    # Incomplete record raises ValueError
    with pytest.raises(ValueError, match="missing required experiment logging field"):
        log_experiment_run({"timestamp": "now"}, csv_path=csv_path, json_path=json_path)


def test_artifact_save_and_load(tmp_path: Path):
    """Test saving pipeline and metadata then loading produces identical predictions."""
    model_path = tmp_path / "best_model.joblib"
    meta_path = tmp_path / "metadata.json"

    pipeline = get_model_pipeline("Linear Regression")
    X = pd.DataFrame({
        "MedInc": [1.0, 2.0, 3.0],
        "HouseAge": [10.0, 20.0, 30.0],
        "AveRooms": [5.0, 5.0, 5.0],
        "AveBedrms": [1.0, 1.0, 1.0],
        "Population": [100.0, 200.0, 300.0],
        "AveOccup": [2.0, 2.0, 2.0],
        "Latitude": [34.0, 34.0, 34.0],
        "Longitude": [-118.0, -118.0, -118.0],
    })
    y = pd.Series([1.5, 2.5, 3.5])
    pipeline.fit(X, y)
    original_preds = pipeline.predict(X)

    metadata = {
        "model_name": "Linear Regression",
        "features": list(X.columns),
        "seed": 42,
    }

    save_artifacts(pipeline, metadata, model_path=model_path, metadata_path=meta_path)
    loaded_pipe, loaded_meta = load_artifacts(model_path=model_path, metadata_path=meta_path)

    loaded_preds = loaded_pipe.predict(X)
    np.testing.assert_allclose(original_preds, loaded_preds)
    assert loaded_meta["model_name"] == "Linear Regression"


def test_load_artifacts_missing():
    """Test clear exception raised if artifacts are missing."""
    non_existent = Path("non_existent_folder_abc/model.joblib")
    with pytest.raises(MissingArtifactError, match="Trained model artifact not found"):
        load_artifacts(model_path=non_existent, metadata_path=non_existent)
