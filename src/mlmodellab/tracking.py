"""Experiment tracking: logs each model training run to CSV and JSON."""

import json
from pathlib import Path
from typing import Any
import pandas as pd

from mlmodellab.config import EXPERIMENTS_CSV, EXPERIMENTS_JSON

REQUIRED_COLUMNS = [
    "timestamp",
    "run_id",
    "model_name",
    "parameters",
    "seed",
    "test_size",
    "cv_rmse_mean",
    "cv_rmse_std",
    "test_mae",
    "test_rmse",
    "test_r2",
    "baseline_rmse",
    "beats_baseline",
    "improvement_over_baseline_pct",
]


def log_experiment_run(record: dict[str, Any], csv_path: Path = EXPERIMENTS_CSV, json_path: Path = EXPERIMENTS_JSON) -> None:
    """Append a completed experiment run record to CSV and JSON tracking files.

    Parameters
    ----------
    record : dict[str, Any]
        Dictionary containing all experiment fields.
    csv_path : Path
        Path to experiments CSV.
    json_path : Path
        Path to experiments JSON.
    """
    for col in REQUIRED_COLUMNS:
        if col not in record:
            raise ValueError(f"Record is missing required experiment logging field: {col}")

    # 1. Update CSV
    df_new = pd.DataFrame([record])
    if csv_path.exists() and csv_path.stat().st_size > 0:
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # Serialize parameters dict to string for CSV cleanliness
    df_combined.to_csv(csv_path, index=False)

    # 2. Update JSON
    history: list[dict[str, Any]] = []
    if json_path.exists() and json_path.stat().st_size > 0:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(record)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
