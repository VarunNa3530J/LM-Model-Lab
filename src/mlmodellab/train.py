"""Main training and experiment execution pipeline.

CLI Entry point:
    python -m mlmodellab.train
"""

from datetime import datetime, timezone
import json
import logging
import platform
import subprocess
import sys
from uuid import uuid4

import pandas as pd
from sklearn.model_selection import train_test_split

# Safe imports check
try:
    import joblib
    import matplotlib
    import numpy as np
    import seaborn
    import sklearn
    import streamlit
except ImportError as err:
    missing_pkg = getattr(err, "name", "required library")
    sys.exit(
        f"Missing dependency: {missing_pkg}. "
        "Run: pip install -r requirements-dev.txt -e ."
    )

from mlmodellab.artifacts import save_artifacts
from mlmodellab.config import (
    BEST_MODEL_PATH,
    ENVIRONMENT_TXT,
    FEATURE_NAMES,
    FIGURES_DIR,
    METADATA_PATH,
    METRICS_CSV,
    MODEL_PARAMS,
    SEED,
    TARGET_NAME,
    TEST_SIZE,
)
from mlmodellab.data import load_housing_data, validate_housing_data, DatasetDownloadError
from mlmodellab.evaluate import compute_metrics, evaluate_cross_validation
from mlmodellab.models import get_all_model_pipelines
from mlmodellab.plots import (
    plot_actual_vs_predicted,
    plot_correlation_heatmap,
    plot_model_comparison,
    plot_residuals,
    plot_target_distribution,
)
from mlmodellab.tracking import log_experiment_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def capture_environment_snapshot() -> None:
    """Save pip freeze snapshot and system information to results/environment.txt."""
    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8")
    except Exception:
        pip_freeze = "pip freeze unavailable"

    content = [
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"Python Version: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"Executable: {sys.executable}",
        "",
        "--- Pip Freeze ---",
        pip_freeze,
    ]
    ENVIRONMENT_TXT.write_text("\n".join(content), encoding="utf-8")
    logger.info("Saved environment snapshot to %s", ENVIRONMENT_TXT)


def run_training() -> None:
    """Execute the full training, evaluation, logging, and plotting pipeline."""
    run_id = f"run_{uuid4().hex[:8]}"
    start_time = datetime.now(timezone.utc).isoformat()
    logger.info("Starting ML Model Lab experiment run: %s", run_id)

    # 1. Capture environment
    capture_environment_snapshot()

    # 2. Load dataset
    logger.info("Loading California Housing dataset...")
    try:
        df = load_housing_data()
    except DatasetDownloadError as e:
        logger.error(str(e))
        print(f"\n[ERROR] {e}\n", file=sys.stderr)
        sys.exit(1)

    # 3. Data validation & quality report
    data_report = validate_housing_data(df)
    logger.info(
        "Data quality check: %d rows, %d cols, %d duplicates, %d missing values.",
        data_report["n_rows"],
        data_report["n_cols"],
        data_report["duplicate_rows"],
        data_report["total_missing"],
    )

    # 4. Exploratory Data Analysis figures
    logger.info("Generating EDA figures...")
    plot_target_distribution(df, target_col=TARGET_NAME, output_dir=FIGURES_DIR)
    plot_correlation_heatmap(df, output_dir=FIGURES_DIR)

    # 5. Split train/test BEFORE any fitting
    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    logger.info(
        "Split dataset: %d train rows, %d test rows (seed=%d, test_size=%.2f)",
        len(X_train),
        len(X_test),
        SEED,
        TEST_SIZE,
    )

    # 6. Fit Baseline first to establish benchmark
    models = get_all_model_pipelines()
    baseline_pipe = models["Baseline"]
    baseline_pipe.fit(X_train, y_train)
    baseline_test_pred = baseline_pipe.predict(X_test)
    baseline_metrics = compute_metrics(y_test, baseline_test_pred)
    baseline_rmse = baseline_metrics["rmse"]
    logger.info("Baseline DummyRegressor Test RMSE: %.4f", baseline_rmse)

    # 7. Cross-Validation, Fitting, and Evaluation for all models
    experiment_results: list[dict] = []
    trained_pipelines: dict = {}
    predictions_map: dict = {}

    for name, pipeline in models.items():
        logger.info("Evaluating model: %s...", name)

        # Cross-validation on training set only
        cv_res = evaluate_cross_validation(pipeline, X_train, y_train, seed=SEED)
        logger.info("  %s CV RMSE: %.4f +/- %.4f", name, cv_res["cv_rmse_mean"], cv_res["cv_rmse_std"])

        # Fit model on full training set
        pipeline.fit(X_train, y_train)
        trained_pipelines[name] = pipeline

        # Final evaluation on held-out test set
        y_test_pred = pipeline.predict(X_test)
        predictions_map[name] = y_test_pred
        test_metrics = compute_metrics(y_test, y_test_pred)
        logger.info(
            "  %s Test Metrics: MAE=%.4f, RMSE=%.4f, R2=%.4f",
            name,
            test_metrics["mae"],
            test_metrics["rmse"],
            test_metrics["r2"],
        )

        beats = bool(test_metrics["rmse"] < baseline_rmse)
        pct_improvement = round(
            float(((baseline_rmse - test_metrics["rmse"]) / baseline_rmse) * 100), 2
        )

        record = {
            "timestamp": start_time,
            "run_id": run_id,
            "model_name": name,
            "parameters": json.dumps(MODEL_PARAMS.get(name, {})),
            "seed": SEED,
            "test_size": TEST_SIZE,
            "cv_rmse_mean": cv_res["cv_rmse_mean"],
            "cv_rmse_std": cv_res["cv_rmse_std"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "baseline_rmse": baseline_rmse,
            "beats_baseline": beats,
            "improvement_over_baseline_pct": pct_improvement,
        }
        experiment_results.append(record)
        log_experiment_run(record)

    # Save summary metrics table
    metrics_df = pd.DataFrame(experiment_results)
    metrics_df.to_csv(METRICS_CSV, index=False)
    logger.info("Saved metrics comparison table to %s", METRICS_CSV)

    # 8. Model Selection (CV RMSE on training set, excluding Baseline)
    candidate_records = [r for r in experiment_results if r["model_name"] != "Baseline"]
    best_record = min(candidate_records, key=lambda r: r["cv_rmse_mean"])
    best_model_name = best_record["model_name"]
    best_pipeline = trained_pipelines[best_model_name]
    logger.info(
        "Selected best model by training CV RMSE: %s (CV RMSE=%.4f)",
        best_model_name,
        best_record["cv_rmse_mean"],
    )

    # 9. Diagnostic Figures
    logger.info("Generating model comparison and diagnostic figures...")
    plot_model_comparison(metrics_df, output_dir=FIGURES_DIR)

    for name in models.keys():
        plot_actual_vs_predicted(y_test, predictions_map[name], model_name=name, output_dir=FIGURES_DIR)
        plot_residuals(y_test, predictions_map[name], model_name=name, output_dir=FIGURES_DIR)

    # 10. Save Best Model and Metadata
    train_feature_ranges = {}
    for feat in FEATURE_NAMES:
        train_feature_ranges[feat] = {
            "min": float(X_train[feat].min()),
            "max": float(X_train[feat].max()),
            "mean": float(X_train[feat].mean()),
            "std": float(X_train[feat].std()),
        }

    metadata = {
        "model_name": best_model_name,
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "cv_rmse_mean": best_record["cv_rmse_mean"],
        "test_metrics": {
            "mae": best_record["test_mae"],
            "rmse": best_record["test_rmse"],
            "r2": best_record["test_r2"],
        },
        "feature_ranges": train_feature_ranges,
        "data_quality_report": data_report,
        "timestamp": start_time,
        "run_id": run_id,
    }
    save_artifacts(best_pipeline, metadata, model_path=BEST_MODEL_PATH, metadata_path=METADATA_PATH)
    logger.info("Saved best model and metadata to %s and %s", BEST_MODEL_PATH, METADATA_PATH)
    logger.info("ML Model Lab training completed successfully!")


if __name__ == "__main__":
    run_training()
