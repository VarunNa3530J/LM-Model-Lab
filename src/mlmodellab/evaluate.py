"""Evaluation metrics computation and cross-validation."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from mlmodellab.config import CV_FOLDS, SEED


def compute_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R2 regression metrics.

    Explicitly computes RMSE as square root of MSE for broad scikit-learn version compatibility.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        True target values.
    y_pred : pd.Series | np.ndarray
        Predicted target values.

    Returns
    -------
    dict[str, float]
        Dictionary with keys 'mae', 'rmse', 'r2'.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def evaluate_cross_validation(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    seed: int = SEED,
) -> dict[str, float]:
    """Perform K-Fold cross validation on training data using RMSE.

    Parameters
    ----------
    pipeline : Pipeline
        The scikit-learn pipeline (preprocessor + model).
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    cv_folds : int
        Number of folds (default from config).
    seed : int
        Random seed for KFold shuffling.

    Returns
    -------
    dict[str, float]
        Mean CV RMSE and standard deviation.
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    # neg_mean_squared_error is universally supported in scikit-learn
    neg_mse_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring="neg_mean_squared_error",
        cv=kf,
    )
    rmse_scores = np.sqrt(-neg_mse_scores)
    return {
        "cv_rmse_mean": round(float(np.mean(rmse_scores)), 4),
        "cv_rmse_std": round(float(np.std(rmse_scores)), 4),
    }
