"""Data loading, validation, and data quality reporting."""

import logging
from typing import Any
import pandas as pd
from sklearn.datasets import fetch_california_housing

from mlmodellab.config import FEATURE_NAMES, TARGET_NAME

logger = logging.getLogger(__name__)


class DatasetDownloadError(RuntimeError):
    """Raised when the dataset fails to download or load."""
    pass


def load_housing_data(data_home: str | None = None, download_if_missing: bool = True) -> pd.DataFrame:
    """Load the California Housing dataset as a pandas DataFrame.

    Parameters
    ----------
    data_home : str | None
        Optional local cache directory.
    download_if_missing : bool
        Whether to download if missing from cache.

    Returns
    -------
    pd.DataFrame
        Complete DataFrame containing 8 features and MedHouseVal target.

    Raises
    ------
    DatasetDownloadError
        If fetching the dataset fails due to network or download issues.
    """
    try:
        bunch = fetch_california_housing(
            data_home=data_home,
            download_if_missing=download_if_missing,
            as_frame=True,
        )
        df: pd.DataFrame = bunch.frame
        return df
    except Exception as exc:
        msg = (
            f"Failed to fetch California Housing dataset: {exc}. "
            "Please check your internet connection or provide a pre-downloaded dataset in data_home. "
            "Do not fall back to synthetic or fabricated data."
        )
        logger.error(msg)
        raise DatasetDownloadError(msg) from exc


def validate_housing_data(df: pd.DataFrame) -> dict[str, Any]:
    """Validate data schema and generate a comprehensive data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        The California housing dataset.

    Returns
    -------
    dict[str, Any]
        Data quality report dictionary.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required_cols = FEATURE_NAMES + [TARGET_NAME]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

    n_rows = len(df)
    n_cols = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())

    missing_values = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())

    # Range and distribution summary
    feature_ranges = {}
    for col in required_cols:
        feature_ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
        }

    report = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "duplicate_rows": duplicate_rows,
        "total_missing": total_missing,
        "missing_per_column": missing_values,
        "feature_ranges": feature_ranges,
    }
    return report
