"""Prediction module: validates single input rows and generates predictions using the saved model."""

from typing import Any
import pandas as pd
from sklearn.pipeline import Pipeline

from mlmodellab.artifacts import load_artifacts
from mlmodellab.config import FEATURE_NAMES


class InputValidationError(ValueError):
    """Raised when prediction input values fail validation."""
    pass


def validate_input_row(input_data: dict[str, Any], metadata: dict[str, Any]) -> pd.DataFrame:
    """Validate a single prediction input dictionary and convert it to a 1-row DataFrame.

    Parameters
    ----------
    input_data : dict[str, Any]
        Dictionary with feature names as keys and numeric values.
    metadata : dict[str, Any]
        Model metadata containing feature names and expected ranges.

    Returns
    -------
    pd.DataFrame
        1-row DataFrame with columns in exact expected feature order.

    Raises
    ------
    InputValidationError
        If missing features, NaN, non-numeric values, or wild out-of-range values occur.
    """
    expected_features = metadata.get("feature_names", FEATURE_NAMES)
    missing = [f for f in expected_features if f not in input_data]
    if missing:
        raise InputValidationError(f"Missing required feature(s): {missing}")

    validated_values = {}
    for feat in expected_features:
        val = input_data[feat]
        if val is None or pd.isna(val):
            raise InputValidationError(f"Feature '{feat}' cannot be null or NaN.")

        try:
            float_val = float(val)
        except (ValueError, TypeError) as exc:
            raise InputValidationError(f"Feature '{feat}' must be numeric. Received: {val}") from exc

        # Wild bounds check: cannot be +/- inf
        if pd.isna(float_val) or float_val in (float("inf"), float("-inf")):
            raise InputValidationError(f"Feature '{feat}' has invalid infinite value.")

        validated_values[feat] = float_val

    df_row = pd.DataFrame([validated_values], columns=expected_features)
    return df_row


def predict_one(
    input_data: dict[str, Any],
    pipeline: Pipeline | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate inputs and predict housing value for a single sample.

    Parameters
    ----------
    input_data : dict[str, Any]
        Dictionary containing the 8 feature values.
    pipeline : Pipeline | None
        Loaded pipeline. If None, loaded from disk.
    metadata : dict[str, Any] | None
        Loaded metadata. If None, loaded from disk.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - raw_prediction: value in original target units ($100k)
        - price_usd: raw_prediction * 100,000
        - model_name: name of winning model
        - warning: optional warning message if any feature is outside training min/max
    """
    if pipeline is None or metadata is None:
        pipeline, metadata = load_artifacts()

    df_row = validate_input_row(input_data, metadata)

    # Check for soft range warnings
    warnings: list[str] = []
    feature_ranges = metadata.get("feature_ranges", {})
    for feat, stats in feature_ranges.items():
        val = df_row.loc[0, feat]
        min_v = stats.get("min")
        max_v = stats.get("max")
        if min_v is not None and val < min_v:
            warnings.append(f"'{feat}' ({val}) is below training minimum ({min_v:.2f}).")
        elif max_v is not None and val > max_v:
            warnings.append(f"'{feat}' ({val}) is above training maximum ({max_v:.2f}).")

    raw_pred = float(pipeline.predict(df_row)[0])
    price_usd = round(raw_pred * 100000.0, 2)

    return {
        "raw_prediction": round(raw_pred, 4),
        "price_usd": price_usd,
        "model_name": metadata.get("model_name", "Best Model"),
        "warnings": warnings,
    }
