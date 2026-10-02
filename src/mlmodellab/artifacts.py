"""Model artifact persistence (saving/loading models and metadata)."""

import json
from pathlib import Path
from typing import Any
import joblib
from sklearn.pipeline import Pipeline

from mlmodellab.config import BEST_MODEL_PATH, METADATA_PATH


class MissingArtifactError(FileNotFoundError):
    """Raised when model artifact or metadata is missing."""
    pass


def save_artifacts(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    model_path: Path = BEST_MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> None:
    """Save the winning model pipeline and metadata to disk.

    Parameters
    ----------
    pipeline : Pipeline
        Trained scikit-learn pipeline (preprocessor + model).
    metadata : dict[str, Any]
        Dictionary of metadata (feature order, ranges, metrics, seed, etc.).
    model_path : Path
        Destination for joblib model file.
    metadata_path : Path
        Destination for metadata JSON file.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_artifacts(
    model_path: Path = BEST_MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> tuple[Pipeline, dict[str, Any]]:
    """Load the trained model pipeline and metadata from disk.

    Parameters
    ----------
    model_path : Path
        Path to joblib file.
    metadata_path : Path
        Path to metadata JSON file.

    Returns
    -------
    tuple[Pipeline, dict[str, Any]]
        (pipeline, metadata)

    Raises
    ------
    MissingArtifactError
        If either file is missing.
    """
    if not model_path.exists():
        raise MissingArtifactError(
            f"Trained model artifact not found at {model_path}. "
            "Please train the models first by running: python -m mlmodellab.train"
        )
    if not metadata_path.exists():
        raise MissingArtifactError(
            f"Model metadata not found at {metadata_path}. "
            "Please train the models first by running: python -m mlmodellab.train"
        )

    pipeline: Pipeline = joblib.load(model_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata: dict[str, Any] = json.load(f)

    return pipeline, metadata
