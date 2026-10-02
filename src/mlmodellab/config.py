"""Configuration settings and constants for ML Model Lab."""

from pathlib import Path

# Fixed random seed for total reproducibility
SEED: int = 42

# Train / Test split ratio
TEST_SIZE: float = 0.2

# Cross-validation folds on training set
CV_FOLDS: int = 5

# Paths (relative to repository root)
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
RESULTS_DIR: Path = ROOT_DIR / "results"
REPORTS_DIR: Path = ROOT_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# File paths
BEST_MODEL_PATH: Path = MODELS_DIR / "best_model.joblib"
METADATA_PATH: Path = MODELS_DIR / "metadata.json"
EXPERIMENTS_CSV: Path = RESULTS_DIR / "experiments.csv"
EXPERIMENTS_JSON: Path = RESULTS_DIR / "experiments.json"
METRICS_CSV: Path = RESULTS_DIR / "metrics.csv"
ENVIRONMENT_TXT: Path = RESULTS_DIR / "environment.txt"

# Feature definitions for California Housing
FEATURE_NAMES: list[str] = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

TARGET_NAME: str = "MedHouseVal"

# Model Hyperparameters
MODEL_PARAMS: dict = {
    "Baseline": {
        "strategy": "mean",
    },
    "Linear Regression": {},
    "Random Forest": {
        "n_estimators": 200,
        "random_state": SEED,
        "n_jobs": -1,
    },
    "Gradient Boosting": {
        "random_state": SEED,
    },
}
