"""Feature transformation and preprocessing pipeline construction."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlmodellab.config import FEATURE_NAMES


def build_preprocessor() -> Pipeline:
    """Build the feature preprocessing pipeline.

    Includes median imputation followed by standard scaling.
    Preprocessing is fitted exclusively on training data within Pipeline.

    Returns
    -------
    Pipeline
        Preprocessing pipeline for numeric features.
    """
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return preprocessor
