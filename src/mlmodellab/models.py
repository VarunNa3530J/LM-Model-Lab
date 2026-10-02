"""Model pipeline constructors for ML Model Lab."""

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from mlmodellab.config import MODEL_PARAMS
from mlmodellab.features import build_preprocessor


def get_model_pipeline(model_name: str) -> Pipeline:
    """Build a complete scikit-learn Pipeline with preprocessor and estimator.

    Parameters
    ----------
    model_name : str
        One of 'Baseline', 'Linear Regression', 'Random Forest', 'Gradient Boosting'.

    Returns
    -------
    Pipeline
        Pipeline containing preprocessing and the regressor.

    Raises
    ------
    ValueError
        If unknown model_name is provided.
    """
    preprocessor = build_preprocessor()
    params = MODEL_PARAMS.get(model_name, {})

    if model_name == "Baseline":
        estimator = DummyRegressor(**params)
    elif model_name == "Linear Regression":
        estimator = LinearRegression(**params)
    elif model_name == "Random Forest":
        estimator = RandomForestRegressor(**params)
    elif model_name == "Gradient Boosting":
        estimator = GradientBoostingRegressor(**params)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            "Supported models: 'Baseline', 'Linear Regression', 'Random Forest', 'Gradient Boosting'."
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )
    return pipeline


def get_all_model_pipelines() -> dict[str, Pipeline]:
    """Get all candidate models mapped by their display names."""
    return {
        "Baseline": get_model_pipeline("Baseline"),
        "Linear Regression": get_model_pipeline("Linear Regression"),
        "Random Forest": get_model_pipeline("Random Forest"),
        "Gradient Boosting": get_model_pipeline("Gradient Boosting"),
    }
