"""Tests for configuration settings and paths."""

from mlmodellab import config


def test_config_constants():
    """Verify seed, split size, and feature lists are set."""
    assert config.SEED == 42
    assert config.TEST_SIZE == 0.2
    assert config.CV_FOLDS == 5
    assert len(config.FEATURE_NAMES) == 8
    assert config.TARGET_NAME == "MedHouseVal"


def test_config_paths_exist():
    """Verify required output folders are creatable and paths defined."""
    assert config.MODELS_DIR.exists()
    assert config.RESULTS_DIR.exists()
    assert config.FIGURES_DIR.exists()
