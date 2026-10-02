"""Plotting functions for EDA, model comparisons, and regression diagnostics."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless file generation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mlmodellab.config import FIGURES_DIR

# Establish clean, readable plotting aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def plot_target_distribution(df: pd.DataFrame, target_col: str = "MedHouseVal", output_dir: Path = FIGURES_DIR) -> Path:
    """Plot target variable distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[target_col], kde=True, ax=ax, color="#26D67C", bins=40)
    ax.set_title(f"Target Distribution: {target_col} ($100k)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Median House Value ($100,000)")
    ax.set_ylabel("Count")

    output_path = output_dir / "target_distribution.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Plot correlation heatmap across all numeric features and target."""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature & Target Correlation Matrix", fontsize=14, fontweight="bold")

    output_path = output_dir / "correlation_heatmap.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_model_comparison(metrics_df: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> Path:
    """Plot model comparison bar chart highlighting RMSE."""
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["#26D67C" if m != "Baseline" else "#A0A0A0" for m in metrics_df["model_name"]]
    sns.barplot(data=metrics_df, x="model_name", y="test_rmse", hue="model_name", palette=palette, legend=False, ax=ax)

    # Annotate baseline line
    baseline_rmse = metrics_df.loc[metrics_df["model_name"] == "Baseline", "test_rmse"].values[0]
    ax.axhline(baseline_rmse, color="#E53935", linestyle="--", linewidth=1.5, label=f"Baseline RMSE ({baseline_rmse:.4f})")

    for p in ax.patches:
        val = p.get_height()
        if not np.isnan(val) and val > 0:
            ax.annotate(f"{val:.4f}", (p.get_x() + p.get_width() / 2.0, val / 2.0),
                        ha="center", va="center", color="black", fontweight="bold", fontsize=11)

    ax.set_title("Model Comparison: Test RMSE (Lower is Better)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test RMSE ($100,000)")
    ax.set_xlabel("Model")
    ax.legend()

    output_path = output_dir / "model_comparison.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Plot Actual vs Predicted scatter with ideal diagonal."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, color="#26D67C", edgecolors="none", s=20)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")

    ax.set_title(f"Actual vs. Predicted: {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Actual MedHouseVal ($100k)")
    ax.set_ylabel("Predicted MedHouseVal ($100k)")
    ax.legend()

    safe_name = model_name.lower().replace(" ", "_")
    output_path = output_dir / f"actual_vs_predicted_{safe_name}.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> tuple[Path, Path]:
    """Plot Residuals vs Predicted and Residuals distribution histogram."""
    residuals = np.array(y_true) - np.array(y_pred)
    safe_name = model_name.lower().replace(" ", "_")

    # 1. Residuals vs Predicted
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.scatter(y_pred, residuals, alpha=0.3, color="#1976D2", edgecolors="none", s=20)
    ax1.axhline(0, color="red", linestyle="--", lw=1.5)
    ax1.set_title(f"Residuals vs. Predicted: {model_name}", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Predicted MedHouseVal ($100k)")
    ax1.set_ylabel("Residual (Actual - Predicted)")
    res_pred_path = output_dir / f"residuals_vs_predicted_{safe_name}.png"
    fig1.savefig(res_pred_path, dpi=200)
    plt.close(fig1)

    # 2. Residuals histogram
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, ax=ax2, color="#E65100", bins=40)
    ax2.axvline(0, color="black", linestyle="--", lw=1.5)
    ax2.set_title(f"Residuals Distribution: {model_name}", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Residual (Actual - Predicted)")
    ax2.set_ylabel("Count")
    res_dist_path = output_dir / f"residuals_dist_{safe_name}.png"
    fig2.savefig(res_dist_path, dpi=200)
    plt.close(fig2)

    return res_pred_path, res_dist_path
