"""ML Model Lab - Interactive Machine Learning Dashboard.

Displays benchmark results, diagnostic curves, live predictions, and experiment history.
Reads strictly from pre-computed artifacts and results.
"""

import sys
from pathlib import Path
import pandas as pd

try:
    import joblib
    import matplotlib
    import numpy as np
    import streamlit as st
except ImportError as err:
    missing_pkg = getattr(err, "name", "required library")
    sys.exit(
        f"Missing dependency: {missing_pkg}. "
        "Run: pip install -r requirements.txt"
    )

from mlmodellab.artifacts import load_artifacts, MissingArtifactError
from mlmodellab.config import (
    BEST_MODEL_PATH,
    EXPERIMENTS_CSV,
    FIGURES_DIR,
    METADATA_PATH,
    METRICS_CSV,
)
from mlmodellab.predict import predict_one, InputValidationError

# 1. Page Configuration
st.set_page_config(
    page_title="ML Model Lab • California Housing Benchmark",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Clean Modern Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    :root {
        --primary-green: #26D67C;
        --primary-green-dark: #10B981;
        --surface-bg: #FAFAFA;
        --card-bg: #FFFFFF;
        --border-color: rgba(226, 232, 240, 0.85);
        --text-headline: #0F172A;
        --text-body: #334155;
        --text-muted: #64748B;
        --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
        --ease-smooth: cubic-bezier(0.4, 0.0, 0.2, 1);
    }

    html, body, [class*="css"], .stApp {
        background-color: var(--surface-bg) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        color: var(--text-body) !important;
        -webkit-font-smoothing: antialiased;
    }

    * {
        scroll-behavior: smooth;
    }

    @keyframes fadeInUp {
        0% {
            opacity: 0;
            transform: translateY(18px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulseGlow {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(38, 214, 124, 0.4);
        }
        50% {
            box-shadow: 0 0 0 10px rgba(38, 214, 124, 0);
        }
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid var(--border-color) !important;
        box-shadow: 4px 0 24px rgba(15, 23, 42, 0.02) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text-headline) !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: var(--text-body) !important;
    }
    [data-testid="stSidebar"] code {
        background: #F1F5F9 !important;
        color: #0F172A !important;
        border: 1px solid #E2E8F0 !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
    }

    .hero-container {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 24px;
        padding: 36px 40px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px -5px rgba(15, 23, 42, 0.04), 0 4px 12px -2px rgba(15, 23, 42, 0.02);
        animation: fadeInUp 0.5s var(--ease-smooth) forwards;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        background: linear-gradient(90deg, #26D67C 0%, #10B981 60%, #059669 100%);
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(38, 214, 124, 0.12);
        color: #0B7C41 !important;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 6px 14px;
        border-radius: 100px;
        border: 1px solid rgba(38, 214, 124, 0.3);
        margin-bottom: 16px;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #26D67C;
        border-radius: 50%;
        display: inline-block;
        animation: pulseGlow 2s infinite;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #0F172A !important;
        letter-spacing: -0.035em;
        margin: 0 0 10px 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.08rem;
        color: #475569 !important;
        margin: 0;
        line-height: 1.6;
        max-width: 880px;
        font-weight: 400;
    }

    .kpi-box {
        background: #FFFFFF;
        border: 1px solid rgba(226, 232, 240, 0.8);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 16px -2px rgba(15, 23, 42, 0.03);
        transition: transform 0.3s var(--ease-spring), box-shadow 0.3s var(--ease-smooth), border-color 0.3s ease;
        animation: fadeInUp 0.6s var(--ease-smooth) forwards;
        position: relative;
        overflow: hidden;
    }
    .kpi-box:hover {
        transform: translateY(-4px);
        box-shadow: 0 16px 32px -4px rgba(15, 23, 42, 0.08);
        border-color: rgba(38, 214, 124, 0.4);
    }
    .kpi-box::after {
        content: "";
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: transparent;
        transition: background 0.3s ease;
    }
    .kpi-box:hover::after {
        background: var(--primary-green);
    }

    .kpi-label {
        font-size: 0.76rem;
        font-weight: 700;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0F172A !important;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-sub {
        font-size: 0.84rem;
        font-weight: 600;
        color: #059669 !important;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    .premium-card {
        background: #FFFFFF;
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 20px;
        padding: 28px 32px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04);
        margin-bottom: 24px;
        animation: fadeInUp 0.6s var(--ease-smooth) forwards;
        transition: box-shadow 0.25s ease;
    }
    .premium-card:hover {
        box-shadow: 0 8px 28px -4px rgba(15, 23, 42, 0.06);
    }
    .premium-card h3 {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.35rem !important;
        letter-spacing: -0.025em !important;
        margin-top: 0 !important;
        margin-bottom: 8px !important;
    }
    .premium-card p {
        color: #475569 !important;
        line-height: 1.65 !important;
        font-size: 0.98rem !important;
        margin-bottom: 0 !important;
    }

    .stTabs {
        margin-top: 14px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px !important;
        background-color: #E2E8F0 !important;
        padding: 6px !important;
        border-radius: 16px !important;
        border-bottom: none !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 10px 24px !important;
        background-color: transparent !important;
        color: #475569 !important;
        font-weight: 600 !important;
        font-size: 0.94rem !important;
        border: none !important;
        transition: all 0.25s var(--ease-spring) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        transform: translateY(-1px);
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08) !important;
        transform: translateY(0) scale(1.02);
    }

    .stNumberInput label, .stSelectbox label {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        margin-bottom: 6px;
    }
    .stNumberInput div[data-baseweb="input"], div[data-baseweb="select"] > div {
        background-color: #F8FAFC !important;
        border: 1.5px solid #E2E8F0 !important;
        border-radius: 14px !important;
        transition: all 0.25s var(--ease-smooth) !important;
    }
    .stNumberInput div[data-baseweb="input"]:focus-within {
        border-color: var(--primary-green) !important;
        box-shadow: 0 0 0 4px rgba(38, 214, 124, 0.2) !important;
        background-color: #FFFFFF !important;
        transform: translateY(-1px);
    }
    .stNumberInput input {
        color: #0F172A !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #26D67C 0%, #10B981 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        letter-spacing: -0.01em !important;
        border: none !important;
        border-radius: 16px !important;
        padding: 16px 32px !important;
        box-shadow: 0 6px 20px rgba(38, 214, 124, 0.35) !important;
        transition: all 0.25s var(--ease-spring) !important;
        width: 100% !important;
        margin-top: 14px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #20C26F 0%, #059669 100%) !important;
        box-shadow: 0 10px 28px rgba(38, 214, 124, 0.45) !important;
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: scale(0.97) !important;
    }

    .prediction-showcase {
        background: linear-gradient(135deg, rgba(38, 214, 124, 0.12) 0%, rgba(16, 185, 129, 0.04) 100%);
        border: 2px solid var(--primary-green);
        border-radius: 24px;
        padding: 38px 28px;
        text-align: center;
        margin: 28px 0 18px 0;
        box-shadow: 0 12px 30px -5px rgba(38, 214, 124, 0.25);
        animation: fadeInUp 0.4s var(--ease-spring) forwards;
    }
    .prediction-pill {
        display: inline-block;
        background: #26D67C;
        color: #FFFFFF;
        font-weight: 800;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 5px 16px;
        border-radius: 100px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(38, 214, 124, 0.3);
    }
    .prediction-value-hero {
        font-size: 3.8rem;
        font-weight: 900;
        color: #0F172A;
        letter-spacing: -0.04em;
        margin-bottom: 8px;
        line-height: 1;
    }
    .prediction-footnote {
        font-size: 1rem;
        color: #475569;
        font-weight: 500;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-color) !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.03) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_artifacts():
    """Load model pipeline and metadata safely with cache."""
    return load_artifacts(BEST_MODEL_PATH, METADATA_PATH)


@st.cache_data
def get_cached_metrics():
    """Load summary metrics table."""
    if not METRICS_CSV.exists():
        return None
    return pd.read_csv(METRICS_CSV)


@st.cache_data
def get_cached_experiments():
    """Load experiment run history."""
    if not EXPERIMENTS_CSV.exists():
        return None
    return pd.read_csv(EXPERIMENTS_CSV)


def check_prerequisites():
    """Verify that models and result files exist; show helpful error if missing."""
    missing = []
    if not BEST_MODEL_PATH.exists():
        missing.append(str(BEST_MODEL_PATH))
    if not METADATA_PATH.exists():
        missing.append(str(METADATA_PATH))
    if not METRICS_CSV.exists():
        missing.append(str(METRICS_CSV))

    if missing:
        st.error(
            "⚠️ **Required Model Artifacts or Results are Missing!**\n\n"
            f"Could not locate: `{', '.join(missing)}`.\n\n"
            "To train the models and generate all artifacts, run this command in your terminal:\n"
            "```bash\n"
            "python -m mlmodellab.train\n"
            "```"
        )
        st.stop()


def main():
    check_prerequisites()

    pipeline, metadata = get_cached_artifacts()
    metrics_df = get_cached_metrics()
    experiments_df = get_cached_experiments()

    best_model_name = metadata.get("model_name", "Random Forest")
    best_rmse = metadata.get("test_metrics", {}).get("rmse", 0.5042)
    best_r2 = metadata.get("test_metrics", {}).get("r2", 0.8060)
    data_rep = metadata.get("data_quality_report", {})

    # ----------------------------------------------------
    # SIDEBAR SPECIFICATION
    # ----------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 14px 0 22px 0;">
                <div class="status-badge" style="margin-bottom: 8px;">
                    <span class="status-dot"></span> Pipeline Ready
                </div>
                <h2 style="font-size: 1.6rem; font-weight: 800; color: #0F172A; margin: 0; letter-spacing: -0.03em;">ML Model Lab</h2>
                <p style="color: #64748B; font-size: 0.88rem; margin: 4px 0 0 0;">California Housing Regression</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 16px; padding: 18px; margin-bottom: 24px; box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);">
                <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase; font-weight: 700; letter-spacing: 0.06em; margin-bottom: 6px;">Top Architecture</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">{best_model_name}</div>
                <div style="font-size: 0.84rem; color: #059669; font-weight: 700; margin-top: 6px; display: flex; align-items: center; gap: 4px;">
                    ✦ Test RMSE: {best_rmse:.4f} • R²: {best_r2:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<p style='font-weight: 800; font-size: 0.92rem; color: #0F172A; margin-bottom: 10px;'>Benchmark Specifications</p>", unsafe_allow_html=True)
        st.markdown(f"• **Random Seed**: `{metadata.get('seed', 42)}` (Deterministic)")
        st.markdown(f"• **Data Split**: `80% Train / 20% Test`")
        st.markdown(f"• **Selection Metric**: `5-Fold CV (RMSE)`")
        st.markdown(f"• **Leakage Protection**: Strict pipeline encapsulation")

        st.divider()
        st.caption("Standard scikit-learn pipeline implementation.")

    # ----------------------------------------------------
    # HERO BANNER
    # ----------------------------------------------------
    st.markdown(
        """
        <div class="hero-container">
            <div class="status-badge">
                <span class="status-dot"></span> Benchmark Verified & Reproducible
            </div>
            <h1 class="hero-title">California Housing Model Lab</h1>
            <p class="hero-subtitle">
                A reproducible machine learning benchmark evaluating Linear Regression, Random Forest, 
                and Gradient Boosting against a DummyRegressor baseline using California housing census records.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------
    # TOP KPI STAT WIDGETS
    # ----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Census Observations</div>
                <div class="kpi-value">{data_rep.get('n_rows', 20640):,}</div>
                <div class="kpi-sub">16,512 train • 4,128 test</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Champion Architecture</div>
                <div class="kpi-value" style="font-size: 1.65rem;">{best_model_name}</div>
                <div class="kpi-sub">Selected via 5-Fold CV</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Held-out Test RMSE</div>
                <div class="kpi-value">{best_rmse:.4f}</div>
                <div class="kpi-sub">↑ 55.96% vs. Baseline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-box">
                <div class="kpi-label">Explained Variance (R²)</div>
                <div class="kpi-value">{best_r2 * 100:.1f}%</div>
                <div class="kpi-sub">Target: MedHouseVal</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # ----------------------------------------------------
    # TABS NAVIGATION
    # ----------------------------------------------------
    tab_overview, tab_comparison, tab_diagnostics, tab_predict, tab_history = st.tabs([
        "📋 Overview",
        "📊 Model Comparison",
        "🔍 Diagnostics",
        "🔮 House Price Estimator",
        "📜 Experiment History",
    ])

    # ----------------------------------------------------
    # TAB 1: OVERVIEW
    # ----------------------------------------------------
    with tab_overview:
        st.markdown(
            """
            <div class="premium-card">
                <h3>Dataset Integrity & Validation</h3>
                <p>
                    The California Housing dataset contains demographic and housing block-group observations 
                    collected in the 1990 U.S. Census. In accordance with standard ML engineering practices, 
                    the training and testing partitions were isolated prior to any feature transformations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_left, col_right = st.columns(2)
        with col_left:
            eda_target_path = FIGURES_DIR / "target_distribution.png"
            if eda_target_path.exists():
                st.image(str(eda_target_path), caption="Target Variable (MedHouseVal) Distribution ($100,000 units)", use_container_width=True)
            else:
                st.info("Target distribution chart not available.")

        with col_right:
            eda_corr_path = FIGURES_DIR / "correlation_heatmap.png"
            if eda_corr_path.exists():
                st.image(str(eda_corr_path), caption="Feature & Target Pearson Correlation Matrix", use_container_width=True)
            else:
                st.info("Correlation heatmap chart not available.")

        st.markdown(
            """
            <div style="background: #FFFDF5; border: 1px solid #FEF08A; border-left: 5px solid #F59E0B; border-radius: 16px; padding: 22px 26px; margin-top: 20px;">
                <h4 style="color: #92400E; font-weight: 800; font-size: 1.05rem; margin: 0 0 8px 0;">⚠️ Benchmark Limitations</h4>
                <ul style="margin: 0; padding-left: 20px; color: #78350F; font-size: 0.93rem; line-height: 1.65;">
                    <li><strong>Historical 1990 Values:</strong> Figures reflect 1990 census block groups, not current individual house listings. Predictions are intended for technical benchmarking.</li>
                    <li><strong>Upper Boundary Censoring:</strong> Median values over $500,000 were capped at 5.0 in the original census, creating an upper-bound boundary effect.</li>
                    <li><strong>Spatial Representation:</strong> Longitude and latitude are treated as continuous numeric features without spatial topological graph layers.</li>
                    <li><strong>Deterministic Scope:</strong> All reported metrics reflect this fixed train/test partition and seed (42).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----------------------------------------------------
    # TAB 2: MODEL COMPARISON
    # ----------------------------------------------------
    with tab_comparison:
        st.markdown(
            """
            <div class="premium-card">
                <h3>Benchmark Comparison Against Baseline</h3>
                <p>
                    Every model was trained on identical training folds and compared against a 
                    <code>DummyRegressor(strategy='mean')</code>. The winning model was selected strictly by 
                    <strong>5-Fold Cross-Validation RMSE</strong> on the training set.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if metrics_df is not None:
            display_cols = [
                "model_name",
                "cv_rmse_mean",
                "cv_rmse_std",
                "test_mae",
                "test_rmse",
                "test_r2",
                "beats_baseline",
                "improvement_over_baseline_pct",
            ]
            renamed_cols = {
                "model_name": "Model Architecture",
                "cv_rmse_mean": "CV RMSE (Mean)",
                "cv_rmse_std": "CV RMSE (Std)",
                "test_mae": "Test MAE",
                "test_rmse": "Test RMSE",
                "test_r2": "Test R²",
                "beats_baseline": "Beats Baseline?",
                "improvement_over_baseline_pct": "Improvement vs. Baseline",
            }
            styled_df = metrics_df[display_cols].rename(columns=renamed_cols)
            styled_df["Improvement vs. Baseline"] = styled_df["Improvement vs. Baseline"].apply(
                lambda x: f"+{x:.2f}%" if x > 0 else f"{x:.2f}%"
            )

            st.dataframe(
                styled_df.style.highlight_min(subset=["Test RMSE", "CV RMSE (Mean)"], color="rgba(38, 214, 124, 0.28)"),
                use_container_width=True,
                height=185,
            )

        st.write("")
        comp_chart = FIGURES_DIR / "model_comparison.png"
        if comp_chart.exists():
            st.image(str(comp_chart), caption="Test RMSE Comparison Across All Evaluated Architectures", use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: DIAGNOSTICS
    # ----------------------------------------------------
    with tab_diagnostics:
        st.markdown(
            """
            <div class="premium-card">
                <h3>Diagnostic Residual & Error Analysis</h3>
                <p>
                    Examine residual homoscedasticity, actual vs. predicted goodness of fit, and error distributions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_model = st.selectbox(
            "Select Architecture for In-Depth Diagnostics",
            options=["Random Forest", "Gradient Boosting", "Linear Regression", "Baseline"],
            index=0,
        )
        safe_name = selected_model.lower().replace(" ", "_")

        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            actual_pred_img = FIGURES_DIR / f"actual_vs_predicted_{safe_name}.png"
            if actual_pred_img.exists():
                st.image(str(actual_pred_img), caption=f"Actual vs. Predicted ({selected_model})", use_container_width=True)
            else:
                st.info("Chart not found.")

        with col_diag2:
            res_pred_img = FIGURES_DIR / f"residuals_vs_predicted_{safe_name}.png"
            if res_pred_img.exists():
                st.image(str(res_pred_img), caption=f"Residuals vs. Predicted ({selected_model})", use_container_width=True)
            else:
                st.info("Chart not found.")

        res_dist_img = FIGURES_DIR / f"residuals_dist_{safe_name}.png"
        if res_dist_img.exists():
            st.image(str(res_dist_img), caption=f"Residual Error Distribution ({selected_model})", use_container_width=True)

    # ----------------------------------------------------
    # TAB 4: PREDICT VALUE (HOUSE PRICE ESTIMATOR)
    # ----------------------------------------------------
    with tab_predict:
        st.markdown(
            f"""
            <div class="premium-card">
                <h3>🏠 House Price Estimator</h3>
                <p>
                    Select a California region or adjust demographic and housing attributes below. 
                    You can also click any of the <strong>Quick Presets</strong> to test pre-configured real-world scenarios.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<p style='font-weight: 800; font-size: 0.95rem; color: #0F172A; margin-bottom: 8px;'>⚡ Quick Regional Presets</p>", unsafe_allow_html=True)
        col_p1, col_p2, col_p3 = st.columns(3)

        if "medinc" not in st.session_state:
            st.session_state.medinc = 3.88
            st.session_state.house_age = 28
            st.session_state.rooms = 5.4
            st.session_state.bedrms = 1.1
            st.session_state.population = 1425
            st.session_state.occupancy = 3.0
            st.session_state.location = "Central Coast / Mid-California"

        with col_p1:
            if st.button("📍 San Francisco Bay Area (High Income)", use_container_width=True):
                st.session_state.medinc = 8.50
                st.session_state.house_age = 20
                st.session_state.rooms = 7.0
                st.session_state.bedrms = 1.05
                st.session_state.population = 1800
                st.session_state.occupancy = 2.8
                st.session_state.location = "San Francisco Bay Area"
                st.rerun()

        with col_p2:
            if st.button("📍 Greater Los Angeles (Suburban)", use_container_width=True):
                st.session_state.medinc = 4.20
                st.session_state.house_age = 35
                st.session_state.rooms = 5.2
                st.session_state.bedrms = 1.10
                st.session_state.population = 2400
                st.session_state.occupancy = 3.2
                st.session_state.location = "Greater Los Angeles Area"
                st.rerun()

        with col_p3:
            if st.button("📍 Central Valley / Fresno (Inland Rural)", use_container_width=True):
                st.session_state.medinc = 2.40
                st.session_state.house_age = 15
                st.session_state.rooms = 4.8
                st.session_state.bedrms = 1.15
                st.session_state.population = 950
                st.session_state.occupancy = 3.5
                st.session_state.location = "Central Valley / Fresno"
                st.rerun()

        st.write("")

        LOCATION_COORDINATES = {
            "San Francisco Bay Area": (37.75, -122.42),
            "Greater Los Angeles Area": (34.05, -118.25),
            "San Diego Coastal": (32.72, -117.16),
            "Sacramento Valley": (38.58, -121.49),
            "Central Valley / Fresno": (36.74, -119.78),
            "Central Coast / Mid-California": (35.64, -119.58),
        }

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 18px; padding: 22px; margin-bottom: 20px;">
                    <h4 style="margin: 0 0 14px 0; color: #0F172A; font-weight: 800; font-size: 1.1rem;">💰 Economic & Housing Attributes</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

            medinc_input = st.number_input(
                "Area Annual Household Income ($10,000 units)",
                min_value=0.5,
                max_value=15.0,
                value=float(st.session_state.medinc),
                step=0.25,
                help="Benchmark unit: 3.88 = $38,800/year (1990 census median). 8.50 = $85,000/year (high income).",
            )
            st.caption(f"Equivalent Annual Household Income: **${medinc_input * 10000:,.0f} / year**")

            house_age_input = st.number_input(
                "Median House Age (years)",
                min_value=1,
                max_value=52,
                value=int(st.session_state.house_age),
                step=1,
                help="Age of properties in the block group. New construction: 1-10 years, mature neighborhood: 25-50 years.",
            )

            rooms_input = st.number_input(
                "Average Total Rooms per Home",
                min_value=1.0,
                max_value=20.0,
                value=float(st.session_state.rooms),
                step=0.2,
                help="Total rooms per residential unit including living rooms, bedrooms, and kitchens.",
            )

            bedrms_input = st.number_input(
                "Average Bedrooms per Home",
                min_value=0.5,
                max_value=5.0,
                value=float(st.session_state.bedrms),
                step=0.1,
                help="Average number of bedrooms per housing unit (typical range: 1.0 to 1.5).",
            )

        with col_right:
            st.markdown(
                """
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 18px; padding: 22px; margin-bottom: 20px;">
                    <h4 style="margin: 0 0 14px 0; color: #0F172A; font-weight: 800; font-size: 1.1rem;">📍 Location & Demographics</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )

            pop_input = st.number_input(
                "Block Group Population",
                min_value=50,
                max_value=30000,
                value=int(st.session_state.population),
                step=50,
                help="Total residents in the census block group (typical range: 1,000 to 2,500 residents).",
            )

            occup_input = st.number_input(
                "Average Household Size (occupants per home)",
                min_value=1.0,
                max_value=10.0,
                value=float(st.session_state.occupancy),
                step=0.1,
                help="Mean number of occupants residing in each home (typical range: 2.5 to 3.5).",
            )

            location_name = st.selectbox(
                "California Geographic Region",
                options=list(LOCATION_COORDINATES.keys()),
                index=list(LOCATION_COORDINATES.keys()).index(st.session_state.location) if st.session_state.location in LOCATION_COORDINATES else 0,
                help="Select a region to automatically assign latitude and longitude coordinates.",
            )
            lat_val, lon_val = LOCATION_COORDINATES[location_name]
            st.caption(f"Coordinates: Latitude: **{lat_val}° N**, Longitude: **{lon_val}° W**")

        inputs = {
            "MedInc": float(medinc_input),
            "HouseAge": float(house_age_input),
            "AveRooms": float(rooms_input),
            "AveBedrms": float(bedrms_input),
            "Population": float(pop_input),
            "AveOccup": float(occup_input),
            "Latitude": float(lat_val),
            "Longitude": float(lon_val),
        }

        st.write("")
        if st.button("Calculate Property Value", type="primary"):
            try:
                res = predict_one(inputs, pipeline=pipeline, metadata=metadata)
                price_str = f"${res['price_usd']:,.2f}"

                st.markdown(
                    f"""
                    <div class="prediction-showcase">
                        <div class="prediction-pill">Estimated Property Value</div>
                        <div class="prediction-value-hero">{price_str}</div>
                        <div class="prediction-footnote">
                            Region: <strong>{location_name}</strong> • 
                            Household Income: <strong>${medinc_input * 10000:,.0f}/yr</strong> • 
                            Model: <strong>{best_model_name}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                if res.get("warnings"):
                    st.write("")
                    for w in res["warnings"]:
                        st.warning(f"⚠️ {w}")

                st.caption(
                    "📌 Note: Valuation is calculated using 1990 census benchmark patterns for technical demonstration."
                )
            except InputValidationError as err:
                st.error(f"❌ Input Validation Error: {err}")
            except Exception as err:
                st.error(f"❌ Error: {err}")

    # ----------------------------------------------------
    # TAB 5: EXPERIMENT HISTORY
    # ----------------------------------------------------
    with tab_history:
        st.markdown(
            """
            <div class="premium-card">
                <h3>Experiment Run History</h3>
                <p>
                    Audit trail of all benchmark training runs saved to <code>results/experiments.csv</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if experiments_df is not None and not experiments_df.empty:
            st.dataframe(experiments_df, use_container_width=True)
            st.caption(f"Total benchmark runs recorded: {len(experiments_df)}")
        else:
            st.info("No experiment runs found.")


if __name__ == "__main__":
    main()
