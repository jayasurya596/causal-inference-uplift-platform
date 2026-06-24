"""
06_dashboard.py — Interactive Streamlit Dashboard
===================================================

Launch with:  streamlit run 06_dashboard.py

A 5-tab interactive dashboard that lets stakeholders explore:
1. Problem Setup — data & confounding evidence
2. Power Analysis — sample size & detectable effects
3. Causal Estimates — naïve vs. corrected methods
4. Uplift Model — segmentation, Qini curves, targeting
5. Business Impact — ROI comparison & recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Causal Inference & Uplift Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
DATA_DIR = OUTPUT_DIR / "data"

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ──────────────────────────────────────────────

def load_data():
    """Load all generated data and tables."""
    data = {}
    try:
        data["df"] = pd.read_csv(DATA_DIR / "synthetic_data.csv")
    except FileNotFoundError:
        st.error("⚠️ Data not found! Run the pipeline first: `python 01_data_generation.py`")
        st.stop()

    for name, filename in [
        ("power_summary", "02_power_analysis_summary.csv"),
        ("ate_comparison", "03_ate_comparison.csv"),
        ("roi_comparison", "04_roi_comparison.csv"),
        ("segment_profile", "04_segment_profile.csv"),
        ("sensitivity_summary", "05_sensitivity_summary.csv"),
        ("covariate_balance", "05_covariate_balance.csv"),
        ("rosenbaum_bounds", "05_rosenbaum_bounds.csv"),
    ]:
        try:
            data[name] = pd.read_csv(TABLES_DIR / filename)
        except FileNotFoundError:
            data[name] = None

    try:
        data["segmented"] = pd.read_csv(DATA_DIR / "segmented_users.csv")
    except FileNotFoundError:
        data["segmented"] = None

    return data


def show_figure(filename, caption=""):
    """Display a saved figure if it exists."""
    path = FIGURES_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Figure not found: {filename}. Run the corresponding pipeline step.")


# ═══════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════

def main():
    st.markdown('<h1 class="main-header">🔬 Causal Inference & Uplift Modeling Platform</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center; font-size:1.2rem; color:#555;">'
        'Demonstrating why correlation ≠ causation in marketing decisions'
        '</p>',
        unsafe_allow_html=True,
    )

    data = load_data()
    df = data["df"]

    # ─── Sidebar ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 📊 Dataset Overview")
        st.metric("Total Users", f"{len(df):,}")
        st.metric("Treated", f"{df['treatment'].sum():,}")
        st.metric("Control", f"{(1 - df['treatment']).sum():,}")

        naive_ate = (
            df[df["treatment"] == 1]["purchase_amount"].mean()
            - df[df["treatment"] == 0]["purchase_amount"].mean()
        )
        true_ate = df["tau_true"].mean()

        st.divider()
        st.markdown("### 🎯 Key Numbers")
        st.metric("Naïve ATE (WRONG)", f"₹{naive_ate:,.0f}", delta=f"+{(naive_ate/true_ate-1)*100:.0f}% bias", delta_color="inverse")
        st.metric("True ATE", f"₹{true_ate:,.0f}")
        st.metric("Bias", f"₹{naive_ate - true_ate:,.0f}")

        st.divider()
        st.markdown(
            "**Core Insight:** The treated group has systematically "
            "higher-value users, inflating the naïve estimate by ~2×."
        )

    # ─── Tabs ─────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Problem Setup",
        "📊 Power Analysis",
        "🔬 Causal Estimates",
        "🎯 Uplift Model",
        "💰 Business Impact",
    ])

    # ═══════════════════════════════════════════════════════════════
    # TAB 1: Problem Setup
    # ═══════════════════════════════════════════════════════════════
    with tab1:
        st.header("Problem Setup: Confounded Treatment Assignment")

        st.markdown("""
        > **Scenario:** An e-commerce company sent discount emails to a subset of users.
        > The marketing team selected recipients based on customer value — **high-value
        > users were more likely to receive the discount.** This creates selection bias
        > that makes naïve analysis misleading.
        """)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Monetary (Treated)",
                      f"₹{df[df['treatment']==1]['monetary'].mean():,.0f}")
        with col2:
            st.metric("Mean Monetary (Control)",
                      f"₹{df[df['treatment']==0]['monetary'].mean():,.0f}")
        with col3:
            st.metric("Mean Frequency (Treated)",
                      f"{df[df['treatment']==1]['frequency'].mean():.1f}")
        with col4:
            st.metric("Mean Frequency (Control)",
                      f"{df[df['treatment']==0]['frequency'].mean():.1f}")

        st.subheader("Evidence of Confounding")
        show_figure("01_confounding_diagnostics.png",
                     "Treatment and control groups have systematically different distributions")

        st.subheader("Ground Truth: Heterogeneous Treatment Effects")
        show_figure("01_treatment_effect_heterogeneity.png",
                     "Different user segments respond differently to the campaign")

        # Interactive propensity score explorer
        st.subheader("Propensity Score Distribution")
        if "propensity_true" in df.columns:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df[df["treatment"] == 1]["propensity_true"],
                name="Treated", opacity=0.6,
                marker_color="#e74c3c", nbinsx=50,
            ))
            fig.add_trace(go.Histogram(
                x=df[df["treatment"] == 0]["propensity_true"],
                name="Control", opacity=0.6,
                marker_color="#3498db", nbinsx=50,
            ))
            fig.update_layout(
                barmode="overlay",
                title="True Propensity Score Distributions (P(Treatment | X))",
                xaxis_title="Propensity Score",
                yaxis_title="Count",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Why Does This Matter?

        The **treated group is systematically different** from the control group:
        - Higher monetary value (they're richer customers)
        - Higher purchase frequency (they buy more often)
        - More recent activity (they're more engaged)

        A naïve comparison (`mean(treated) - mean(control)`) conflates the treatment
        effect with these pre-existing differences. **The treated group would have
        had higher outcomes even WITHOUT the discount.**
        """)

    # ═══════════════════════════════════════════════════════════════
    # TAB 2: Power Analysis
    # ═══════════════════════════════════════════════════════════════
    with tab2:
        st.header("Power Analysis: Can We Detect the Effect?")

        st.markdown("""
        Before analyzing results, we need to know if our sample is large enough
        to detect a meaningful effect. This is the question power analysis answers.
        """)

        if data["power_summary"] is not None:
            st.subheader("Sample Size Requirements")
            st.dataframe(data["power_summary"], use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            show_figure("02_power_curves.png", "Power curves for different effect sizes")
        with col2:
            show_figure("02_sample_size_requirements.png", "Required sample sizes by MDE")

        st.markdown("""
        ### Key Takeaway
        With **N = 20,000** users, we can reliably detect treatment effects as small
        as ~₹42 at 80% power and α = 0.05. Since the true ATE is ₹150, we have
        **more than sufficient statistical power** for this analysis.
        """)

    # ═══════════════════════════════════════════════════════════════
    # TAB 3: Causal Estimates
    # ═══════════════════════════════════════════════════════════════
    with tab3:
        st.header("Causal Estimation: Naïve vs. Corrected Methods")

        st.markdown("""
        This is the core demonstration: **the naïve estimate is dramatically wrong,
        while causal methods recover the true treatment effect.**
        """)

        if data["ate_comparison"] is not None:
            # Interactive comparison chart
            ate_df = data["ate_comparison"]
            fig = go.Figure()

            colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
            for i, (_, row) in enumerate(ate_df.iterrows()):
                fig.add_trace(go.Bar(
                    y=[row["Method"]],
                    x=[row["ATE (₹)"]],
                    orientation="h",
                    name=row["Method"],
                    marker_color=colors[i],
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[row["CI Upper (₹)"] - row["ATE (₹)"]],
                        arrayminus=[row["ATE (₹)"] - row["CI Lower (₹)"]],
                    ),
                    text=f"₹{row['ATE (₹)']:,.0f}",
                    textposition="outside",
                ))

            fig.add_vline(
                x=df["tau_true"].mean(), line_dash="dash",
                line_color="#e67e22", line_width=3,
                annotation_text=f"True ATE = ₹{df['tau_true'].mean():,.0f}",
            )

            fig.update_layout(
                title="ATE Estimates: Naïve vs. Causal Methods",
                xaxis_title="Estimated ATE (₹)",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(ate_df, use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            show_figure("03_propensity_overlap.png", "Propensity score overlap assessment")
        with col2:
            show_figure("03_covariate_balance.png", "Covariate balance before/after matching")

        show_figure("03_cate_distribution.png", "Individual treatment effects: estimated vs. ground truth")

        st.markdown("""
        ### What the Numbers Tell Us

        | Method | What It Does | ATE Estimate | Verdict |
        |---|---|---|---|
        | **Naïve** | Simple mean difference | ~₹350 | ❌ 2× overestimate |
        | **PSM** | Match similar users | ~₹150 | ✅ Recovers truth |
        | **IPW** | Re-weight observations | ~₹150 | ✅ Recovers truth |
        | **Causal Forest** | ML-based debiasing | ~₹150 | ✅ Recovers truth + CATEs |

        The naïve estimate includes the confounding bias (high-value users buy more
        regardless). Causal methods remove this bias.
        """)

    # ═══════════════════════════════════════════════════════════════
    # TAB 4: Uplift Model
    # ═══════════════════════════════════════════════════════════════
    with tab4:
        st.header("Uplift Modeling: Who Changes Behavior BECAUSE of the Campaign?")

        st.markdown("""
        > **Response model:** "Who will buy?" → Targets *Sure Things* (wasted spend)
        >
        > **Uplift model:** "Who will buy **because of** the campaign?" → Targets *Persuadables* (incremental value)
        """)

        col1, col2 = st.columns(2)
        with col1:
            show_figure("04_uplift_by_decile.png", "Predicted vs. actual uplift by decile")
        with col2:
            show_figure("04_cumulative_uplift.png", "Cumulative uplift curves")

        st.subheader("User Segmentation: Four Quadrants")
        col1, col2 = st.columns(2)
        with col1:
            show_figure("04_user_segmentation.png", "Segment scatter plot and distribution")
        with col2:
            show_figure("04_segment_profile.png", "Segment profile table")

        if data["segment_profile"] is not None:
            st.dataframe(data["segment_profile"], use_container_width=True, hide_index=True)

        if data["segmented"] is not None:
            st.subheader("Interactive Segment Explorer")
            seg_df = data["segmented"]
            selected_segment = st.selectbox(
                "Select a segment to explore:",
                ["All"] + sorted(seg_df["segment"].unique().tolist()),
            )
            if selected_segment != "All":
                seg_df = seg_df[seg_df["segment"] == selected_segment]

            fig = px.scatter(
                seg_df.sample(min(2000, len(seg_df)), random_state=42),
                x="baseline_prediction", y="cate_uplift",
                color="segment",
                color_discrete_map={
                    "Persuadables": "#27ae60",
                    "Sure Things": "#3498db",
                    "Lost Causes": "#95a5a6",
                    "Sleeping Dogs": "#e74c3c",
                },
                hover_data=["monetary", "frequency", "age_group"],
                title=f"Users in: {selected_segment}",
                labels={
                    "baseline_prediction": "Baseline Prediction (₹)",
                    "cate_uplift": "Uplift / CATE (₹)",
                },
                opacity=0.4,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.3)
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════
    # TAB 5: Business Impact
    # ═══════════════════════════════════════════════════════════════
    with tab5:
        st.header("💰 Business Impact: The ₹ Case for Uplift Targeting")

        if data["roi_comparison"] is not None:
            roi_df = data["roi_comparison"]

            # Headline metrics
            col1, col2, col3 = st.columns(3)
            for i, (col, color) in enumerate(zip([col1, col2, col3],
                                                   ["#95a5a6", "#e74c3c", "#27ae60"])):
                with col:
                    row = roi_df.iloc[i]
                    st.metric(
                        row["Strategy"],
                        f"₹{row['Net Profit (₹)']:,.0f}",
                        delta=f"ROI: {row['ROI (%)']:.0f}%",
                    )

            show_figure("04_roi_comparison.png", "ROI comparison across targeting strategies")

            st.dataframe(roi_df, use_container_width=True, hide_index=True)

            # Calculate uplift advantage
            if len(roi_df) >= 3:
                uplift_profit = roi_df.iloc[2]["Net Profit (₹)"]
                prob_profit = roi_df.iloc[1]["Net Profit (₹)"]
                advantage = uplift_profit - prob_profit

                st.success(
                    f"### 🎯 Switching to uplift-based targeting generates "
                    f"**₹{advantage:,.0f} more profit** than probability-based targeting "
                    f"({(advantage/abs(prob_profit))*100:+.0f}% improvement)."
                )

        # Sensitivity summary
        st.subheader("🛡️ Robustness Checks")
        if data["sensitivity_summary"] is not None:
            st.dataframe(data["sensitivity_summary"], use_container_width=True, hide_index=True)

        col1, col2 = st.columns(2)
        with col1:
            show_figure("05_placebo_test.png", "Placebo test: Is the effect real?")
        with col2:
            show_figure("05_rosenbaum_bounds.png", "Sensitivity to unmeasured confounders")

        show_figure("05_covariate_balance.png", "Pre-matching covariate balance")

        # Decision memo link
        st.divider()
        st.markdown("""
        ### 📝 Decision Memo

        A one-page stakeholder memo is available at `decision_memo.md`.
        It covers:
        1. What the campaign actually achieved (vs. what naïve analysis shows)
        2. Which customers to target (and which to stop targeting)
        3. The ₹ impact of switching strategies
        4. Recommended next steps
        """)


if __name__ == "__main__":
    main()
