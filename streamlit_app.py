import os
from pathlib import Path
import streamlit as st
import pandas as pd
import requests

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Uplift Marketing targeting Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "outputs" / "data"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

# ─── Custom CSS for Premium Design ─────────────────────────────────
st.markdown("""
<style>
    /* Styling headers and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    
    /* Custom Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 1.5rem;
        margin: 1.5rem 0;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.5rem;
        flex: 1;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.08);
        border-color: #c7d2fe;
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e1b4b 0%, #4f46e5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    .metric-value.profit-positive {
        background: linear-gradient(135deg, #065f46 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-value.profit-negative {
        background: linear-gradient(135deg, #991b1b 0%, #ef4444 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Preset buttons container */
    .preset-container {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Alerts and custom recommendations */
    .recommendation-banner {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar info ──────────────────────────────────────────────────
st.sidebar.markdown("### 🔬 About the Platform")
st.sidebar.markdown(
    "This platform optimizes marketing target budgets using **Uplift Modeling** "
    "(Conditional Average Treatment Effect - CATE). Unlike standard response models "
    "which target users likely to buy anyway (Sure Things), uplift models isolate "
    "users who buy **because of** the intervention (Persuadables)."
)

# Load Render URL from Secrets, default to localhost
api_url = st.secrets.get("RENDER_API_URL", "https://causal-inference-uplift-platform.onrender.com/docs")
st.sidebar.markdown(f"**Backend API Service:**\n`{api_url}`")

# ─── Helper function to ping backend (with timeout retry) ───────────
def check_and_wake_backend(url):
    """Pings /health up to 10 times to handle Render free-tier cold start."""
    health_url = f"{url}/health"
    try:
        # Quick check first
        r = requests.get(health_url, timeout=3)
        if r.status_code == 200:
            return True, "Online"
    except requests.exceptions.RequestException:
        pass
        
    # If failed, start waking up loop
    progress_text = "Waking up Render API backend (takes 30-60s on Render Free Plan)..."
    progress_bar = st.progress(0, text=progress_text)
    
    for i in range(12):
        progress_bar.progress((i + 1) / 12, text=f"{progress_text} (Attempt {i+1}/12)")
        try:
            r = requests.get(health_url, timeout=4)
            if r.status_code == 200:
                progress_bar.empty()
                return True, "Awake"
        except requests.exceptions.RequestException:
            pass
        import time
        time.sleep(4)
        
    progress_bar.empty()
    return False, "Offline"

# ─── App Header ───────────────────────────────────────────────────
st.markdown('<div class="main-header">🔬 Causal Inference & Uplift Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Stateless individual-level scoring & causal analytics dashboard</div>', unsafe_allow_html=True)

# Tabs configuration
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Score a Customer",
    "📊 Model Evaluation (Qini Curve)",
    "🛡️ Causal Assumptions",
    "📝 Business Case Memo"
])

# Initialize session state for preset autofills
if "recency" not in st.session_state:
    st.session_state.recency = 30.0
if "frequency" not in st.session_state:
    st.session_state.frequency = 5
if "monetary" not in st.session_state:
    st.session_state.monetary = 500.0
if "tenure_months" not in st.session_state:
    st.session_state.tenure_months = 12.0
if "channel" not in st.session_state:
    st.session_state.channel = "web"
if "age_group" not in st.session_state:
    st.session_state.age_group = "26-35"

# ═══════════════════════════════════════════════════════════════════
# TAB 1: Score a Customer
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Compute Customer Uplift (CATE)")
    st.markdown(
        "Enter customer characteristics below to score their individual treatment effect. "
        "The model will estimate the incremental sales generated if targeted, and recommend whether "
        "to dispatch a discount coupon."
    )

    # Preset profiles buttons
    st.markdown('<div class="preset-label"><b>💡 Quick Presets for Interviewers:</b></div>', unsafe_allow_html=True)
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("High-Value Loyal (Sure Thing)", use_container_width=True):
            st.session_state.recency = 5.0
            st.session_state.frequency = 18
            st.session_state.monetary = 3200.0
            st.session_state.tenure_months = 48.0
            st.session_state.channel = "web"
            st.session_state.age_group = "36-50"
            st.rerun()
            
    with col_p2:
        if st.button("Price-Sensitive Churner (Persuadable)", use_container_width=True):
            st.session_state.recency = 75.0
            st.session_state.frequency = 2
            st.session_state.monetary = 480.0
            st.session_state.tenure_months = 4.0
            st.session_state.channel = "app"
            st.session_state.age_group = "18-25"
            st.rerun()

    with col_p3:
        if st.button("Inactive Buyer (Lost Cause)", use_container_width=True):
            st.session_state.recency = 310.0
            st.session_state.frequency = 1
            st.session_state.monetary = 120.0
            st.session_state.tenure_months = 14.0
            st.session_state.channel = "store"
            st.session_state.age_group = "51+"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form
    with st.form("scoring_form"):
        col1, col2 = st.columns(2)
        with col1:
            recency = st.slider(
                "Recency (Days since last purchase)",
                min_value=1.0, max_value=365.0,
                value=float(st.session_state.recency),
                step=1.0,
                help="Days since this customer bought from us."
            )
            frequency = st.number_input(
                "Frequency (Number of past orders)",
                min_value=0, max_value=100,
                value=int(st.session_state.frequency),
                help="Total purchase count."
            )
            monetary = st.number_input(
                "Average Order Value (₹)",
                min_value=1.0, max_value=50000.0,
                value=float(st.session_state.monetary),
                step=10.0,
                help="Average amount spent per basket."
            )
            
        with col2:
            tenure_months = st.slider(
                "Customer Tenure (Months)",
                min_value=1.0, max_value=120.0,
                value=float(st.session_state.tenure_months),
                step=1.0,
                help="How long the customer has been active with us (in months)."
            )
            channel_opts = ["app", "web", "store"]
            channel = st.selectbox(
                "Preferred Channel",
                channel_opts,
                index=channel_opts.index(st.session_state.channel)
            )
            age_opts = ["18-25", "26-35", "36-50", "51+"]
            age_group = st.selectbox(
                "Age Group",
                age_opts,
                index=age_opts.index(st.session_state.age_group)
            )

        submit = st.form_submit_button("Score Customer HTE 🎯", use_container_width=True)

    if submit:
        # Check backend availability
        is_awake, wake_msg = check_and_wake_backend(api_url)
        if not is_awake:
            st.error("🚨 Render API Backend failed to wake up. Please check if the web service is active.")
        else:
            # Call API
            headers = {"Content-Type": "application/json"}
            payload = {
                "recency": recency,
                "frequency": frequency,
                "monetary": monetary,
                "tenure_months": tenure_months,
                "channel": channel,
                "age_group": age_group
            }
            
            with st.spinner("Scoring customer profile..."):
                try:
                    r = requests.post(f"{api_url}/score_uplift", json=payload, headers=headers, timeout=15)
                    if r.status_code == 200:
                        res = r.json()
                        
                        # Display HSL-styled Metric Cards
                        st.markdown("<br>", unsafe_allow_html=True)
                        uplift = res["uplift_score"]
                        profit = res["expected_net_profit"]
                        rec = res["recommendation"]
                        conf = res["confidence"]
                        exp = res["explanation"]
                        
                        profit_class = "profit-positive" if profit > 0 else "profit-negative"
                        
                        st.markdown(f"""
                        <div class="metric-container">
                            <div class="metric-card">
                                <div class="metric-label">Predicted Uplift Score</div>
                                <div class="metric-value">₹{uplift:,.2f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Expected Net Profit</div>
                                <div class="metric-value {profit_class}">₹{profit:,.2f}</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-label">Confidence</div>
                                <div class="metric-value">{conf}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Banner display
                        if rec == "Target":
                            st.success(f"**🎯 Targeting Decision:** {rec} — {exp}")
                        else:
                            st.warning(f"**⛔ Targeting Decision:** {rec} — {exp}")
                            
                    else:
                        st.error(f"Error from API ({r.status_code}): {r.text}")
                except Exception as e:
                    st.error(f"Could not connect to API at {api_url}: {str(e)}")

# ═══════════════════════════════════════════════════════════════════
# TAB 2: Model Evaluation (Qini/Uplift Curve)
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Evaluation and Profit Comparison")
    st.markdown(
        "Below is the offline model performance comparison. The cumulative uplift curve shows how much "
        "incremental value we capture as we target a larger percentage of our base (sorted by predicted uplift)."
    )

    col_img, col_tbl = st.columns([3, 2])
    with col_img:
        st.markdown("**Cumulative Uplift (Qini-style) Performance:**")
        img_path = FIGURES_DIR / "04_cumulative_uplift.png"
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        else:
            st.info("Performance curve image not found. Run training script first.")

    with col_tbl:
        st.markdown("**Targeting Strategy Economics (Precomputed ROI):**")
        table_path = TABLES_DIR / "04_roi_comparison.csv"
        if table_path.exists():
            roi_df = pd.read_csv(table_path)
            st.dataframe(roi_df, use_container_width=True, hide_index=True)
            
            # Print profit summary card
            if len(roi_df) >= 3:
                profit_uplift = roi_df.iloc[2]["Net Profit (₹)"]
                profit_random = roi_df.iloc[0]["Net Profit (₹)"]
                profit_prob = roi_df.iloc[1]["Net Profit (₹)"]
                
                added_value = profit_uplift - profit_prob
                st.success(
                    f"### 📈 Profit Advantage:\n"
                    f"Switching from standard **Probability-Based** targeting (targets likely buyers) "
                    f"to **Uplift-Based** targeting (targets persuadables) increases net profit by "
                    f"**₹{added_value:,.2f}** per 20,000 customers."
                )
        else:
            st.info("ROI comparison table not found. Run training script first.")

# ═══════════════════════════════════════════════════════════════════
# TAB 3: Causal Assumptions
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Debiasing & Robustness Diagnostics")
    st.markdown(
        "Causal inference claims depend on statistical assumptions. In observational data, "
        "pre-treatment covariates determine who receives the treatment. Below are the covariate balance checks "
        "and overlap diagnostic plots verifying that our estimators successfully debiased the data."
    )

    # 1. Overlap & Love Plot
    col_diag1, col_diag2 = st.columns(2)
    with col_diag1:
        st.markdown("**1. Common Support / Propensity Score Overlap:**")
        overlap_img = FIGURES_DIR / "03_propensity_overlap.png"
        if overlap_img.exists():
            st.image(str(overlap_img), use_container_width=True)
        else:
            st.info("Overlap plot not found.")
        st.caption(
            "**Assumption Check (Overlap/Positivity):** Every customer profile must have a non-zero probability "
            "of being treated or controlled. If treated and control propensity distributions do not overlap, "
            "we cannot make causal comparisons. The chart shows strong common support across groups."
        )

    with col_diag2:
        st.markdown("**2. Covariate Balance (Love Plot):**")
        love_img = FIGURES_DIR / "03_covariate_balance.png"
        if love_img.exists():
            st.image(str(love_img), use_container_width=True)
        else:
            st.info("Love plot not found.")
        st.caption(
            "**Assumption Check (Unconfoundedness):** In observational studies, treated and control groups are imbalanced "
            "on confounders. Nearest-neighbor Propensity Score Matching (PSM) balances the groups. Standardized Mean "
            "Difference (SMD) drops below the 0.1 threshold post-matching, verifying that selection bias is removed."
        )

    st.divider()

    # 2. Placebo & Rosenbaum bounds
    col_diag3, col_diag4 = st.columns(2)
    with col_diag3:
        st.markdown("**3. Placebo Treatment Refutation Test:**")
        placebo_img = FIGURES_DIR / "05_placebo_test.png"
        if placebo_img.exists():
            st.image(str(placebo_img), use_container_width=True)
        else:
            st.info("Placebo test plot not found.")
        st.caption(
            "**Robustness Check:** We shuffle treatment labels randomly (breaking any real causal effect). "
            "Our IPW estimator should report an ATE near ₹0. As shown, the placebo estimates center around 0, "
            "and our real estimate (₹168) sits far in the tail, confirming our estimate is not random noise."
        )

    with col_diag4:
        st.markdown("**4. Rosenbaum Sensitivity Bounds:**")
        bounds_img = FIGURES_DIR / "05_rosenbaum_bounds.png"
        if bounds_img.exists():
            st.image(str(bounds_img), use_container_width=True)
        else:
            st.info("Sensitivity bounds plot not found.")
        st.caption(
            "**Robustness Check (Hidden Bias):** How strong would an unobserved confounder have to be to change the odds of "
            "treatment and nullify our results? The Wilcoxon sign-rank p-value upper bound remains significant ($p < 0.05$) "
            "up to $\Gamma = 3.00$, meaning a hidden confounder would need to triple the odds of treatment to invalidate "
            "our causal conclusions."
        )

# ═══════════════════════════════════════════════════════════════════
# TAB 4: Business Case Memo
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Stakeholder Recommendation Memo")
    st.markdown(
        "Below is the one-page strategic decision memo summarizing the campaign optimization findings "
        "and recommended deployment plan."
    )
    
    memo_path = PROJECT_ROOT / "decision_memo.md"
    if memo_path.exists():
        with open(memo_path, "r", encoding="utf-8") as f:
            memo_content = f.read()
        st.markdown(memo_content)
    else:
        st.info("Stakeholder memo file `decision_memo.md` not found in workspace.")
