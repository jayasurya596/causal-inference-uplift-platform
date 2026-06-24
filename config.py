"""
config.py — Central Configuration for the Causal Inference Platform
====================================================================

All hyperparameters, constants, and paths are defined here so that every
module in the pipeline draws from a single source of truth. Changing a
value here propagates everywhere.
"""

import os
from pathlib import Path

# ─── Reproducibility ────────────────────────────────────────────────
RANDOM_SEED = 42

# ─── Paths ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
DATA_DIR = OUTPUT_DIR / "data"
MODELS_DIR = OUTPUT_DIR / "models"

# Create directories
for d in [FIGURES_DIR, TABLES_DIR, DATA_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Data Generation Parameters ────────────────────────────────────
N_USERS = 20_000  # Total users in the simulated dataset

# Feature distributions
RECENCY_RANGE = (1, 365)           # Days since last purchase
FREQUENCY_LAMBDA = 5               # Poisson λ for purchase frequency
MONETARY_MU = 7.0                  # LogNormal μ for avg order value (₹)
MONETARY_SIGMA = 0.8               # LogNormal σ
TENURE_RANGE = (1, 60)             # Months of customer tenure
CHANNEL_PROBS = {                  # Channel preference distribution
    "app": 0.40,
    "web": 0.35,
    "store": 0.25,
}
AGE_GROUP_PROBS = {                # Age group distribution
    "18-25": 0.25,
    "26-35": 0.35,
    "36-50": 0.25,
    "51+": 0.15,
}

# Treatment assignment (logistic model — NOT random)
# logit(P(T=1)) = TREAT_INTERCEPT + TREAT_MONETARY * log(monetary)
#                + TREAT_FREQUENCY * frequency
#                + TREAT_RECENCY * (recency / 100)
TREAT_INTERCEPT = -7.5
TREAT_MONETARY = 0.8               # High-value → more likely treated
TREAT_FREQUENCY = 0.5              # Frequent buyers → more likely treated
TREAT_RECENCY = -0.3               # Recent buyers → more likely treated

# Outcome model (purchase amount in ₹)
# Y = BASE + OUTCOME_MONETARY * log(monetary) + OUTCOME_FREQUENCY * frequency
#   + OUTCOME_RECENCY * (recency / 100) + τ(X) * T + ε
OUTCOME_BASE = 200.0
OUTCOME_MONETARY = 80.0
OUTCOME_FREQUENCY = 30.0
OUTCOME_RECENCY = -20.0
OUTCOME_NOISE_STD = 100.0          # ε ~ N(0, σ²)

# Heterogeneous treatment effect τ(X)
# τ(X) = TRUE_ATE_BASE + τ_YOUNG * I(age=18-25)
#       + τ_HIGH_FREQ * I(freq > 8) + τ_APP * I(channel=app)
TRUE_ATE_BASE = 150.0              # Base treatment effect (₹)
TAU_YOUNG = 50.0                   # Young users respond more
TAU_HIGH_FREQ = -30.0              # Very frequent buyers respond less (sure things)
TAU_APP = 20.0                     # App users respond more

# Binary outcome
PURCHASE_THRESHOLD_PERCENTILE = 50  # Top X% of Y → purchased = 1

# ─── Treatment Economics ────────────────────────────────────────────
TREATMENT_COST_PER_USER = 50.0     # ₹ cost of sending discount email
DISCOUNT_AMOUNT = 200.0            # ₹ face value of discount coupon
AVERAGE_MARGIN_RATE = 0.30         # 30% margin on purchases

# ─── Power Analysis ────────────────────────────────────────────────
ALPHA = 0.05                       # Significance level
POWER_TARGET = 0.80                # Desired statistical power
MDE_TARGET = 50.0                  # Minimum detectable effect (₹)

# ─── Causal Estimation ─────────────────────────────────────────────
PSM_CALIPER = 0.2                  # Caliper in units of propensity SD
IPW_TRIM_PERCENTILES = (1, 99)     # Trim extreme propensity weights
N_BOOTSTRAP = 500                  # Bootstrap iterations for CIs

# ─── Uplift Modeling ────────────────────────────────────────────────
UPLIFT_TARGET_FRACTION = 0.50      # Target top 50% by uplift score
N_DECILES = 10                     # Decile bins for uplift evaluation

# ─── Sensitivity Analysis ──────────────────────────────────────────
N_PLACEBO_ITERATIONS = 500         # Placebo test permutations
GAMMA_RANGE = (1.0, 3.0)           # Rosenbaum Γ sweep range
GAMMA_STEPS = 21                   # Number of Γ values to test

# ─── Visualization ──────────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_SIZE = (12, 7)
COLOR_PALETTE = {
    "naive": "#e74c3c",            # Red — danger / wrong
    "psm": "#3498db",              # Blue
    "ipw": "#2ecc71",              # Green
    "causal_forest": "#9b59b6",    # Purple
    "true_ate": "#e67e22",         # Orange — ground truth
    "persuadables": "#27ae60",
    "sure_things": "#3498db",
    "lost_causes": "#95a5a6",
    "sleeping_dogs": "#e74c3c",
}
