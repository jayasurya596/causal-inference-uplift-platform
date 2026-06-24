"""
01_data_generation.py — Synthetic Data with Realistic Confounders
=================================================================

Simulates 20,000 e-commerce users who were part of a promotional campaign.
The treatment (discount email) was NOT randomly assigned — high-value,
frequent, recent buyers were far more likely to receive it. This creates
the exact kind of confounding that breaks naïve A/B comparisons.

Ground truth is embedded so we can validate causal estimators.

Key confounding mechanism:
    logit(P(T=1)) = -1 + 0.8·log(monetary) + 0.5·frequency - 0.3·recency/100

This means the treated group is systematically richer, more frequent, and
more recent — so their higher outcomes are partly because they were better
customers, NOT because of the treatment.
"""

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid function
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import config as cfg

warnings.filterwarnings("ignore")


def generate_user_features(n: int, seed: int) -> pd.DataFrame:
    """
    Generate baseline user features BEFORE treatment assignment.

    All features are pre-treatment covariates. Using post-treatment
    features would introduce collider bias.

    Parameters
    ----------
    n : int
        Number of users to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        User-level feature matrix.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        # ── Continuous confounders (affect BOTH treatment and outcome) ──
        "recency": rng.uniform(*cfg.RECENCY_RANGE, size=n),
        "frequency": rng.poisson(cfg.FREQUENCY_LAMBDA, size=n),
        "monetary": rng.lognormal(cfg.MONETARY_MU, cfg.MONETARY_SIGMA, size=n),
        "tenure_months": rng.uniform(*cfg.TENURE_RANGE, size=n),

        # ── Categorical covariates (affect outcome, may affect treatment) ──
        "channel": rng.choice(
            list(cfg.CHANNEL_PROBS.keys()),
            size=n,
            p=list(cfg.CHANNEL_PROBS.values()),
        ),
        "age_group": rng.choice(
            list(cfg.AGE_GROUP_PROBS.keys()),
            size=n,
            p=list(cfg.AGE_GROUP_PROBS.values()),
        ),
    })

    # User ID
    df.insert(0, "user_id", range(1, n + 1))

    return df


def assign_treatment(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Assign treatment with CONFOUNDED selection (not random!).

    The treatment probability depends on monetary value, frequency,
    and recency — the same variables that drive the outcome. This is
    the core of the selection bias problem.

    Assumption being violated by design:
        In a true RCT, P(T|X) = P(T) for all X. Here, P(T|X) varies
        dramatically, which is why naïve comparisons fail.

    Parameters
    ----------
    df : pd.DataFrame
        User features.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'propensity_true' and 'treatment' columns added.
    """
    rng = np.random.default_rng(seed)

    # True propensity score (known because we're generating the data)
    log_monetary = np.log(df["monetary"].values)
    logit_p = (
        cfg.TREAT_INTERCEPT
        + cfg.TREAT_MONETARY * log_monetary
        + cfg.TREAT_FREQUENCY * df["frequency"].values
        + cfg.TREAT_RECENCY * (df["recency"].values / 100)
    )
    propensity = expit(logit_p)

    df = df.copy()
    df["propensity_true"] = propensity
    df["treatment"] = rng.binomial(1, propensity).astype(int)

    return df


def generate_outcome(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """
    Generate the outcome (purchase amount) with heterogeneous treatment effects.

    The outcome depends on the SAME confounders that drove treatment
    assignment — this is what creates the bias. The treatment effect
    τ(X) varies by user segment.

    Ground truth τ(X):
        τ = 150 + 50·I(age=18-25) - 30·I(freq>8) + 20·I(channel=app)

    So:
        - Young app users:  τ = 150 + 50 + 20 = 220  (persuadables!)
        - Old frequent buyers: τ = 150 - 30 = 120    (sure things)
        - Average user:      τ = 150                  (moderate uplift)

    Parameters
    ----------
    df : pd.DataFrame
        User features with treatment assignment.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'tau_true', 'purchase_amount', and 'purchased' added.
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # ── Heterogeneous treatment effect ──
    tau = np.full(n, cfg.TRUE_ATE_BASE)
    tau += cfg.TAU_YOUNG * (df["age_group"].values == "18-25").astype(float)
    tau += cfg.TAU_HIGH_FREQ * (df["frequency"].values > 8).astype(float)
    tau += cfg.TAU_APP * (df["channel"].values == "app").astype(float)

    # ── Potential outcomes framework ──
    # Y(0) = baseline outcome (what would happen without treatment)
    baseline = (
        cfg.OUTCOME_BASE
        + cfg.OUTCOME_MONETARY * np.log(df["monetary"].values)
        + cfg.OUTCOME_FREQUENCY * df["frequency"].values
        + cfg.OUTCOME_RECENCY * (df["recency"].values / 100)
    )
    noise = rng.normal(0, cfg.OUTCOME_NOISE_STD, size=n)

    # Y(0) and Y(1) — fundamental problem: we only observe one
    y0 = baseline + noise
    y1 = baseline + tau + noise  # Same noise for consistency (SUTVA)

    # Observed outcome: Y = T·Y(1) + (1-T)·Y(0)
    treatment = df["treatment"].values
    y_observed = treatment * y1 + (1 - treatment) * y0

    df = df.copy()
    df["tau_true"] = tau                      # Ground truth (unobservable IRL)
    df["y0_true"] = y0                        # Potential outcome under control
    df["y1_true"] = y1                        # Potential outcome under treatment
    df["purchase_amount"] = y_observed        # What we actually observe

    # Binary outcome: purchased if above median of observed outcomes
    threshold = np.percentile(y_observed, cfg.PURCHASE_THRESHOLD_PERCENTILE)
    df["purchased"] = (y_observed > threshold).astype(int)

    return df


def create_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create numeric feature matrix for modeling.

    One-hot encodes categorical variables and selects only pre-treatment
    features. This is critical — including post-treatment variables
    would introduce collider bias and break causal identification.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.

    Returns
    -------
    pd.DataFrame
        Numeric feature matrix suitable for sklearn/econml/causalml.
    """
    # One-hot encode categoricals
    df_features = pd.get_dummies(
        df[["recency", "frequency", "monetary", "tenure_months",
            "channel", "age_group"]],
        columns=["channel", "age_group"],
        drop_first=False,  # Keep all dummies for interpretability
        dtype=float,
    )
    return df_features


def plot_confounding_diagnostics(df: pd.DataFrame) -> None:
    """
    Visualize the confounding: show that treated and control groups
    have systematically different covariate distributions.

    This is the visual proof that naïve comparison will be biased.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(
        "Covariate Imbalance: Treatment vs. Control\n"
        "(Evidence of Confounding — Groups Are NOT Comparable)",
        fontsize=14, fontweight="bold", y=1.02,
    )

    # Continuous confounders
    for ax, col, label in zip(
        axes[0],
        ["monetary", "frequency", "recency"],
        ["Avg Order Value (₹)", "Purchase Frequency", "Recency (days)"],
    ):
        for t_val, t_label, color in [(1, "Treated", "#e74c3c"), (0, "Control", "#3498db")]:
            subset = df[df["treatment"] == t_val][col]
            ax.hist(subset, bins=40, alpha=0.5, label=t_label, color=color, density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend()
        # Add mean lines
        ax.axvline(df[df["treatment"] == 1][col].mean(), color="#e74c3c",
                    linestyle="--", linewidth=2, alpha=0.8)
        ax.axvline(df[df["treatment"] == 0][col].mean(), color="#3498db",
                    linestyle="--", linewidth=2, alpha=0.8)

    # Propensity score distribution
    ax = axes[1][0]
    for t_val, t_label, color in [(1, "Treated", "#e74c3c"), (0, "Control", "#3498db")]:
        subset = df[df["treatment"] == t_val]["propensity_true"]
        ax.hist(subset, bins=40, alpha=0.5, label=t_label, color=color, density=True)
    ax.set_xlabel("True Propensity Score P(T=1|X)")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Overlap")
    ax.legend()

    # Treatment rate by age group
    ax = axes[1][1]
    treat_rate = df.groupby("age_group")["treatment"].mean().sort_values()
    bars = ax.bar(treat_rate.index, treat_rate.values, color="#9b59b6", alpha=0.8)
    ax.set_ylabel("Treatment Rate")
    ax.set_title("Treatment Rate by Age Group")
    ax.axhline(df["treatment"].mean(), color="gray", linestyle="--", label="Overall")
    ax.legend()
    for bar, val in zip(bars, treat_rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=10)

    # Treatment rate by channel
    ax = axes[1][2]
    treat_rate = df.groupby("channel")["treatment"].mean().sort_values()
    bars = ax.bar(treat_rate.index, treat_rate.values, color="#e67e22", alpha=0.8)
    ax.set_ylabel("Treatment Rate")
    ax.set_title("Treatment Rate by Channel")
    ax.axhline(df["treatment"].mean(), color="gray", linestyle="--", label="Overall")
    ax.legend()
    for bar, val in zip(bars, treat_rate.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.1%}", ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "01_confounding_diagnostics.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved confounding diagnostics → {cfg.FIGURES_DIR / '01_confounding_diagnostics.png'}")


def plot_treatment_effect_heterogeneity(df: pd.DataFrame) -> None:
    """
    Visualize the ground-truth heterogeneous treatment effects.
    This plot shows WHY different users respond differently.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Ground Truth: Heterogeneous Treatment Effects τ(X)",
                 fontsize=14, fontweight="bold")

    # τ distribution
    ax = axes[0]
    ax.hist(df["tau_true"], bins=30, color="#27ae60", alpha=0.7, edgecolor="white")
    ax.axvline(df["tau_true"].mean(), color="#e74c3c", linestyle="--",
               linewidth=2, label=f"Mean τ = ₹{df['tau_true'].mean():.0f}")
    ax.set_xlabel("Individual Treatment Effect τ(X) [₹]")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of True CATEs")
    ax.legend()

    # τ by age group
    ax = axes[1]
    tau_by_age = df.groupby("age_group")["tau_true"].mean().sort_values(ascending=False)
    bars = ax.bar(tau_by_age.index, tau_by_age.values,
                  color=["#27ae60" if v > 150 else "#3498db" for v in tau_by_age.values],
                  alpha=0.8)
    ax.set_ylabel("Mean τ(X) [₹]")
    ax.set_title("Treatment Effect by Age Group")
    ax.axhline(cfg.TRUE_ATE_BASE, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, tau_by_age.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"₹{val:.0f}", ha="center", fontsize=11, fontweight="bold")

    # τ by frequency bucket
    ax = axes[2]
    df_temp = df.copy()
    df_temp["freq_bucket"] = pd.cut(df_temp["frequency"], bins=[0, 3, 5, 8, 100],
                                     labels=["1-3", "4-5", "6-8", "9+"])
    tau_by_freq = df_temp.groupby("freq_bucket", observed=True)["tau_true"].mean()
    bars = ax.bar(tau_by_freq.index, tau_by_freq.values,
                  color=["#27ae60" if v > 150 else "#e67e22" if v > 130 else "#e74c3c"
                         for v in tau_by_freq.values],
                  alpha=0.8)
    ax.set_ylabel("Mean τ(X) [₹]")
    ax.set_title("Treatment Effect by Purchase Frequency")
    ax.set_xlabel("Frequency Bucket")
    for bar, val in zip(bars, tau_by_freq.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"₹{val:.0f}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "01_treatment_effect_heterogeneity.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved treatment effect heterogeneity → {cfg.FIGURES_DIR / '01_treatment_effect_heterogeneity.png'}")


def print_data_summary(df: pd.DataFrame) -> None:
    """Print a comprehensive summary of the generated dataset."""
    print("\n" + "=" * 70)
    print("  SYNTHETIC DATASET SUMMARY")
    print("=" * 70)

    n_treated = df["treatment"].sum()
    n_control = len(df) - n_treated

    print(f"\n  Total users:       {len(df):,}")
    print(f"  Treated:           {n_treated:,} ({n_treated/len(df):.1%})")
    print(f"  Control:           {n_control:,} ({n_control/len(df):.1%})")

    print(f"\n  ── Confounding Evidence ──")
    print(f"  Mean monetary (Treated):  ₹{df[df['treatment']==1]['monetary'].mean():,.0f}")
    print(f"  Mean monetary (Control):  ₹{df[df['treatment']==0]['monetary'].mean():,.0f}")
    print(f"  Mean frequency (Treated): {df[df['treatment']==1]['frequency'].mean():.1f}")
    print(f"  Mean frequency (Control): {df[df['treatment']==0]['frequency'].mean():.1f}")
    print(f"  Mean recency (Treated):   {df[df['treatment']==1]['recency'].mean():.0f} days")
    print(f"  Mean recency (Control):   {df[df['treatment']==0]['recency'].mean():.0f} days")

    # Naïve comparison (the WRONG answer)
    naive_ate = (
        df[df["treatment"] == 1]["purchase_amount"].mean()
        - df[df["treatment"] == 0]["purchase_amount"].mean()
    )
    true_ate = df["tau_true"].mean()

    print(f"\n  ── Treatment Effect Estimates ──")
    print(f"  Naïve ATE (BIASED):   ₹{naive_ate:,.0f}  ← WRONG due to confounding!")
    print(f"  True ATE (ground truth): ₹{true_ate:,.0f}  ← What we need to recover")
    print(f"  Bias:                  ₹{naive_ate - true_ate:,.0f}  ({(naive_ate/true_ate - 1)*100:+.0f}%)")

    print(f"\n  ── Ground Truth τ(X) Distribution ──")
    print(f"  Min τ:   ₹{df['tau_true'].min():.0f}")
    print(f"  Mean τ:  ₹{df['tau_true'].mean():.0f}")
    print(f"  Max τ:   ₹{df['tau_true'].max():.0f}")
    print(f"  Std τ:   ₹{df['tau_true'].std():.0f}")

    print("\n" + "=" * 70)


def main():
    """Run the full data generation pipeline."""
    print("\n🔧 STEP 1: Generating Synthetic Dataset")
    print("-" * 50)

    # Step 1: Generate features
    print("  Generating user features...")
    df = generate_user_features(cfg.N_USERS, cfg.RANDOM_SEED)

    # Step 2: Assign treatment (with confounding!)
    print("  Assigning treatment (confounded — NOT random)...")
    df = assign_treatment(df, cfg.RANDOM_SEED + 1)

    # Step 3: Generate outcomes
    print("  Generating outcomes with heterogeneous effects...")
    df = generate_outcome(df, cfg.RANDOM_SEED + 2)

    # Step 4: Create analysis features
    print("  Creating analysis feature matrix...")
    X = create_analysis_features(df)

    # Step 5: Save
    df.to_csv(cfg.DATA_DIR / "synthetic_data.csv", index=False)
    X.to_csv(cfg.DATA_DIR / "feature_matrix.csv", index=False)
    print(f"  ✓ Saved dataset → {cfg.DATA_DIR / 'synthetic_data.csv'}")
    print(f"  ✓ Saved features → {cfg.DATA_DIR / 'feature_matrix.csv'}")

    # Step 6: Summary & plots
    print_data_summary(df)
    plot_confounding_diagnostics(df)
    plot_treatment_effect_heterogeneity(df)

    return df, X


if __name__ == "__main__":
    df, X = main()
