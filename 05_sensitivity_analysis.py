"""
05_sensitivity_analysis.py — Robustness & Sensitivity Checks
==============================================================

Causal estimates are only as good as their assumptions. This module
stress-tests those assumptions:

1. PLACEBO TEST: If we randomly reassign treatment (breaking the
   true causal relationship), does our estimator correctly return ≈0?
   If yes → the method isn't just picking up noise.

2. ROSENBAUM BOUNDS: How strong would an unmeasured confounder need
   to be to nullify our causal estimate? If Γ=2 still gives a
   significant result, our estimate is robust to any confounder that
   doubles the odds of treatment.

3. COVARIATE BALANCE: After matching/weighting, are the treatment and
   control groups actually comparable on all observed covariates?

These checks don't PROVE causality (nothing can with observational data),
but they quantify how much you'd have to believe in hidden confounders
to dismiss the result.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import config as cfg

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# 1. PLACEBO TEST (Randomization Inference)
# ═══════════════════════════════════════════════════════════════════

def placebo_test(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    n_iterations: int = cfg.N_PLACEBO_ITERATIONS,
) -> dict:
    """
    Placebo test via randomization inference.

    Procedure:
    1. Randomly shuffle the treatment labels (breaking the true DGP).
    2. Re-estimate the ATE using IPW on the shuffled treatment.
    3. Repeat N times to get the null distribution.
    4. Compare the real ATE to this distribution.

    If our method is valid:
    - Placebo ATEs should be centered around 0 (no effect when there
      IS no effect).
    - The real ATE should be far outside the placebo distribution.

    The p-value from this test is:
        p = fraction of |placebo ATE| ≥ |real ATE|
    This is a non-parametric, assumption-free p-value.
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED)

    # Real ATE via IPW
    real_ate = _ipw_estimate(X, y, treatment)

    # Placebo distribution
    print(f"    Running {n_iterations} placebo iterations...")
    placebo_ates = []
    for i in range(n_iterations):
        # Shuffle treatment labels (breaks causal link)
        shuffled_treatment = rng.permutation(treatment)
        placebo_ate = _ipw_estimate(X, y, shuffled_treatment)
        placebo_ates.append(placebo_ate)

        if (i + 1) % 100 == 0:
            print(f"      ... {i + 1}/{n_iterations} done")

    placebo_ates = np.array(placebo_ates)

    # Placebo p-value
    p_value = np.mean(np.abs(placebo_ates) >= np.abs(real_ate))

    result = {
        "real_ate": real_ate,
        "placebo_ates": placebo_ates,
        "placebo_mean": placebo_ates.mean(),
        "placebo_std": placebo_ates.std(),
        "p_value": p_value,
    }

    print(f"\n    Real ATE:       ₹{real_ate:,.0f}")
    print(f"    Placebo mean:   ₹{placebo_ates.mean():,.1f} (should be ≈ 0)")
    print(f"    Placebo std:    ₹{placebo_ates.std():,.1f}")
    print(f"    Placebo p-value: {p_value:.4f}")
    print(f"    → {'✓ PASSED' if p_value < 0.05 else '✗ FAILED'}: "
          f"Real effect is {'distinguishable' if p_value < 0.05 else 'NOT distinguishable'} "
          f"from placebo")

    return result


def _ipw_estimate(X, y, treatment):
    """Quick IPW estimate for internal use in placebo test."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500, random_state=cfg.RANDOM_SEED)
    model.fit(X_scaled, treatment)
    ps = model.predict_proba(X_scaled)[:, 1]
    ps = np.clip(ps, 0.01, 0.99)

    weights = np.where(treatment == 1, 1.0 / ps, 1.0 / (1.0 - ps))

    w_t = weights[treatment == 1]
    w_c = weights[treatment == 0]
    w_t = w_t / w_t.sum()
    w_c = w_c / w_c.sum()

    ate = np.sum(w_t * y[treatment == 1]) - np.sum(w_c * y[treatment == 0])
    return ate


def plot_placebo_test(placebo_result: dict) -> None:
    """
    Plot the placebo distribution with the real ATE marked.

    A good result: the real ATE is far in the tail, well outside
    the range of placebo estimates.
    """
    fig, ax = plt.subplots(figsize=cfg.FIGURE_SIZE)

    # Histogram of placebo ATEs
    ax.hist(
        placebo_result["placebo_ates"], bins=40,
        color="#3498db", alpha=0.6, edgecolor="white",
        label="Placebo ATEs (null distribution)",
        density=True,
    )

    # Real ATE
    ax.axvline(
        placebo_result["real_ate"], color="#e74c3c",
        linewidth=3, linestyle="-",
        label=f"Real ATE = ₹{placebo_result['real_ate']:,.0f}",
    )

    # Zero line
    ax.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)

    # Annotate p-value
    p = placebo_result["p_value"]
    ax.text(
        0.02, 0.95,
        f"Placebo p-value: {p:.4f}\n"
        f"{'✓ Robust: effect is real' if p < 0.05 else '⚠ Not robust'}",
        transform=ax.transAxes, fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#d5f5e3" if p < 0.05 else "#fadbd8",
                  alpha=0.8),
    )

    ax.set_xlabel("Estimated ATE (₹)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(
        "Placebo Test: Is the Treatment Effect Real?\n"
        f"({cfg.N_PLACEBO_ITERATIONS} random treatment reassignments)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "05_placebo_test.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved placebo test → {cfg.FIGURES_DIR / '05_placebo_test.png'}")


# ═══════════════════════════════════════════════════════════════════
# 2. ROSENBAUM BOUNDS (Sensitivity to Unmeasured Confounding)
# ═══════════════════════════════════════════════════════════════════

def rosenbaum_bounds(
    y_treated: np.ndarray,
    y_control: np.ndarray,
    gamma_range: tuple = cfg.GAMMA_RANGE,
    gamma_steps: int = cfg.GAMMA_STEPS,
) -> dict:
    """
    Rosenbaum bounds for matched pairs.

    Idea: Even after matching on observed covariates, there might be an
    UNMEASURED confounder U that affects treatment assignment. Gamma (Γ)
    represents the maximum odds ratio by which U can change P(T=1).

    For each Γ, we compute the worst-case p-value assuming a confounder
    of strength Γ exists. The key question:

        "How large would Γ need to be to make our result non-significant?"

    If Γ > 2: "An unmeasured confounder would need to DOUBLE the odds of
    treatment to explain away our result." → Fairly robust.

    If Γ < 1.5: "Even a moderate confounder could nullify our result."
    → Fragile estimate.

    Implementation uses the Wilcoxon signed-rank test on matched pairs
    with modified treatment probabilities per Rosenbaum (2002).
    """
    # Compute pair-level differences
    n_pairs = min(len(y_treated), len(y_control))
    differences = y_treated[:n_pairs] - y_control[:n_pairs]

    # Sort by absolute difference
    abs_diffs = np.abs(differences)
    ranks = stats.rankdata(abs_diffs)
    signs = np.sign(differences)

    # The test statistic under no hidden bias
    T_obs = np.sum(ranks[signs > 0])

    gammas = np.linspace(gamma_range[0], gamma_range[1], gamma_steps)
    results = []

    for gamma in gammas:
        # Under sensitivity parameter Γ, the probability that the
        # treated unit in a pair has the larger potential outcome is
        # bounded between 1/(1+Γ) and Γ/(1+Γ)
        p_upper = gamma / (1 + gamma)

        # Expected value and variance under worst case
        E_T = np.sum(ranks * p_upper)
        Var_T = np.sum(ranks ** 2 * p_upper * (1 - p_upper))

        # Normal approximation for upper bound p-value
        if Var_T > 0:
            z = (T_obs - E_T) / np.sqrt(Var_T)
            p_upper_bound = 1 - stats.norm.cdf(z)
        else:
            p_upper_bound = 0.5

        results.append({
            "gamma": gamma,
            "p_upper_bound": p_upper_bound,
            "significant_at_005": p_upper_bound < 0.05,
            "significant_at_010": p_upper_bound < 0.10,
        })

    results_df = pd.DataFrame(results)

    # Find critical Γ (smallest Γ where result becomes non-significant)
    non_sig = results_df[~results_df["significant_at_005"]]
    critical_gamma = non_sig["gamma"].min() if len(non_sig) > 0 else gamma_range[1]

    result = {
        "results_df": results_df,
        "critical_gamma": critical_gamma,
        "T_obs": T_obs,
    }

    print(f"\n    Critical Γ (α=0.05): {critical_gamma:.2f}")
    print(f"    → An unmeasured confounder would need to change treatment odds by "
          f"{critical_gamma:.1f}× to nullify the result.")
    if critical_gamma >= 2.0:
        print(f"    ✓ ROBUST: Result survives even strong unmeasured confounding.")
    elif critical_gamma >= 1.5:
        print(f"    ⚠ MODERATE: Result is somewhat sensitive to unmeasured confounding.")
    else:
        print(f"    ✗ FRAGILE: Result is very sensitive to unmeasured confounding.")

    return result


def plot_rosenbaum_bounds(rb_result: dict) -> None:
    """Plot Rosenbaum bounds: Γ vs. upper-bound p-value."""
    df = rb_result["results_df"]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["gamma"], df["p_upper_bound"],
            color="#9b59b6", linewidth=2.5, marker="o", markersize=5)

    # Significance thresholds
    ax.axhline(0.05, color="#e74c3c", linestyle="--", linewidth=2,
               label="α = 0.05")
    ax.axhline(0.10, color="#e67e22", linestyle="--", linewidth=1.5,
               label="α = 0.10")

    # Critical Γ
    ax.axvline(rb_result["critical_gamma"], color="#27ae60", linestyle=":",
               linewidth=2, label=f"Critical Γ = {rb_result['critical_gamma']:.2f}")

    # Shade regions
    ax.fill_between(df["gamma"], 0, df["p_upper_bound"],
                     where=df["p_upper_bound"] < 0.05,
                     color="#27ae60", alpha=0.15, label="Significant region")

    ax.set_xlabel("Sensitivity Parameter Γ", fontsize=12)
    ax.set_ylabel("Upper Bound p-value", fontsize=12)
    ax.set_title(
        "Rosenbaum Sensitivity Analysis\n"
        "How Robust Is Our Causal Estimate to Unmeasured Confounding?",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(0.5, df["p_upper_bound"].max() * 1.1))
    ax.grid(alpha=0.3)

    # Interpretation box
    gamma_c = rb_result["critical_gamma"]
    if gamma_c >= 2.0:
        interp = f"✓ ROBUST\nA confounder would need to change\ntreatment odds by {gamma_c:.1f}×"
        bg_color = "#d5f5e3"
    elif gamma_c >= 1.5:
        interp = f"⚠ MODERATE\nGamma = {gamma_c:.1f}"
        bg_color = "#fdebd0"
    else:
        interp = f"✗ FRAGILE\nGamma = {gamma_c:.1f}"
        bg_color = "#fadbd8"

    ax.text(
        0.98, 0.95, interp, transform=ax.transAxes,
        fontsize=11, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=bg_color, alpha=0.9),
    )

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "05_rosenbaum_bounds.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved Rosenbaum bounds → {cfg.FIGURES_DIR / '05_rosenbaum_bounds.png'}")


# ═══════════════════════════════════════════════════════════════════
# 3. COVARIATE BALANCE ASSESSMENT
# ═══════════════════════════════════════════════════════════════════

def compute_smd(x, treatment):
    """Compute Standardized Mean Difference for a single covariate."""
    x_t = x[treatment == 1]
    x_c = x[treatment == 0]
    pooled_std = np.sqrt((x_t.var() + x_c.var()) / 2)
    if pooled_std == 0:
        return 0
    return (x_t.mean() - x_c.mean()) / pooled_std


def compute_variance_ratio(x, treatment):
    """Compute variance ratio (treated/control)."""
    var_t = x[treatment == 1].var()
    var_c = x[treatment == 0].var()
    if var_c == 0:
        return np.inf
    return var_t / var_c


def covariate_balance_assessment(
    X: pd.DataFrame,
    treatment: np.ndarray,
) -> dict:
    """
    Comprehensive covariate balance assessment.

    Metrics:
    - SMD (Standardized Mean Difference): |SMD| < 0.1 is acceptable.
    - Variance Ratio: Should be between 0.5 and 2.0.

    We compute these for the ORIGINAL (unmatched) data to quantify
    the severity of confounding.
    """
    balance_metrics = []
    for col in X.columns:
        x = X[col].values
        smd = compute_smd(x, treatment)
        vr = compute_variance_ratio(x, treatment)
        balance_metrics.append({
            "Covariate": col,
            "SMD": smd,
            "|SMD|": abs(smd),
            "Variance Ratio": vr,
            "Balance OK (|SMD|<0.1)": abs(smd) < 0.1,
            "Variance OK (0.5-2.0)": 0.5 <= vr <= 2.0,
        })

    return pd.DataFrame(balance_metrics)


def plot_balance_assessment(balance_df: pd.DataFrame) -> None:
    """
    Detailed balance diagnostic plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(balance_df) * 0.35)))

    # SMD plot
    ax = axes[0]
    sorted_df = balance_df.sort_values("|SMD|", ascending=True)
    colors = ["#27ae60" if ok else "#e74c3c" for ok in sorted_df["Balance OK (|SMD|<0.1)"]]
    ax.barh(sorted_df["Covariate"], sorted_df["|SMD|"], color=colors, alpha=0.8)
    ax.axvline(0.1, color="orange", linestyle="--", linewidth=2, label="SMD = 0.1 threshold")
    ax.set_xlabel("|Standardized Mean Difference|", fontsize=12)
    ax.set_title("Covariate Balance (Before Matching)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    # Variance ratio plot
    ax = axes[1]
    colors = ["#27ae60" if ok else "#e74c3c" for ok in sorted_df["Variance OK (0.5-2.0)"]]
    ax.barh(sorted_df["Covariate"], sorted_df["Variance Ratio"], color=colors, alpha=0.8)
    ax.axvline(0.5, color="orange", linestyle="--", linewidth=1.5)
    ax.axvline(2.0, color="orange", linestyle="--", linewidth=1.5, label="Acceptable range")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlabel("Variance Ratio (Treated / Control)", fontsize=12)
    ax.set_title("Variance Ratio Check", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Pre-Matching Covariate Balance Diagnostics",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "05_covariate_balance.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved covariate balance → {cfg.FIGURES_DIR / '05_covariate_balance.png'}")


# ═══════════════════════════════════════════════════════════════════
# 4. REFUTATION: RANDOM COMMON CAUSE
# ═══════════════════════════════════════════════════════════════════

def random_common_cause_test(X, y, treatment, n_trials=20):
    """
    Add a random (irrelevant) variable as a common cause and re-estimate.

    If the estimate changes significantly, the method is fragile.
    A robust method should give similar estimates regardless of
    irrelevant noise variables.
    """
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    base_ate = _ipw_estimate(X, y, treatment)

    perturbed_ates = []
    for _ in range(n_trials):
        X_aug = X.copy()
        X_aug["random_noise"] = rng.normal(0, 1, size=len(X))
        perturbed_ate = _ipw_estimate(X_aug, y, treatment)
        perturbed_ates.append(perturbed_ate)

    perturbed_ates = np.array(perturbed_ates)
    mean_shift = np.abs(perturbed_ates.mean() - base_ate)
    max_shift = np.max(np.abs(perturbed_ates - base_ate))

    result = {
        "base_ate": base_ate,
        "perturbed_ates": perturbed_ates,
        "mean_shift": mean_shift,
        "max_shift": max_shift,
        "robust": mean_shift < 0.1 * abs(base_ate),
    }

    print(f"\n    Base ATE:    ₹{base_ate:,.0f}")
    print(f"    Mean shift:  ₹{mean_shift:,.1f} ({mean_shift/abs(base_ate)*100:.1f}% of ATE)")
    print(f"    Max shift:   ₹{max_shift:,.1f}")
    print(f"    → {'✓ ROBUST' if result['robust'] else '⚠ SENSITIVE'}: "
          f"Adding random noise {'does not' if result['robust'] else 'DOES'} change the estimate")

    return result


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    """Run all sensitivity and robustness checks."""
    print("\n🛡️ STEP 5: Sensitivity & Robustness Analysis")
    print("-" * 50)

    # Load data
    df = pd.read_csv(cfg.DATA_DIR / "synthetic_data.csv")
    X = pd.read_csv(cfg.DATA_DIR / "feature_matrix.csv")
    treatment = df["treatment"].values
    y = df["purchase_amount"].values

    # ── 1. Placebo Test ──
    print("\n  ── 1. Placebo Test (Randomization Inference) ──")
    placebo_result = placebo_test(X, y, treatment)
    plot_placebo_test(placebo_result)

    # ── 2. Rosenbaum Bounds ──
    print("\n  ── 2. Rosenbaum Sensitivity Bounds ──")

    # Need matched pairs for Rosenbaum bounds
    print("    Creating matched pairs for Rosenbaum analysis...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ps_model = LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_SEED)
    ps_model.fit(X_scaled, treatment)
    ps = ps_model.predict_proba(X_scaled)[:, 1]

    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(ps[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))

    # Apply caliper
    max_dist = cfg.PSM_CALIPER * ps.std()
    valid = distances.flatten() < max_dist
    matched_t = treated_idx[valid]
    matched_c = control_idx[indices.flatten()[valid]]

    rb_result = rosenbaum_bounds(y[matched_t], y[matched_c])
    plot_rosenbaum_bounds(rb_result)

    # ── 3. Covariate Balance ──
    print("\n  ── 3. Covariate Balance Assessment ──")
    balance_df = covariate_balance_assessment(X, treatment)
    plot_balance_assessment(balance_df)
    n_imbalanced = (~balance_df["Balance OK (|SMD|<0.1)"]).sum()
    print(f"    {n_imbalanced} / {len(balance_df)} covariates have |SMD| > 0.1 (pre-matching)")

    # ── 4. Random Common Cause ──
    print("\n  ── 4. Random Common Cause Refutation ──")
    rcc_result = random_common_cause_test(X, y, treatment)

    # ── Summary ──
    print(f"\n\n  {'=' * 60}")
    print(f"  SENSITIVITY ANALYSIS SUMMARY")
    print(f"  {'=' * 60}")
    print(f"  Placebo Test:         {'✓ PASSED' if placebo_result['p_value'] < 0.05 else '✗ FAILED'} (p = {placebo_result['p_value']:.4f})")
    print(f"  Rosenbaum Γ:          {rb_result['critical_gamma']:.2f} ({'Robust' if rb_result['critical_gamma'] >= 2.0 else 'Moderate' if rb_result['critical_gamma'] >= 1.5 else 'Fragile'})")
    print(f"  Covariate Imbalance:  {n_imbalanced} covariates (pre-matching)")
    print(f"  Random Cause Test:    {'✓ ROBUST' if rcc_result['robust'] else '⚠ SENSITIVE'}")
    print(f"  {'=' * 60}")

    # Save results
    balance_df.to_csv(cfg.TABLES_DIR / "05_covariate_balance.csv", index=False)
    rb_result["results_df"].to_csv(cfg.TABLES_DIR / "05_rosenbaum_bounds.csv", index=False)

    sensitivity_summary = pd.DataFrame([{
        "Test": "Placebo Test",
        "Result": "PASSED" if placebo_result["p_value"] < 0.05 else "FAILED",
        "Metric": f"p = {placebo_result['p_value']:.4f}",
    }, {
        "Test": "Rosenbaum Bounds",
        "Result": "Robust" if rb_result["critical_gamma"] >= 2.0 else "Moderate",
        "Metric": f"Γ = {rb_result['critical_gamma']:.2f}",
    }, {
        "Test": "Random Common Cause",
        "Result": "ROBUST" if rcc_result["robust"] else "SENSITIVE",
        "Metric": f"Mean shift = ₹{rcc_result['mean_shift']:,.1f}",
    }])
    sensitivity_summary.to_csv(cfg.TABLES_DIR / "05_sensitivity_summary.csv", index=False)
    print(f"\n  ✓ Saved sensitivity summary → {cfg.TABLES_DIR / '05_sensitivity_summary.csv'}")

    return {
        "placebo": placebo_result,
        "rosenbaum": rb_result,
        "balance": balance_df,
        "random_cause": rcc_result,
    }


if __name__ == "__main__":
    results = main()
