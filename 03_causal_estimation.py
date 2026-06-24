"""
03_causal_estimation.py — Causal Methods: PSM, IPW, Causal Forest
==================================================================

The core demonstration of WHY correlation ≠ causation.

We implement four estimation strategies and show their results side-by-side:

1. Naïve Comparison — simple difference in means (BIASED by confounding)
2. Propensity Score Matching (PSM) — match treated to similar controls
3. Inverse Propensity Weighting (IPW) — re-weight to remove selection bias
4. Causal Forest (EconML) — ML-based heterogeneous effect estimation

The naïve estimate will be ~2× the true ATE. The causal methods should
all recover the true ATE within their confidence intervals.

Key Assumptions (documented because they matter):
─────────────────────────────────────────────────
1. UNCONFOUNDEDNESS (Conditional Ignorability):
   Treatment assignment is independent of potential outcomes, given X.
   Y(0), Y(1) ⊥ T | X
   → Breaks if there's an unobserved variable affecting both T and Y.

2. SUTVA (Stable Unit Treatment Value Assumption):
   One unit's treatment doesn't affect another's outcome.
   → Breaks with network effects (e.g., users sharing discount codes).

3. OVERLAP (Positivity):
   0 < P(T=1|X) < 1 for all X in the support.
   → Breaks if some users have zero probability of being treated.

4. CORRECT MODEL SPECIFICATION:
   The propensity model captures the true selection mechanism.
   → Always a risk; we mitigate with flexible models.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings

import config as cfg

# EconML for Causal Forest
try:
    from econml.dml import CausalForestDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False
    print("⚠ econml not installed. Causal Forest will be skipped.")

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# 1. NAÏVE COMPARISON (the WRONG answer)
# ═══════════════════════════════════════════════════════════════════

def naive_comparison(df: pd.DataFrame) -> dict:
    """
    Simple difference in group means — the approach that FAILS
    when treatment is not randomly assigned.

    This is what a well-meaning analyst might compute:
        ATE_naive = E[Y|T=1] - E[Y|T=0]

    The problem: E[Y|T=1] - E[Y|T=0] ≠ E[Y(1) - Y(0)]
    when T is correlated with Y through confounders.

    In our data, high-value users got treated more AND buy more
    regardless of treatment → upward bias.
    """
    treated = df[df["treatment"] == 1]["purchase_amount"]
    control = df[df["treatment"] == 0]["purchase_amount"]

    ate = treated.mean() - control.mean()
    t_stat, p_value = stats.ttest_ind(treated, control)

    # Bootstrap CI
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    boot_ates = []
    for _ in range(cfg.N_BOOTSTRAP):
        t_boot = rng.choice(treated.values, size=len(treated), replace=True)
        c_boot = rng.choice(control.values, size=len(control), replace=True)
        boot_ates.append(t_boot.mean() - c_boot.mean())
    ci_lower, ci_upper = np.percentile(boot_ates, [2.5, 97.5])

    result = {
        "method": "Naïve Comparison",
        "ate": ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_treated": len(treated),
        "n_control": len(control),
    }

    print(f"\n  ── Naïve Comparison (BIASED) ──")
    print(f"  ATE = ₹{ate:,.0f}  [95% CI: ₹{ci_lower:,.0f} to ₹{ci_upper:,.0f}]")
    print(f"  p-value = {p_value:.2e}")
    print(f"  ⚠ This is WRONG! It includes the effect of confounders.")

    return result


# ═══════════════════════════════════════════════════════════════════
# 2. PROPENSITY SCORE MATCHING (PSM)
# ═══════════════════════════════════════════════════════════════════

def estimate_propensity_scores(X: pd.DataFrame, treatment: np.ndarray) -> np.ndarray:
    """
    Estimate propensity scores using logistic regression.

    P(T=1|X) — the probability of receiving treatment given covariates.

    Why logistic regression? It's simple, interpretable, and performs
    well for propensity estimation when the model is reasonably specified.
    More flexible models (RF, GBM) can be used but risk overfitting
    the propensity, which actually HURTS matching quality.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=cfg.RANDOM_SEED)
    model.fit(X_scaled, treatment)

    propensity = model.predict_proba(X_scaled)[:, 1]
    return propensity


def propensity_score_matching(
    df: pd.DataFrame,
    X: pd.DataFrame,
    caliper: float = cfg.PSM_CALIPER,
) -> dict:
    """
    1:1 Nearest-neighbor propensity score matching with caliper.

    Algorithm:
    1. Estimate P(T=1|X) for all units.
    2. For each treated unit, find the closest control unit by |PS_t - PS_c|.
    3. Only match if distance < caliper × std(PS).
    4. Compute ATT from matched pairs.

    The caliper prevents bad matches — if no good match exists, the
    treated unit is dropped. This trades sample size for match quality.

    Returns ATT (Average Treatment effect on the Treated), not ATE,
    because we're matching controls TO treated units.
    """
    treatment = df["treatment"].values
    y = df["purchase_amount"].values

    # Step 1: Estimate propensity scores
    ps = estimate_propensity_scores(X, treatment)
    ps_std = ps.std()

    # Step 2: Find nearest control for each treated unit
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(ps[control_idx].reshape(-1, 1))
    distances, indices = nn.kneighbors(ps[treated_idx].reshape(-1, 1))

    # Step 3: Apply caliper
    max_distance = caliper * ps_std
    valid_mask = distances.flatten() < max_distance

    matched_treated = treated_idx[valid_mask]
    matched_control = control_idx[indices.flatten()[valid_mask]]

    n_matched = len(matched_treated)
    n_dropped = len(treated_idx) - n_matched

    # Step 4: Compute ATT from matched sample
    att = y[matched_treated].mean() - y[matched_control].mean()

    # Bootstrap CI on matched pairs
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    boot_atts = []
    for _ in range(cfg.N_BOOTSTRAP):
        boot_idx = rng.choice(n_matched, size=n_matched, replace=True)
        bt = y[matched_treated[boot_idx]].mean()
        bc = y[matched_control[boot_idx]].mean()
        boot_atts.append(bt - bc)
    ci_lower, ci_upper = np.percentile(boot_atts, [2.5, 97.5])

    result = {
        "method": "Propensity Score Matching",
        "ate": att,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": stats.ttest_rel(
            y[matched_treated], y[matched_control]
        ).pvalue,
        "n_treated": n_matched,
        "n_control": n_matched,
        "n_dropped": n_dropped,
        "propensity_scores": ps,
        "matched_treated_idx": matched_treated,
        "matched_control_idx": matched_control,
    }

    print(f"\n  ── Propensity Score Matching ──")
    print(f"  Matched pairs: {n_matched:,} (dropped {n_dropped:,} for caliper)")
    print(f"  ATT = ₹{att:,.0f}  [95% CI: ₹{ci_lower:,.0f} to ₹{ci_upper:,.0f}]")
    print(f"  Caliper: {caliper} × σ(PS) = {max_distance:.4f}")

    return result


# ═══════════════════════════════════════════════════════════════════
# 3. INVERSE PROPENSITY WEIGHTING (IPW)
# ═══════════════════════════════════════════════════════════════════

def inverse_propensity_weighting(
    df: pd.DataFrame,
    X: pd.DataFrame,
) -> dict:
    """
    Horvitz-Thompson IPW estimator.

    Instead of matching, we RE-WEIGHT each observation to create a
    pseudo-population where treatment is independent of X.

    Weight for treated: w = 1 / P(T=1|X)
    Weight for control: w = 1 / (1 - P(T=1|X))

    The intuition: A treated user with P(T=1|X) = 0.9 is "expected"
    to be treated, so they get low weight (1/0.9). A treated user
    with P(T=1|X) = 0.1 is "surprising" — they're similar to controls
    — so they get high weight (1/0.1), making them count more.

    Extreme weights (P close to 0 or 1) are trimmed to prevent
    any single observation from dominating the estimate.
    """
    treatment = df["treatment"].values
    y = df["purchase_amount"].values

    # Estimate propensity scores
    ps = estimate_propensity_scores(X, treatment)

    # Trim extreme propensity scores (positivity enforcement)
    ps_clipped = np.clip(
        ps,
        np.percentile(ps, cfg.IPW_TRIM_PERCENTILES[0]),
        np.percentile(ps, cfg.IPW_TRIM_PERCENTILES[1]),
    )

    # IPW weights
    weights = np.where(
        treatment == 1,
        1.0 / ps_clipped,             # Treated: weight by 1/P(T=1|X)
        1.0 / (1.0 - ps_clipped),     # Control: weight by 1/P(T=0|X)
    )

    # Normalize weights within each group
    w_treated = weights[treatment == 1]
    w_control = weights[treatment == 0]
    w_treated = w_treated / w_treated.sum()
    w_control = w_control / w_control.sum()

    # Weighted means
    y_treated_weighted = np.sum(w_treated * y[treatment == 1])
    y_control_weighted = np.sum(w_control * y[treatment == 0])
    ate = y_treated_weighted - y_control_weighted

    # Bootstrap CI
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    boot_ates = []
    n = len(df)
    for _ in range(cfg.N_BOOTSTRAP):
        boot_idx = rng.choice(n, size=n, replace=True)
        t_boot = treatment[boot_idx]
        y_boot = y[boot_idx]
        ps_boot = ps_clipped[boot_idx]

        w_b = np.where(t_boot == 1, 1.0 / ps_boot, 1.0 / (1.0 - ps_boot))
        wt = w_b[t_boot == 1]
        wc = w_b[t_boot == 0]
        if len(wt) == 0 or len(wc) == 0:
            continue
        wt = wt / wt.sum()
        wc = wc / wc.sum()
        ate_b = np.sum(wt * y_boot[t_boot == 1]) - np.sum(wc * y_boot[t_boot == 0])
        boot_ates.append(ate_b)

    ci_lower, ci_upper = np.percentile(boot_ates, [2.5, 97.5])

    # Approximate p-value from bootstrap
    se = np.std(boot_ates)
    z = ate / se if se > 0 else np.inf
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    result = {
        "method": "Inverse Propensity Weighting",
        "ate": ate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_treated": int(treatment.sum()),
        "n_control": int((1 - treatment).sum()),
        "effective_n_treated": 1.0 / np.sum((w_treated) ** 2),  # Kish effective N
        "effective_n_control": 1.0 / np.sum((w_control) ** 2),
    }

    print(f"\n  ── Inverse Propensity Weighting ──")
    print(f"  ATE = ₹{ate:,.0f}  [95% CI: ₹{ci_lower:,.0f} to ₹{ci_upper:,.0f}]")
    print(f"  Effective N (Kish): treated={result['effective_n_treated']:.0f}, "
          f"control={result['effective_n_control']:.0f}")
    print(f"  PS range after trimming: [{ps_clipped.min():.3f}, {ps_clipped.max():.3f}]")

    return result


# ═══════════════════════════════════════════════════════════════════
# 4. CAUSAL FOREST (EconML — Double ML)
# ═══════════════════════════════════════════════════════════════════

def causal_forest_estimation(
    df: pd.DataFrame,
    X: pd.DataFrame,
) -> dict:
    """
    CausalForestDML from EconML — the modern ML approach.

    Double Machine Learning (DML) works in two stages:
    1. Residualize: Remove the effect of X on both Y and T using
       flexible ML models (random forests here).
    2. Estimate: Regress the Y-residuals on the T-residuals using
       a causal forest, which gives heterogeneous effects.

    Advantages over PSM/IPW:
    - Automatically handles high-dimensional confounders.
    - Gives individual-level CATEs, not just a single ATE.
    - Cross-fitting prevents overfitting of nuisance models.
    - Valid confidence intervals via honest splitting.

    We'll extract both the ATE and the full CATE distribution.
    """
    if not HAS_ECONML:
        return {
            "method": "Causal Forest (EconML)",
            "ate": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "p_value": np.nan,
            "cate": np.full(len(df), np.nan),
            "skipped": True,
        }

    treatment = df["treatment"].values.reshape(-1, 1)
    y = df["purchase_amount"].values

    # CausalForestDML with honest splitting and cross-fitting
    est = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=200, max_depth=10,
            min_samples_leaf=20, random_state=cfg.RANDOM_SEED,
        ),
        model_t=RandomForestClassifier(
            n_estimators=200, max_depth=10,
            min_samples_leaf=20, random_state=cfg.RANDOM_SEED,
        ),
        n_estimators=500,
        min_samples_leaf=30,
        max_depth=15,
        discrete_treatment=True,
        random_state=cfg.RANDOM_SEED,
        cv=5,  # 5-fold cross-fitting
    )

    # Fit the causal forest
    est.fit(
        Y=y,
        T=df["treatment"].values,
        X=X.values,
        W=None,  # All features in X are effect modifiers
    )

    # ATE with inference
    ate_inference = est.ate_inference(X=X.values)
    ate = ate_inference.mean_point
    ci = ate_inference.conf_int_mean(alpha=cfg.ALPHA)
    p_val = ate_inference.pvalue(value=0)

    # Individual CATEs
    cate = est.effect(X=X.values).flatten()
    cate_inference = est.effect_inference(X=X.values)
    cate_ci_lower, cate_ci_upper = cate_inference.conf_int(alpha=cfg.ALPHA)
    cate_ci_lower = cate_ci_lower.flatten()
    cate_ci_upper = cate_ci_upper.flatten()

    result = {
        "method": "Causal Forest (EconML)",
        "ate": float(ate),
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
        "p_value": float(p_val) if np.isscalar(p_val) else float(p_val[0]),
        "cate": cate,
        "cate_ci_lower": cate_ci_lower,
        "cate_ci_upper": cate_ci_upper,
        "estimator": est,
        "n_treated": int(df["treatment"].sum()),
        "n_control": int((1 - df["treatment"]).sum()),
    }

    print(f"\n  ── Causal Forest (EconML — CausalForestDML) ──")
    print(f"  ATE = ₹{ate:,.0f}  [95% CI: ₹{ci[0]:,.0f} to ₹{ci[1]:,.0f}]")
    print(f"  CATE range: ₹{cate.min():,.0f} to ₹{cate.max():,.0f}")
    print(f"  CATE mean:  ₹{cate.mean():,.0f} (should ≈ ATE)")

    return result


# ═══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def plot_ate_comparison(results: list, true_ate: float) -> None:
    """
    Side-by-side bar chart of all ATE estimates with CIs.

    This is THE key plot — it shows that the naïve estimate is
    dramatically wrong while causal methods recover the truth.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    methods = [r["method"] for r in results]
    ates = [r["ate"] for r in results]
    ci_lowers = [r["ci_lower"] for r in results]
    ci_uppers = [r["ci_upper"] for r in results]
    errors_lower = [a - cl for a, cl in zip(ates, ci_lowers)]
    errors_upper = [cu - a for a, cu in zip(ates, ci_uppers)]

    colors = [
        cfg.COLOR_PALETTE["naive"],
        cfg.COLOR_PALETTE["psm"],
        cfg.COLOR_PALETTE["ipw"],
        cfg.COLOR_PALETTE["causal_forest"],
    ]

    bars = ax.barh(
        methods, ates,
        xerr=[errors_lower, errors_upper],
        color=colors, alpha=0.85, edgecolor="white", linewidth=2,
        capsize=8, error_kw={"linewidth": 2, "capthick": 2},
    )

    # True ATE line
    ax.axvline(
        true_ate, color=cfg.COLOR_PALETTE["true_ate"],
        linestyle="--", linewidth=3, label=f"True ATE = ₹{true_ate:,.0f}",
    )

    # Annotate values
    for bar, ate_val in zip(bars, ates):
        ax.text(
            ate_val + 5, bar.get_y() + bar.get_height() / 2,
            f"₹{ate_val:,.0f}", va="center", fontsize=12, fontweight="bold",
        )

    # Bias annotation for naïve
    bias = ates[0] - true_ate
    ax.annotate(
        f"Bias: +₹{bias:,.0f}\n({bias/true_ate*100:+.0f}%)",
        xy=(ates[0], 0), xytext=(ates[0] + 30, 0.3),
        fontsize=11, color=cfg.COLOR_PALETTE["naive"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=cfg.COLOR_PALETTE["naive"]),
    )

    ax.set_xlabel("Estimated Average Treatment Effect (₹)", fontsize=13)
    ax.set_title(
        "Naïve vs. Causal Estimates of Treatment Effect\n"
        "Confounding Inflates the Naïve Estimate by ~2×",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "03_ate_comparison.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ Saved ATE comparison → {cfg.FIGURES_DIR / '03_ate_comparison.png'}")


def plot_propensity_overlap(ps: np.ndarray, treatment: np.ndarray) -> None:
    """
    Plot propensity score distributions for treated vs. control.

    Good overlap = the distributions have substantial overlap.
    Poor overlap = positivity violation → unreliable estimates.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    ax.hist(ps[treatment == 1], bins=50, alpha=0.5, color="#e74c3c",
            label="Treated", density=True)
    ax.hist(ps[treatment == 0], bins=50, alpha=0.5, color="#3498db",
            label="Control", density=True)
    ax.set_xlabel("Estimated Propensity Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Propensity Score Distributions", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    # Mirror plot (better for overlap assessment)
    ax = axes[1]
    ax.hist(ps[treatment == 1], bins=50, alpha=0.6, color="#e74c3c",
            label="Treated", density=True)
    ax.hist(ps[treatment == 0], bins=50, alpha=0.6, color="#3498db",
            label="Control", density=True, bottom=0)
    ax.set_xlabel("Estimated Propensity Score", fontsize=12)
    ax.set_title("Overlap Assessment", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    # Overlap region shading
    min_ps = max(ps[treatment == 1].min(), ps[treatment == 0].min())
    max_ps = min(ps[treatment == 1].max(), ps[treatment == 0].max())
    ax.axvspan(min_ps, max_ps, alpha=0.1, color="green", label="Common support")

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "03_propensity_overlap.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved propensity overlap → {cfg.FIGURES_DIR / '03_propensity_overlap.png'}")


def plot_covariate_balance(
    df: pd.DataFrame,
    X: pd.DataFrame,
    matched_treated_idx: np.ndarray,
    matched_control_idx: np.ndarray,
) -> None:
    """
    Love plot: Standardized Mean Differences before and after matching.

    SMD < 0.1 is the standard threshold for acceptable balance.
    This shows that matching successfully removes the confounding.
    """
    treatment = df["treatment"].values
    feature_names = X.columns.tolist()

    smd_before = []
    smd_after = []

    for col in feature_names:
        x = X[col].values

        # Before matching
        mean_t = x[treatment == 1].mean()
        mean_c = x[treatment == 0].mean()
        pooled_std = np.sqrt((x[treatment == 1].var() + x[treatment == 0].var()) / 2)
        smd_b = abs(mean_t - mean_c) / pooled_std if pooled_std > 0 else 0
        smd_before.append(smd_b)

        # After matching
        mean_t_m = x[matched_treated_idx].mean()
        mean_c_m = x[matched_control_idx].mean()
        pooled_std_m = np.sqrt(
            (x[matched_treated_idx].var() + x[matched_control_idx].var()) / 2
        )
        smd_a = abs(mean_t_m - mean_c_m) / pooled_std_m if pooled_std_m > 0 else 0
        smd_after.append(smd_a)

    # Love plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.4)))

    y_pos = range(len(feature_names))
    ax.scatter(smd_before, y_pos, color="#e74c3c", s=80, zorder=5,
               label="Before Matching", marker="s")
    ax.scatter(smd_after, y_pos, color="#27ae60", s=80, zorder=5,
               label="After Matching", marker="o")

    # Connect before/after
    for i in range(len(feature_names)):
        ax.plot([smd_before[i], smd_after[i]], [i, i],
                color="gray", alpha=0.4, linewidth=1)

    # Threshold line
    ax.axvline(0.1, color="orange", linestyle="--", linewidth=2,
               label="SMD = 0.1 threshold")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel("Absolute Standardized Mean Difference", fontsize=12)
    ax.set_title("Covariate Balance: Before vs. After Matching (Love Plot)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "03_covariate_balance.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved covariate balance (Love plot) → {cfg.FIGURES_DIR / '03_covariate_balance.png'}")


def plot_cate_distribution(cate: np.ndarray, true_tau: np.ndarray) -> None:
    """
    Compare estimated CATEs from Causal Forest vs. ground truth τ(X).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution comparison
    ax = axes[0]
    ax.hist(true_tau, bins=30, alpha=0.5, color="#e67e22",
            label="True τ(X)", density=True)
    ax.hist(cate, bins=30, alpha=0.5, color="#9b59b6",
            label="Estimated CATE", density=True)
    ax.set_xlabel("Treatment Effect (₹)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("CATE Distribution: Estimated vs. Ground Truth", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    # Scatter: estimated vs. true
    ax = axes[1]
    ax.scatter(true_tau, cate, alpha=0.05, s=5, color="#9b59b6")
    lims = [min(true_tau.min(), cate.min()) - 20, max(true_tau.max(), cate.max()) + 20]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=2, label="Perfect calibration")
    ax.set_xlabel("True τ(X) [₹]", fontsize=12)
    ax.set_ylabel("Estimated CATE [₹]", fontsize=12)
    ax.set_title("CATE Calibration", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Correlation
    corr = np.corrcoef(true_tau, cate)[0, 1]
    ax.text(0.05, 0.92, f"r = {corr:.3f}", transform=ax.transAxes,
            fontsize=13, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "03_cate_distribution.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved CATE distribution → {cfg.FIGURES_DIR / '03_cate_distribution.png'}")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    """Run all causal estimation methods and compare."""
    print("\n🔬 STEP 3: Causal Estimation Methods")
    print("-" * 50)

    # Load data
    df = pd.read_csv(cfg.DATA_DIR / "synthetic_data.csv")
    X = pd.read_csv(cfg.DATA_DIR / "feature_matrix.csv")
    true_ate = df["tau_true"].mean()

    print(f"  Ground truth ATE: ₹{true_ate:,.0f}")

    # Run all methods
    results = []

    # 1. Naïve (biased)
    results.append(naive_comparison(df))

    # 2. PSM
    psm_result = propensity_score_matching(df, X)
    results.append(psm_result)

    # 3. IPW
    results.append(inverse_propensity_weighting(df, X))

    # 4. Causal Forest
    cf_result = causal_forest_estimation(df, X)
    results.append(cf_result)

    # ── Summary Table ──
    print(f"\n\n  {'=' * 70}")
    print(f"  COMPARISON OF ESTIMATION METHODS")
    print(f"  {'=' * 70}")
    print(f"  {'Method':<30} {'ATE (₹)':>10} {'95% CI':>25} {'Bias':>10}")
    print(f"  {'-' * 75}")
    for r in results:
        bias = r["ate"] - true_ate
        print(f"  {r['method']:<30} {r['ate']:>10,.0f} "
              f"[{r['ci_lower']:>8,.0f}, {r['ci_upper']:>8,.0f}] "
              f"{bias:>+10,.0f}")
    print(f"  {'-' * 75}")
    print(f"  {'True ATE':<30} {true_ate:>10,.0f}")
    print(f"  {'=' * 70}")

    # Save results table
    results_df = pd.DataFrame([{
        "Method": r["method"],
        "ATE (₹)": round(r["ate"]),
        "CI Lower (₹)": round(r["ci_lower"]),
        "CI Upper (₹)": round(r["ci_upper"]),
        "Bias (₹)": round(r["ate"] - true_ate),
        "Bias (%)": round((r["ate"] / true_ate - 1) * 100, 1),
    } for r in results])
    results_df.to_csv(cfg.TABLES_DIR / "03_ate_comparison.csv", index=False)

    # ── Plots ──
    plot_ate_comparison(results, true_ate)

    if "propensity_scores" in psm_result:
        plot_propensity_overlap(
            psm_result["propensity_scores"], df["treatment"].values
        )
        plot_covariate_balance(
            df, X,
            psm_result["matched_treated_idx"],
            psm_result["matched_control_idx"],
        )

    if "cate" in cf_result and not cf_result.get("skipped"):
        plot_cate_distribution(cf_result["cate"], df["tau_true"].values)

        # Save CATEs for uplift modeling
        df["cate_causal_forest"] = cf_result["cate"]
        df.to_csv(cfg.DATA_DIR / "synthetic_data.csv", index=False)
        print(f"\n  ✓ Saved CATEs to dataset")

    return results


if __name__ == "__main__":
    results = main()
