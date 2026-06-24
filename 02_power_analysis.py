"""
02_power_analysis.py — Statistical Power & Sample Size Analysis
================================================================

Simulates the power analysis a data scientist would perform BEFORE
running the experiment (ex ante design), even though we're analyzing
historical data. This answers:

    "How large a sample do we need to detect a meaningful effect?"

Key concepts:
    - MDE (Minimum Detectable Effect): The smallest effect we can
      reliably detect. Below this, we can't distinguish signal from noise.
    - Power: P(reject H0 | H1 is true) — the probability of finding
      a real effect when it exists.
    - α: P(reject H0 | H0 is true) — false positive rate.

We compute power for both continuous (purchase amount) and binary
(purchased yes/no) outcomes.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings

import config as cfg

warnings.filterwarnings("ignore")


def compute_mde_continuous(
    n_treatment: int,
    n_control: int,
    outcome_std: float,
    alpha: float = cfg.ALPHA,
    power: float = cfg.POWER_TARGET,
) -> float:
    """
    Compute the Minimum Detectable Effect for a continuous outcome.

    Uses a two-sample t-test power calculation. The MDE is the smallest
    true difference in means we can detect with the specified power.

    Parameters
    ----------
    n_treatment : int
        Number of treated units.
    n_control : int
        Number of control units.
    outcome_std : float
        Pooled standard deviation of the outcome.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.

    Returns
    -------
    float
        MDE in the original units (₹).
    """
    analysis = TTestIndPower()
    # ratio = n_control / n_treatment
    ratio = n_control / n_treatment

    # Solve for effect size (Cohen's d)
    effect_size = analysis.solve_power(
        effect_size=None,
        nobs1=n_treatment,
        ratio=ratio,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )

    # Convert Cohen's d back to raw units
    mde = effect_size * outcome_std
    return mde


def compute_required_n(
    mde_raw: float,
    outcome_std: float,
    treat_fraction: float = 0.5,
    alpha: float = cfg.ALPHA,
    power: float = cfg.POWER_TARGET,
) -> int:
    """
    Compute required sample size for a target MDE.

    Parameters
    ----------
    mde_raw : float
        Target minimum detectable effect in raw units (₹).
    outcome_std : float
        Expected pooled standard deviation.
    treat_fraction : float
        Fraction of sample assigned to treatment.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.

    Returns
    -------
    int
        Total required sample size (treatment + control).
    """
    analysis = TTestIndPower()
    effect_size = mde_raw / outcome_std  # Cohen's d
    ratio = (1 - treat_fraction) / treat_fraction

    n_treatment = analysis.solve_power(
        effect_size=effect_size,
        nobs1=None,
        ratio=ratio,
        alpha=alpha,
        power=power,
        alternative="two-sided",
    )

    n_total = int(np.ceil(n_treatment / treat_fraction))
    return n_total


def compute_power_curve(
    outcome_std: float,
    effect_sizes: list,
    sample_sizes: list,
    treat_fraction: float = 0.5,
    alpha: float = cfg.ALPHA,
) -> pd.DataFrame:
    """
    Compute power for a grid of (effect_size, sample_size) combinations.

    Returns a DataFrame suitable for plotting power curves.
    """
    analysis = TTestIndPower()
    ratio = (1 - treat_fraction) / treat_fraction
    results = []

    for mde in effect_sizes:
        for n_total in sample_sizes:
            n_treat = int(n_total * treat_fraction)
            d = mde / outcome_std  # Cohen's d
            pwr = analysis.power(
                effect_size=d,
                nobs1=n_treat,
                ratio=ratio,
                alpha=alpha,
                alternative="two-sided",
            )
            results.append({
                "MDE (₹)": mde,
                "Sample Size": n_total,
                "Power": pwr,
            })

    return pd.DataFrame(results)


def plot_power_curves(power_df: pd.DataFrame, actual_n: int, actual_mde: float) -> None:
    """
    Plot power curves for different effect sizes across sample sizes.

    Marks the actual dataset size and MDE for reference.
    """
    fig, ax = plt.subplots(figsize=cfg.FIGURE_SIZE)

    colors = ["#e74c3c", "#e67e22", "#27ae60", "#3498db", "#9b59b6"]
    mde_values = sorted(power_df["MDE (₹)"].unique())

    for mde, color in zip(mde_values, colors):
        subset = power_df[power_df["MDE (₹)"] == mde].sort_values("Sample Size")
        ax.plot(
            subset["Sample Size"], subset["Power"],
            color=color, linewidth=2.5, label=f"MDE = ₹{mde:.0f}",
            marker="o", markersize=4,
        )

    # Reference lines
    ax.axhline(cfg.POWER_TARGET, color="gray", linestyle="--", alpha=0.6,
               label=f"Target Power = {cfg.POWER_TARGET}")
    ax.axvline(actual_n, color="black", linestyle=":", alpha=0.5,
               label=f"Our N = {actual_n:,}")

    ax.set_xlabel("Total Sample Size", fontsize=12)
    ax.set_ylabel("Statistical Power", fontsize=12)
    ax.set_title(
        "Power Analysis: Can We Detect the Treatment Effect?\n"
        f"(α = {cfg.ALPHA}, two-sided test)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "02_power_curves.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved power curves → {cfg.FIGURES_DIR / '02_power_curves.png'}")


def plot_sample_size_requirements(outcome_std: float) -> None:
    """
    Bar chart showing required sample size for different MDE targets.
    """
    mde_targets = [25, 50, 75, 100, 150, 200]
    required_ns = [
        compute_required_n(mde, outcome_std, treat_fraction=0.5)
        for mde in mde_targets
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        [f"₹{m}" for m in mde_targets], required_ns,
        color=["#e74c3c" if n > cfg.N_USERS else "#27ae60" for n in required_ns],
        alpha=0.8, edgecolor="white", linewidth=1.5,
    )

    # Annotate
    for bar, n in zip(bars, required_ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{n:,}", ha="center", fontsize=11, fontweight="bold",
        )

    ax.axhline(cfg.N_USERS, color="black", linestyle="--", linewidth=2,
               label=f"Our sample: {cfg.N_USERS:,}")
    ax.set_xlabel("Minimum Detectable Effect (₹)", fontsize=12)
    ax.set_ylabel("Required Total Sample Size", fontsize=12)
    ax.set_title(
        "Sample Size Requirements by Target MDE\n"
        f"(Power = {cfg.POWER_TARGET}, α = {cfg.ALPHA})",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "02_sample_size_requirements.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved sample size requirements → {cfg.FIGURES_DIR / '02_sample_size_requirements.png'}")


def main():
    """Run the full power analysis pipeline."""
    print("\n📊 STEP 2: Power Analysis")
    print("-" * 50)

    # Load data
    df = pd.read_csv(cfg.DATA_DIR / "synthetic_data.csv")
    n_treated = df["treatment"].sum()
    n_control = len(df) - n_treated
    outcome_std = df["purchase_amount"].std()

    print(f"  Dataset: {len(df):,} users ({n_treated:,} treated, {n_control:,} control)")
    print(f"  Outcome std: ₹{outcome_std:,.0f}")
    print(f"  Treatment fraction: {n_treated / len(df):.1%}")

    # ── MDE Calculation ──
    mde = compute_mde_continuous(n_treated, n_control, outcome_std)
    print(f"\n  ── Minimum Detectable Effect (MDE) ──")
    print(f"  With N = {len(df):,} (α = {cfg.ALPHA}, power = {cfg.POWER_TARGET}):")
    print(f"  MDE = ₹{mde:,.0f}")
    print(f"  → We can detect effects ≥ ₹{mde:,.0f} with {cfg.POWER_TARGET:.0%} power")
    print(f"  → True ATE (₹{cfg.TRUE_ATE_BASE:.0f}) is {'detectable ✓' if cfg.TRUE_ATE_BASE > mde else 'NOT detectable ✗'}")

    # ── Required Sample Size for Target MDE ──
    required_n = compute_required_n(cfg.MDE_TARGET, outcome_std)
    print(f"\n  ── Required Sample Size ──")
    print(f"  To detect MDE = ₹{cfg.MDE_TARGET:.0f}:")
    print(f"  Required N = {required_n:,}")
    print(f"  Our N = {len(df):,} → {'Sufficient ✓' if len(df) >= required_n else 'Insufficient ✗'}")

    # ── Power Curves ──
    print(f"\n  Generating power curves...")
    effect_sizes = [50, 75, 100, 150, 200]
    sample_sizes = list(range(500, 25001, 500))
    power_df = compute_power_curve(outcome_std, effect_sizes, sample_sizes)
    plot_power_curves(power_df, len(df), mde)

    # ── Sample Size Requirements ──
    plot_sample_size_requirements(outcome_std)

    # ── Summary Table ──
    summary = []
    for mde_target in [25, 50, 75, 100, 150, 200]:
        req_n = compute_required_n(mde_target, outcome_std)
        summary.append({
            "MDE (₹)": mde_target,
            "Cohen's d": mde_target / outcome_std,
            "Required N": req_n,
            "Feasible?": "✓" if cfg.N_USERS >= req_n else "✗",
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(cfg.TABLES_DIR / "02_power_analysis_summary.csv", index=False)
    print(f"\n  Summary Table:")
    print(summary_df.to_string(index=False))
    print(f"\n  ✓ Saved summary → {cfg.TABLES_DIR / '02_power_analysis_summary.csv'}")

    return power_df, summary_df


if __name__ == "__main__":
    power_df, summary_df = main()
