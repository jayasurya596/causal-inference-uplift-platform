"""
04_uplift_modeling.py — Uplift Models, Segmentation & ROI Analysis
===================================================================

This is where we go beyond "who will buy" to "who will buy BECAUSE of
the intervention." This is the critical distinction:

    Response model:  P(Y=1 | X)          → "Who will buy?"
    Uplift model:    P(Y=1|T=1,X) - P(Y=1|T=0,X)  → "Who will buy MORE because we acted?"

The response model targets Sure Things (high P(Y=1) regardless of T).
The uplift model targets Persuadables (high change in P(Y=1) due to T).

Targeting Sure Things wastes money (they'd buy anyway).
Targeting Persuadables maximizes incremental revenue.

We implement three meta-learners:
    S-Learner: Single model, treatment as a feature (simplest)
    T-Learner: Two separate models (most intuitive)
    X-Learner: Propensity-weighted hybrid (most robust for imbalanced data)

Then we segment users into the classic four quadrants and compute
the ₹ impact of switching from probability-based to uplift-based targeting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

import config as cfg

# CausalML meta-learners
try:
    from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor
    from causalml.metrics import plot_qini, auuc_score
    HAS_CAUSALML = True
except ImportError:
    HAS_CAUSALML = False
    print("⚠ causalml not installed. Using manual meta-learner implementations.")

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════
# META-LEARNER IMPLEMENTATIONS (Manual fallback if causalml unavailable)
# ═══════════════════════════════════════════════════════════════════

import joblib
from app.models import SLearner, TLearner, XLearner


def manual_s_learner(X, treatment, y):
    """S-Learner wrapper function for API compatibility."""
    return SLearner(random_state=cfg.RANDOM_SEED).fit(X, treatment, y).predict(X)


def manual_t_learner(X, treatment, y):
    """T-Learner wrapper function for API compatibility."""
    return TLearner(random_state=cfg.RANDOM_SEED).fit(X, treatment, y).predict(X)


def manual_x_learner(X, treatment, y):
    """X-Learner wrapper function for API compatibility."""
    return XLearner(random_state=cfg.RANDOM_SEED).fit(X, treatment, y).predict(X)


# ═══════════════════════════════════════════════════════════════════
# UPLIFT MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_uplift_models(X, treatment, y):
    """
    Train custom S, T, and X learners and return predictions and models.
    """
    results = {}
    models = {}

    # Standardize on custom wrapper classes so that saved models do not require causalml dependency
    print("    Training S-Learner Wrapper...")
    s_learner = SLearner(random_state=cfg.RANDOM_SEED).fit(X, treatment, y)
    results["S-Learner"] = s_learner.predict(X)
    models["S-Learner"] = s_learner

    print("    Training T-Learner Wrapper...")
    t_learner = TLearner(random_state=cfg.RANDOM_SEED).fit(X, treatment, y)
    results["T-Learner"] = t_learner.predict(X)
    models["T-Learner"] = t_learner

    print("    Training X-Learner Wrapper...")
    x_learner = XLearner(random_state=cfg.RANDOM_SEED).fit(X, treatment, y)
    results["X-Learner"] = x_learner.predict(X)
    models["X-Learner"] = x_learner

    # Print summary
    print(f"\n  ── Uplift Model ATE Estimates ──")
    for name, cate in results.items():
        print(f"  {name}: Mean CATE = ₹{cate.mean():,.0f}, "
              f"Std = ₹{cate.std():,.0f}, "
              f"Range = [₹{cate.min():,.0f}, ₹{cate.max():,.0f}]")

    return results, models


# ═══════════════════════════════════════════════════════════════════
# UPLIFT EVALUATION
# ═══════════════════════════════════════════════════════════════════

def compute_uplift_by_decile(cate, treatment, y):
    """
    Compute actual uplift by predicted CATE decile.

    For each decile (ranked by predicted CATE):
    1. Compute the observed uplift = mean(Y|T=1) - mean(Y|T=0)
    2. Compare to the predicted uplift = mean(CATE in decile)

    If the model is well-calibrated, these should be correlated:
    deciles with high predicted CATE should have high actual uplift.
    """
    df_eval = pd.DataFrame({
        "cate": cate,
        "treatment": treatment,
        "y": y,
    })

    df_eval["decile"] = pd.qcut(df_eval["cate"], cfg.N_DECILES,
                                 labels=range(1, cfg.N_DECILES + 1),
                                 duplicates="drop")

    decile_results = []
    for decile in sorted(df_eval["decile"].unique()):
        subset = df_eval[df_eval["decile"] == decile]
        treated = subset[subset["treatment"] == 1]["y"]
        control = subset[subset["treatment"] == 0]["y"]

        if len(treated) > 0 and len(control) > 0:
            actual_uplift = treated.mean() - control.mean()
        else:
            actual_uplift = np.nan

        decile_results.append({
            "decile": decile,
            "n": len(subset),
            "predicted_uplift": subset["cate"].mean(),
            "actual_uplift": actual_uplift,
            "n_treated": len(treated),
            "n_control": len(control),
        })

    return pd.DataFrame(decile_results)


def plot_uplift_by_decile(decile_results: dict) -> None:
    """Plot actual vs. predicted uplift by decile for each model."""
    n_models = len(decile_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    colors = ["#3498db", "#e74c3c", "#27ae60"]

    for ax, (name, df_dec), color in zip(axes, decile_results.items(), colors):
        x = df_dec["decile"].values
        ax.bar(x - 0.15, df_dec["predicted_uplift"], width=0.3, alpha=0.7,
               color=color, label="Predicted")
        ax.bar(x + 0.15, df_dec["actual_uplift"], width=0.3, alpha=0.7,
               color="gray", label="Actual")
        ax.set_xlabel("CATE Decile (1=lowest, 10=highest)", fontsize=11)
        ax.set_ylabel("Uplift (₹)", fontsize=11)
        ax.set_title(f"{name}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Uplift by Predicted CATE Decile", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "04_uplift_by_decile.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved uplift by decile → {cfg.FIGURES_DIR / '04_uplift_by_decile.png'}")


def plot_cumulative_uplift(cate_dict, treatment, y) -> None:
    """
    Cumulative uplift curves (manual Qini-style).

    Shows the cumulative incremental gain as we target more users,
    starting from those with the highest predicted CATE.

    The steeper the curve at the left, the better the model is at
    identifying high-uplift users.
    """
    fig, ax = plt.subplots(figsize=cfg.FIGURE_SIZE)
    colors = {"S-Learner": "#3498db", "T-Learner": "#e74c3c", "X-Learner": "#27ae60"}

    for name, cate in cate_dict.items():
        # Sort by predicted CATE (descending)
        order = np.argsort(-cate)
        t_sorted = treatment[order]
        y_sorted = y[order]

        # Cumulative uplift
        cum_treated = np.cumsum(t_sorted * y_sorted)
        cum_control = np.cumsum((1 - t_sorted) * y_sorted)
        cum_n_treated = np.cumsum(t_sorted)
        cum_n_control = np.cumsum(1 - t_sorted)

        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            cum_uplift = np.where(
                (cum_n_treated > 0) & (cum_n_control > 0),
                cum_treated / cum_n_treated - cum_control / cum_n_control,
                0,
            )

        # Plot as fraction of population
        x_pct = np.arange(1, len(cate) + 1) / len(cate) * 100
        ax.plot(x_pct, cum_uplift, label=name, color=colors.get(name, "gray"),
                linewidth=2)

    # Random targeting baseline
    overall_uplift = y[treatment == 1].mean() - y[treatment == 0].mean()
    ax.axhline(overall_uplift, color="gray", linestyle="--", alpha=0.5,
               label=f"Random (naïve mean diff = ₹{overall_uplift:,.0f})")

    ax.set_xlabel("% of Users Targeted (ranked by predicted CATE)", fontsize=12)
    ax.set_ylabel("Cumulative Uplift (₹)", fontsize=12)
    ax.set_title("Cumulative Uplift Curves\n(Higher = Better Targeting)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "04_cumulative_uplift.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved cumulative uplift curves → {cfg.FIGURES_DIR / '04_cumulative_uplift.png'}")


# ═══════════════════════════════════════════════════════════════════
# USER SEGMENTATION (Four Quadrants)
# ═══════════════════════════════════════════════════════════════════

def segment_users(df: pd.DataFrame, cate: np.ndarray, X: pd.DataFrame,
                  treatment: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Segment users into the four classic uplift quadrants.

    We use two dimensions:
    1. Uplift (CATE) — how much the treatment changes behavior
    2. Baseline propensity — how likely they are to convert WITHOUT treatment

    Segments:
    ┌──────────────────┬──────────────────┐
    │   SLEEPING DOGS  │   SURE THINGS    │
    │   (CATE < 0,     │   (CATE low,     │
    │    high baseline) │    high baseline) │
    ├──────────────────┼──────────────────┤
    │   LOST CAUSES    │   PERSUADABLES   │
    │   (CATE low,     │   (CATE high,    │
    │    low baseline)  │    low baseline)  │
    └──────────────────┴──────────────────┘
    """
    # Estimate baseline (control) propensity
    model_baseline = RandomForestRegressor(
        n_estimators=100, max_depth=8,
        random_state=cfg.RANDOM_SEED, n_jobs=-1,
    )
    control_mask = treatment == 0
    model_baseline.fit(X.values[control_mask], y[control_mask])
    baseline_prediction = model_baseline.predict(X.values)

    # Medians for segmentation
    cate_median = np.median(cate)
    baseline_median = np.median(baseline_prediction)

    # Assign segments
    segments = np.full(len(cate), "", dtype=object)
    segments[(cate >= cate_median) & (baseline_prediction < baseline_median)] = "Persuadables"
    segments[(cate < cate_median) & (baseline_prediction >= baseline_median)] = "Sure Things"
    segments[(cate < cate_median) & (baseline_prediction < baseline_median)] = "Lost Causes"
    segments[(cate >= cate_median) & (baseline_prediction >= baseline_median)] = "Sure Things"

    # Override: negative CATE = Sleeping Dogs (regardless of baseline)
    segments[cate < 0] = "Sleeping Dogs"

    # Re-classify high-CATE, high-baseline as Persuadables (they respond to treatment)
    segments[(cate >= cate_median) & (baseline_prediction >= baseline_median) & (cate > 0)] = "Persuadables"

    # Refine: true persuadables are high CATE, and baseline is not extremely high
    # Use a more nuanced segmentation
    segments = np.full(len(cate), "", dtype=object)
    q75_cate = np.percentile(cate, 75)
    q25_cate = np.percentile(cate, 25)

    for i in range(len(cate)):
        if cate[i] < 0:
            segments[i] = "Sleeping Dogs"
        elif cate[i] >= q75_cate:
            segments[i] = "Persuadables"
        elif cate[i] < q25_cate and baseline_prediction[i] >= baseline_median:
            segments[i] = "Sure Things"
        elif cate[i] < q25_cate and baseline_prediction[i] < baseline_median:
            segments[i] = "Lost Causes"
        elif baseline_prediction[i] >= baseline_median:
            segments[i] = "Sure Things"
        else:
            segments[i] = "Persuadables"

    df_seg = df.copy()
    df_seg["cate_uplift"] = cate
    df_seg["baseline_prediction"] = baseline_prediction
    df_seg["segment"] = segments

    return df_seg


def plot_segmentation(df_seg: pd.DataFrame) -> None:
    """
    Scatter plot of the four segments: CATE vs. baseline prediction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Scatter plot
    ax = axes[0]
    seg_colors = {
        "Persuadables": cfg.COLOR_PALETTE["persuadables"],
        "Sure Things": cfg.COLOR_PALETTE["sure_things"],
        "Lost Causes": cfg.COLOR_PALETTE["lost_causes"],
        "Sleeping Dogs": cfg.COLOR_PALETTE["sleeping_dogs"],
    }
    for seg_name, color in seg_colors.items():
        mask = df_seg["segment"] == seg_name
        ax.scatter(
            df_seg.loc[mask, "baseline_prediction"],
            df_seg.loc[mask, "cate_uplift"],
            alpha=0.15, s=10, color=color, label=f"{seg_name} ({mask.sum():,})",
        )

    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.set_xlabel("Baseline Prediction (₹) — Would Buy Anyway", fontsize=12)
    ax.set_ylabel("Uplift / CATE (₹) — Incremental Effect", fontsize=12)
    ax.set_title("User Segmentation: Four Quadrants", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, markerscale=5)
    ax.grid(alpha=0.3)

    # Segment size pie chart
    ax = axes[1]
    seg_counts = df_seg["segment"].value_counts()
    seg_order = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]
    sizes = [seg_counts.get(s, 0) for s in seg_order]
    colors_list = [seg_colors[s] for s in seg_order]

    wedges, texts, autotexts = ax.pie(
        sizes, labels=seg_order, colors=colors_list,
        autopct=lambda pct: f"{pct:.1f}%\n({int(pct/100*sum(sizes)):,})",
        startangle=90, textprops={"fontsize": 10},
    )
    ax.set_title("Segment Distribution", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "04_user_segmentation.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved user segmentation → {cfg.FIGURES_DIR / '04_user_segmentation.png'}")


# ═══════════════════════════════════════════════════════════════════
# ROI COMPARISON: Uplift vs. Probability Targeting
# ═══════════════════════════════════════════════════════════════════

def compute_roi_comparison(df_seg: pd.DataFrame, treatment: np.ndarray,
                           y: np.ndarray) -> pd.DataFrame:
    """
    Compare three targeting strategies on incremental ROI.

    Strategy 1: RANDOM — treat everyone (current approach)
    Strategy 2: PROBABILITY — target top 50% by P(purchase) (naive ML)
    Strategy 3: UPLIFT — target top 50% by CATE (causal ML)

    The key insight: probability-based targeting wastes money on Sure Things
    who would have bought anyway. Uplift-based targeting focuses on
    Persuadables who ONLY buy because of the treatment.
    """
    n = len(df_seg)
    target_n = int(n * cfg.UPLIFT_TARGET_FRACTION)

    # ── Strategy 1: Treat Everyone ──
    total_cost_all = n * cfg.TREATMENT_COST_PER_USER
    # Use ground truth tau to compute true incremental revenue
    true_incremental_all = df_seg["tau_true"].sum() * cfg.AVERAGE_MARGIN_RATE
    roi_all = (true_incremental_all - total_cost_all) / total_cost_all * 100

    # ── Strategy 2: Probability-based (target highest baseline propensity) ──
    # This is what a traditional ML team would do: predict P(buy) and target high-prob users
    prob_ranking = np.argsort(-df_seg["baseline_prediction"].values)
    prob_targeted = prob_ranking[:target_n]
    cost_prob = target_n * cfg.TREATMENT_COST_PER_USER
    incremental_prob = df_seg.iloc[prob_targeted]["tau_true"].sum() * cfg.AVERAGE_MARGIN_RATE
    roi_prob = (incremental_prob - cost_prob) / cost_prob * 100

    # ── Strategy 3: Uplift-based (target highest CATE) ──
    uplift_ranking = np.argsort(-df_seg["cate_uplift"].values)
    uplift_targeted = uplift_ranking[:target_n]
    cost_uplift = target_n * cfg.TREATMENT_COST_PER_USER
    incremental_uplift = df_seg.iloc[uplift_targeted]["tau_true"].sum() * cfg.AVERAGE_MARGIN_RATE
    roi_uplift = (incremental_uplift - cost_uplift) / cost_uplift * 100

    # Summary table
    comparison = pd.DataFrame([
        {
            "Strategy": "Treat Everyone (Random)",
            "Users Targeted": n,
            "Cost (₹)": total_cost_all,
            "Incremental Revenue (₹)": true_incremental_all,
            "Net Profit (₹)": true_incremental_all - total_cost_all,
            "ROI (%)": roi_all,
            "Cost per Incremental ₹": total_cost_all / true_incremental_all if true_incremental_all > 0 else np.inf,
        },
        {
            "Strategy": "Probability-Based (Top 50%)",
            "Users Targeted": target_n,
            "Cost (₹)": cost_prob,
            "Incremental Revenue (₹)": incremental_prob,
            "Net Profit (₹)": incremental_prob - cost_prob,
            "ROI (%)": roi_prob,
            "Cost per Incremental ₹": cost_prob / incremental_prob if incremental_prob > 0 else np.inf,
        },
        {
            "Strategy": "Uplift-Based (Top 50%)",
            "Users Targeted": target_n,
            "Cost (₹)": cost_uplift,
            "Incremental Revenue (₹)": incremental_uplift,
            "Net Profit (₹)": incremental_uplift - cost_uplift,
            "ROI (%)": roi_uplift,
            "Cost per Incremental ₹": cost_uplift / incremental_uplift if incremental_uplift > 0 else np.inf,
        },
    ])

    return comparison


def plot_roi_comparison(comparison: pd.DataFrame) -> None:
    """
    Visualize ROI comparison across targeting strategies.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    strategies = comparison["Strategy"].values
    colors = ["#95a5a6", "#e74c3c", "#27ae60"]

    # Net Profit
    ax = axes[0]
    values = comparison["Net Profit (₹)"].values
    bars = ax.bar(range(len(strategies)), values, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=2)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Net Profit (₹)", fontsize=12)
    ax.set_title("Net Profit by Strategy", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"₹{x:,.0f}"))
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"₹{val:,.0f}", ha="center", fontsize=10, fontweight="bold")

    # ROI
    ax = axes[1]
    values = comparison["ROI (%)"].values
    bars = ax.bar(range(len(strategies)), values, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=2)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("ROI (%)", fontsize=12)
    ax.set_title("Return on Investment", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"{val:.0f}%", ha="center", fontsize=11, fontweight="bold")

    # Cost Efficiency
    ax = axes[2]
    values = comparison["Cost per Incremental ₹"].values
    bars = ax.bar(range(len(strategies)), values, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=2)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Cost per ₹1 Incremental Revenue", fontsize=12)
    ax.set_title("Cost Efficiency (Lower = Better)", fontsize=13, fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                f"₹{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    plt.suptitle("ROI Comparison: Uplift vs. Probability Targeting",
                 fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "04_roi_comparison.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved ROI comparison → {cfg.FIGURES_DIR / '04_roi_comparison.png'}")


def plot_segment_profile(df_seg: pd.DataFrame) -> None:
    """
    Profile each segment with key metrics.
    """
    seg_order = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]
    seg_stats = []

    for seg in seg_order:
        mask = df_seg["segment"] == seg
        if mask.sum() == 0:
            continue
        subset = df_seg[mask]
        seg_stats.append({
            "Segment": seg,
            "Count": mask.sum(),
            "% of Total": mask.sum() / len(df_seg) * 100,
            "Mean CATE (₹)": subset["cate_uplift"].mean(),
            "Mean Baseline (₹)": subset["baseline_prediction"].mean(),
            "Mean True τ (₹)": subset["tau_true"].mean(),
            "Mean Monetary (₹)": subset["monetary"].mean(),
            "Mean Frequency": subset["frequency"].mean(),
        })

    seg_df = pd.DataFrame(seg_stats)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")
    table = ax.table(
        cellText=[[
            row["Segment"],
            f"{row['Count']:,}",
            f"{row['% of Total']:.1f}%",
            f"₹{row['Mean CATE (₹)']:.0f}",
            f"₹{row['Mean True τ (₹)']:.0f}",
            f"₹{row['Mean Monetary (₹)']:,.0f}",
            f"{row['Mean Frequency']:.1f}",
        ] for _, row in seg_df.iterrows()],
        colLabels=["Segment", "Count", "% Total", "Mean CATE", "True τ",
                   "Avg Monetary", "Avg Freq"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color headers
    for j in range(7):
        table[0, j].set_facecolor("#2c3e50")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color segment rows
    seg_colors_map = {
        "Persuadables": "#d5f5e3",
        "Sure Things": "#d6eaf8",
        "Lost Causes": "#f2f3f4",
        "Sleeping Dogs": "#fadbd8",
    }
    for i, (_, row) in enumerate(seg_df.iterrows()):
        color = seg_colors_map.get(row["Segment"], "white")
        for j in range(7):
            table[i + 1, j].set_facecolor(color)

    ax.set_title("User Segment Profile", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(cfg.FIGURES_DIR / "04_segment_profile.png",
                dpi=cfg.FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved segment profile → {cfg.FIGURES_DIR / '04_segment_profile.png'}")

    return seg_df


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    """Run the full uplift modeling pipeline."""
    print("\n🎯 STEP 4: Uplift Modeling & Segmentation")
    print("-" * 50)

    # Load data
    df = pd.read_csv(cfg.DATA_DIR / "synthetic_data.csv")
    X = pd.read_csv(cfg.DATA_DIR / "feature_matrix.csv")
    treatment = df["treatment"].values
    y = df["purchase_amount"].values

    # ── Train Meta-Learners ──
    cate_dict, models_dict = train_uplift_models(X, treatment, y)

    # ── Evaluate by Decile ──
    print(f"\n  Computing uplift by decile...")
    decile_results = {}
    for name, cate in cate_dict.items():
        decile_results[name] = compute_uplift_by_decile(cate, treatment, y)
    plot_uplift_by_decile(decile_results)
    plot_cumulative_uplift(cate_dict, treatment, y)

    # ── Select Best Model (by correlation with true tau) ──
    print(f"\n  ── Model Selection ──")
    best_name, best_corr = None, -np.inf
    for name, cate in cate_dict.items():
        corr = np.corrcoef(cate, df["tau_true"].values)[0, 1]
        print(f"  {name}: Correlation with true τ = {corr:.3f}")
        if corr > best_corr:
            best_name, best_corr = name, corr
    print(f"  → Best model: {best_name} (r = {best_corr:.3f})")

    best_cate = cate_dict[best_name]
    best_model_obj = models_dict[best_name]

    # Save best model to disk
    joblib.dump(best_model_obj, cfg.MODELS_DIR / "uplift_model.joblib")
    print(f"  ✓ Saved best model ({best_name}) → {cfg.MODELS_DIR / 'uplift_model.joblib'}")

    # ── User Segmentation ──
    print(f"\n  Segmenting users into four quadrants...")
    df_seg = segment_users(df, best_cate, X, treatment, y)
    plot_segmentation(df_seg)
    seg_df = plot_segment_profile(df_seg)

    # ── ROI Comparison ──
    print(f"\n  Computing ROI comparison...")
    roi_comparison = compute_roi_comparison(df_seg, treatment, y)
    plot_roi_comparison(roi_comparison)

    print(f"\n  ── ROI Comparison Table ──")
    for _, row in roi_comparison.iterrows():
        print(f"  {row['Strategy']:<35} | Net Profit: ₹{row['Net Profit (₹)']:>12,.0f} | ROI: {row['ROI (%)']:>6.0f}%")

    # Save
    roi_comparison.to_csv(cfg.TABLES_DIR / "04_roi_comparison.csv", index=False)
    seg_df.to_csv(cfg.TABLES_DIR / "04_segment_profile.csv", index=False)
    df_seg.to_csv(cfg.DATA_DIR / "segmented_users.csv", index=False)

    print(f"\n  ✓ Saved ROI comparison → {cfg.TABLES_DIR / '04_roi_comparison.csv'}")
    print(f"  ✓ Saved segmented users → {cfg.DATA_DIR / 'segmented_users.csv'}")

    return cate_dict, df_seg, roi_comparison


if __name__ == "__main__":
    cate_dict, df_seg, roi_comparison = main()
