"""Figures for note 04 -- Linear Regression."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, LINE, NAVY, PAPER, SKY, WARN, blank_axes, figure, panel_title

# The house-price worked example used throughout note 04.
AREA = np.array([500, 750, 1000, 1250, 1500.])
PRICE = np.array([1.8, 2.6, 3.4, 4.2, 5.1])
B1, B0 = 0.0033, 0.14


@figure("best_fit_line")
def _best_fit_line() -> plt.Figure:
    """The five house points, the fitted line, and the 2000 sq.ft extrapolation."""
    fig, ax = plt.subplots(figsize=(9.0, 5.6), facecolor=PAPER)

    grid = np.linspace(400, 2100, 200)
    ax.plot(grid, B1 * grid + B0, color=NAVY, linewidth=2.4, zorder=2,
            label="Fitted line:  Price = 0.0033 x Area + 0.14")

    # Vertical gaps from each point to the line -- what the cost function squares.
    # On this data the fit is near-perfect, so they are deliberately tiny.
    for a, p in zip(AREA, PRICE):
        ax.plot([a, a], [p, B1 * a + B0], color=WARN, linewidth=1.8, linestyle="--", zorder=1)
    ax.scatter(AREA, PRICE, s=120, color=BLUE, zorder=3, label="Actual sales")

    predicted = B1 * 2000 + B0
    ax.scatter([2000], [predicted], s=190, marker="*", color=GOOD, zorder=4,
               label=f"Prediction at 2000 sq.ft = {predicted:.2f} lakhs")
    ax.plot([2000, 2000], [0, predicted], color=GOOD, linewidth=1.4, linestyle=":", zorder=1)
    ax.plot([400, 2000], [predicted, predicted], color=GOOD, linewidth=1.4, linestyle=":", zorder=1)

    ax.annotate("beyond the training data\n-- the formula generalizes",
                xy=(2000, predicted), xytext=(1430, 2.05), fontsize=9, color=GOOD,
                ha="center", arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.5))
    ax.text(520, 6.30, "the points sit almost exactly on the line --\nthe errors here are tiny, which is why J is near zero",
            fontsize=8.5, color=GREY, style="italic")

    ax.set_xlim(400, 2150)
    ax.set_ylim(0, 7.4)
    ax.set_xlabel("Area (sq.ft)", fontsize=10, color=INK)
    ax.set_ylabel("Price (Lakhs)", fontsize=10, color=INK)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("The Best-Fit Line: Predicting House Price from Area",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig


@figure("cost_function")
def _cost_function() -> plt.Figure:
    """Three candidate lines, and why the middle one wins."""
    candidates = [
        ("Too steep", 0.0050, -0.85, WARN),
        ("Best fit — lowest J", B1, B0, GOOD),
        ("Too shallow", 0.0018, 1.35, WARN),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), facecolor=PAPER)
    grid = np.linspace(450, 1550, 100)

    for ax, (title, slope, intercept, colour) in zip(axes, candidates):
        fitted = slope * AREA + intercept
        cost = float(np.mean((PRICE - fitted) ** 2))

        ax.plot(grid, slope * grid + intercept, color=colour, linewidth=2.3, zorder=2)
        for a, p, f in zip(AREA, PRICE, fitted):
            ax.plot([a, a], [p, f], color=GREY, linewidth=1.6, linestyle="--", zorder=1)
        ax.scatter(AREA, PRICE, s=95, color=BLUE, zorder=3)

        panel_title(ax, title, f"J = {cost:.4f}")
        ax.set_xlim(430, 1570)
        ax.set_ylim(0, 7.0)
        ax.set_xticks([500, 1000, 1500])
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Area (sq.ft)", fontsize=9, color=INK)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    axes[0].set_ylabel("Price (Lakhs)", fontsize=9, color=INK)
    fig.suptitle("The Cost Function: J = average of (Actual − Predicted)²  —  smallest J wins",
                 fontsize=14, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig


@figure("assumptions_grid")
def _assumptions_grid() -> plt.Figure:
    """Good-versus-bad pairs for four of the six assumptions."""
    rng = np.random.default_rng(11)
    fig, axes = plt.subplots(2, 4, figsize=(15.0, 7.0), facecolor=PAPER)

    x = np.linspace(0.4, 4.6, 26)

    # --- Column 1: linearity -------------------------------------------------
    axes[0, 0].scatter(x, 0.85 * x + 0.9 + rng.normal(0, 0.16, x.size), s=32, color=BLUE)
    axes[0, 0].plot([0.3, 4.7], [1.15, 4.9], color=GOOD, linewidth=2)
    curved = 4.3 - 0.72 * (x - 2.5) ** 2 + rng.normal(0, 0.13, x.size)
    axes[1, 0].scatter(x, curved, s=32, color=BLUE)
    axes[1, 0].plot([0.3, 4.7], [1.6, 2.3], color=WARN, linewidth=2)

    # --- Column 2: normality of residuals -----------------------------------
    bins = np.linspace(-3, 3, 19)
    axes[0, 1].hist(rng.normal(0, 1, 4000), bins=bins, color=SKY, edgecolor=NAVY, linewidth=0.6)
    axes[1, 1].hist(rng.exponential(1, 4000) - 1, bins=bins, color="#F5B7B1",
                    edgecolor=WARN, linewidth=0.6)

    # --- Column 3: independence ---------------------------------------------
    axes[0, 2].scatter(x, rng.normal(0, 0.55, x.size), s=32, color=BLUE)
    axes[0, 2].axhline(0, color=GREY, linewidth=1)
    axes[1, 2].scatter(x, 1.25 * np.sin(2.4 * x), s=32, color=BLUE)
    axes[1, 2].plot(np.linspace(0.4, 4.6, 200), 1.25 * np.sin(2.4 * np.linspace(0.4, 4.6, 200)),
                    color=WARN, linewidth=1.6, alpha=0.65)
    axes[1, 2].axhline(0, color=GREY, linewidth=1)

    # --- Column 4: homoscedasticity -----------------------------------------
    axes[0, 3].scatter(x, rng.normal(0, 0.5, x.size), s=32, color=BLUE)
    axes[0, 3].axhline(0, color=GREY, linewidth=1)
    axes[1, 3].scatter(x, rng.normal(0, 1, x.size) * (0.12 + 0.42 * x), s=32, color=BLUE)
    axes[1, 3].axhline(0, color=GREY, linewidth=1)
    for sign in (1, -1):
        axes[1, 3].plot([0.4, 4.6], [sign * 0.35, sign * 2.6], color=WARN,
                        linewidth=1.5, linestyle="--")

    headers = [
        ("LINEARITY", "is the relationship a straight line?"),
        ("NORMALITY", "are the residuals bell-shaped?"),
        ("INDEPENDENCE", "are the errors unrelated?"),
        ("HOMOSCEDASTICITY", "is the error spread constant?"),
    ]
    verdicts = [
        ("linear — good fit", "non-linear — poor fit"),
        ("normally distributed", "not normally distributed"),
        ("no pattern — independent", "visible pattern — dependent"),
        ("constant spread", "widening spread — a cone"),
    ]

    for col, ((title, subtitle), (good, bad)) in enumerate(zip(headers, verdicts)):
        axes[0, col].set_title(f"{title}\n{subtitle}", fontsize=10.5, fontweight="bold",
                               color=NAVY, pad=10)
        axes[0, col].set_xlabel(f" {good}", fontsize=9, color=GOOD, fontweight="bold", labelpad=6)
        axes[1, col].set_xlabel(f" {bad}", fontsize=9, color=WARN, fontweight="bold", labelpad=6)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_edgecolor(LINE)

    fig.suptitle("Four Assumptions, Side by Side — Top Row Passes, Bottom Row Fails",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.00)
    fig.tight_layout()
    return fig


@figure("residuals_vs_fitted")
def _residuals_vs_fitted() -> plt.Figure:
    """The four residual patterns and what each one diagnoses."""
    rng = np.random.default_rng(3)
    fitted = np.linspace(0.5, 4.5, 40)

    panels = [
        ("Healthy", "random, even band", rng.normal(0, 0.5, fitted.size), GOOD,
         "all three assumptions look fine"),
        ("Curve", "an arc through the middle", 1.5 - 0.85 * (fitted - 2.5) ** 2 * 0.55, WARN,
         "LINEARITY violated"),
        ("Cone", "spread widens to the right", rng.normal(0, 1, fitted.size) * (0.08 + 0.26 * fitted), WARN,
         "HOMOSCEDASTICITY violated"),
        ("Wave", "a repeating ripple", 1.15 * np.sin(2.6 * fitted), WARN,
         "INDEPENDENCE violated"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 4.2), facecolor=PAPER)

    for ax, (title, subtitle, resid, colour, verdict) in zip(axes, panels):
        ax.axhline(0, color=NAVY, linewidth=1.4)
        ax.scatter(fitted, resid, s=34, color=BLUE, zorder=3)
        panel_title(ax, title, subtitle)
        ax.text(2.5, -2.62, verdict, fontsize=9.5, color=colour, ha="center", fontweight="bold")
        ax.set_xlim(0.2, 4.8)
        ax.set_ylim(-3.0, 2.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Fitted value →", fontsize=9, color=INK)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    axes[0].set_ylabel("Residual", fontsize=9, color=INK)
    fig.suptitle("Residuals vs Fitted — One Plot, Three Assumptions",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig


@figure("vif_scale")
def _vif_scale() -> plt.Figure:
    """VIF thresholds, plus the measured Auto MPG before/after result."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.4), facecolor=PAPER,
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # -- Left: the threshold scale -------------------------------------------
    ax = axes[0]
    bands = [
        (0.0, 1.6, "#EAF7EF", GOOD, "VIF = 1", "no correlation"),
        (1.6, 5.0, "#DFF0FB", BLUE, "1 < VIF < 5", "moderate — acceptable"),
        (5.0, 10.0, "#FDF2D9", "#B9770E", "5 <= VIF < 10", "high — investigate"),
        (10.0, 14.0, "#FDEDEC", WARN, "VIF >= 10", "severe — act now"),
    ]
    for lo, hi, face, edge, label, note in bands:
        ax.add_patch(plt.Rectangle((lo, 0.9), hi - lo, 1.0, facecolor=face,
                                   edgecolor=edge, linewidth=1.8))
        mid = (lo + hi) / 2
        ax.text(mid, 1.62, label, fontsize=9.5, fontweight="bold", color=edge, ha="center")
        ax.text(mid, 1.18, note, fontsize=8, color=INK, ha="center", style="italic")

    # The first band is drawn wider than its true range so the "VIF = 1" label
    # fits; only the real thresholds (1, 5, 10) get tick marks.
    for tick, pos in ((1, 1.6), (5, 5.0), (10, 10.0)):
        ax.plot([pos, pos], [0.62, 0.9], color=GREY, linewidth=1.2)
        ax.text(pos, 0.44, str(tick), fontsize=9, color=INK, ha="center")

    ax.text(7.0, 0.06, "Tolerance = 1 / VIF   ·   tolerance below 0.1  =  VIF above 10",
            fontsize=9, color=INK, ha="center", style="italic")
    ax.set_xlim(-0.4, 14.4)
    ax.set_ylim(0, 2.3)
    ax.axis("off")
    ax.set_title("Variance Inflation Factor thresholds", fontsize=12.5,
                 fontweight="bold", color=NAVY, pad=12)

    # -- Right: what actually happened on Auto MPG ---------------------------
    ax = axes[1]
    labels = ["Original\n(VIF flagged)", "Drop ALL four\nhigh-VIF", "Keep one\n(weight)"]
    values = [2.89, 4.44, 2.95]
    colours = [BLUE, WARN, GOOD]
    ax.bar(range(3), values, 0.55, color=colours)
    for i, v in enumerate(values):
        ax.text(i, v + 0.13, f"{v:.2f}", fontsize=13, fontweight="bold",
                color=colours[i], ha="center")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("RMSE (lower is better)", fontsize=9.5, color=INK)
    ax.set_ylim(0, 5.4)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("Auto MPG: don't blindly drop flagged features",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=12)

    fig.suptitle("Multicollinearity: Measuring VIF, and Measuring the Fix",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.03)
    fig.tight_layout()
    return fig


@figure("hypothesis_testing")
def _hypothesis_testing() -> plt.Figure:
    """The p-value decision rule, and coefficients with confidence intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), facecolor=PAPER)

    # -- Left: the decision tree ---------------------------------------------
    ax = axes[0]
    ax.add_patch(plt.Rectangle((1.15, 4.05), 3.7, 0.85, facecolor="#CFE0FF",
                               edgecolor=NAVY, linewidth=1.8))
    ax.text(3.0, 4.48, "H₀ : β₁ = 0\n(variable not significant)", fontsize=10,
            fontweight="bold", color=NAVY, ha="center", va="center")

    branches = [
        (1.35, "p > 0.05", "Fail to Reject H₀", "Variable is\nINSIGNIFICANT", WARN, "#FDEDEC"),
        (4.65, "p < 0.05", "Reject H₀", "Variable is\nSIGNIFICANT", GOOD, "#EAF7EF"),
    ]
    for cx, ptext, verdict, meaning, colour, face in branches:
        ax.annotate("", xy=(cx, 3.25), xytext=(3.0, 4.05),
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.8))
        ax.text(cx, 3.02, ptext, fontsize=10.5, fontweight="bold", color=colour, ha="center")
        ax.add_patch(plt.Rectangle((cx - 1.18, 1.55), 2.36, 1.18, facecolor=face,
                                   edgecolor=colour, linewidth=1.6))
        ax.text(cx, 2.42, verdict, fontsize=9.5, fontweight="bold", color=colour, ha="center")
        ax.text(cx, 1.95, meaning, fontsize=9, color=INK, ha="center", va="center")

    ax.text(3.0, 0.85, "0.05 is a convention, not a law of nature",
            fontsize=9, color=GREY, ha="center", style="italic")
    ax.set_xlim(0, 6)
    ax.set_ylim(0.4, 5.3)
    ax.axis("off")
    ax.set_title("The p-value decision rule", fontsize=12.5, fontweight="bold", color=NAVY, pad=10)

    # -- Right: coefficients with 95% confidence intervals -------------------
    ax = axes[1]
    coefs = [
        ("age", -0.800, -2.226, 0.626, 0.240),
        ("bedrooms", 2.510, 0.482, 4.538, 0.022),
        ("area (x1000)", 3.300, 3.000, 4.000, 0.000),
    ]
    for i, (name, coef, lo, hi, p) in enumerate(coefs):
        crosses_zero = lo < 0 < hi
        colour = WARN if crosses_zero else GOOD
        ax.plot([lo, hi], [i, i], color=colour, linewidth=3.0, solid_capstyle="round")
        ax.plot([lo, lo], [i - 0.12, i + 0.12], color=colour, linewidth=2)
        ax.plot([hi, hi], [i - 0.12, i + 0.12], color=colour, linewidth=2)
        ax.scatter([coef], [i], s=95, color=colour, zorder=3)
        ax.text(hi + 0.28, i, f"p = {p:.3f}", fontsize=9, color=colour,
                va="center", fontweight="bold")

    ax.axvline(0, color=NAVY, linewidth=1.8, linestyle="--")
    ax.text(0, 2.78, "zero", fontsize=9, color=NAVY, ha="center", fontweight="bold")
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels([c[0] for c in coefs], fontsize=10)
    ax.set_xlim(-3.4, 6.9)
    ax.set_ylim(-0.6, 3.15)
    ax.set_xlabel("Coefficient value with 95% confidence interval", fontsize=9.5, color=INK)
    ax.tick_params(axis="x", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("If the interval crosses zero, it is not significant",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=10)

    fig.suptitle("Hypothesis Testing: Is the Coefficient Real?",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.03)
    fig.tight_layout()
    return fig
