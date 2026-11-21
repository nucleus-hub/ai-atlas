"""Figures for Topic 07 -- Regularization & Cross-Validation."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from .core import (
    BLUE,
    GOOD,
    GREY,
    INK,
    LINE,
    MIST,
    NAVY,
    PAPER,
    WARN,
    blank_axes,
    figure,
    node_box,
    panel_title,
)


@figure("regularization_penalty")
def regularization_penalty() -> plt.Figure:
    """The regularised objective, split into its two competing jobs."""
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    blank_axes(ax, (0, 10), (0, 3.6))
    panel_title(ax, "The Regularized Objective", "J_reg(theta) = J(theta) + lambda * Omega(theta)")

    node_box(ax, (2.1, 2.3), "J(theta)\noriginal loss", facecolor=MIST, width=2.6, height=0.9, fontsize=9.5)
    ax.text(3.8, 2.3, "+", ha="center", va="center", fontsize=18, color=INK, fontweight="bold")
    node_box(ax, (5.6, 2.3), "lambda", facecolor="#FDE9D9", width=1.3, height=0.9, fontsize=9.5)
    ax.text(6.6, 2.3, "x", ha="center", va="center", fontsize=14, color=INK)
    node_box(ax, (8.1, 2.3), "Omega(theta)\npenalty term", facecolor="#CDEBD6", edgecolor=GOOD,
             width=2.6, height=0.9, fontsize=9.5)

    ax.text(2.1, 1.35, "fit the training data", ha="center", fontsize=8.5, color=GREY)
    ax.text(5.6, 1.35, "strength dial\nlambda = 0 -> no regularization", ha="center", fontsize=8.5, color=GREY)
    ax.text(8.1, 1.35, "punish complexity", ha="center", fontsize=8.5, color=GREY)
    ax.text(5.0, 0.42, "Two jobs pulling against each other -- lambda decides who wins",
            ha="center", fontsize=9, color=NAVY, style="italic")
    fig.tight_layout()
    return fig


@figure("l1_vs_l2")
def l1_vs_l2() -> plt.Figure:
    """Why L1 lands on zero and L2 merely shrinks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.2))

    # Constraint-region intuition.
    t = np.linspace(0, 2 * np.pi, 400)
    ax1.plot(np.cos(t), np.sin(t), color=BLUE, linewidth=2.2, label="L2: circle")
    ax1.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], color="#E07A3F", linewidth=2.2, label="L1: diamond")
    for radius, alpha in ((0.55, 0.9), (0.8, 0.6), (1.05, 0.35)):
        ax1.add_patch(plt.Circle((1.75, 1.15), radius, fill=False, edgecolor=GREY, alpha=alpha, linewidth=1.1))
    ax1.scatter([0], [1], s=90, color="#E07A3F", zorder=6)
    ax1.annotate("L1 touches at a CORNER\n-> theta_1 is exactly 0", xy=(0, 1), xytext=(-2.5, 1.55),
                 fontsize=8.5, color="#E07A3F", arrowprops=dict(arrowstyle="->", color="#E07A3F", linewidth=1.2))
    ax1.scatter([0.63], [0.78], s=90, color=BLUE, zorder=6)
    ax1.annotate("L2 touches on the curve\n-> both shrink, neither vanishes",
                 xy=(0.63, 0.78), xytext=(-2.6, -1.5),
                 fontsize=8.5, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.2))
    ax1.set_xlim(-3.0, 3.0)
    ax1.set_ylim(-2.0, 2.4)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Why L1 produces exact zeros", fontsize=10.5, color=NAVY, fontweight="bold")
    ax1.legend(fontsize=8, frameon=False, loc="lower right")

    # Coefficient profile.
    names = [f"f{i}" for i in range(1, 9)]
    plain = np.array([2.4, -1.9, 0.9, 1.6, -0.6, 0.35, -1.2, 0.75])
    lasso = np.array([1.9, -1.3, 0.0, 1.0, 0.0, 0.0, -0.55, 0.0])
    ridge = plain * 0.55

    y = np.arange(len(names))
    ax2.barh(y + 0.26, plain, 0.24, color=GREY, label="No regularization")
    ax2.barh(y, ridge, 0.24, color=BLUE, label="L2 (Ridge): shrunk")
    ax2.barh(y - 0.26, lasso, 0.24, color="#E07A3F", label="L1 (Lasso): some exactly 0")
    ax2.axvline(0, color=INK, linewidth=1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=8.5)
    ax2.set_xlabel("Coefficient value", fontsize=9.5, color=INK)
    ax2.set_title("Effect on the coefficients", fontsize=10.5, color=NAVY, fontweight="bold")
    ax2.legend(fontsize=8, frameon=False, loc="lower right")
    ax2.grid(axis="x", color=LINE, linewidth=0.7)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    for spine in ax1.spines.values():
        spine.set_edgecolor(LINE)
    fig.tight_layout()
    return fig


@figure("lambda_effect")
def lambda_effect() -> plt.Figure:
    """Small, moderate and large lambda, as fits and as error curves."""
    rng = np.random.default_rng(6)
    x = np.sort(rng.uniform(0, 1, 20))
    y = np.sin(2.2 * np.pi * x) * 0.85 + rng.normal(0, 0.24, 20)
    grid = np.linspace(0, 1, 300)

    fig, axes = plt.subplots(1, 4, figsize=(12.6, 3.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.3]})

    settings = [
        (16, "Small lambda (~0)", "overfits -- high variance", WARN),
        (5, "Moderate lambda", "best balance", GOOD),
        (1, "Large lambda", "underfits -- high bias", WARN),
    ]
    for ax, (degree, title, sub, colour) in zip(axes[:3], settings):
        ax.scatter(x, y, s=22, color=NAVY, alpha=0.75, zorder=4)
        ax.plot(grid, np.polyval(np.polyfit(x, y, degree), grid), color=colour, linewidth=2.3)
        ax.set_title(f"{title}\n{sub}", fontsize=9.5, color=NAVY, fontweight="bold")
        ax.set_ylim(-2.0, 2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    ax = axes[3]
    lam = np.logspace(-3, 2, 200)
    train = 0.12 + 0.55 * lam / (lam + 0.8)
    val = 0.12 + 0.55 * lam / (lam + 0.8) + 0.5 / (1 + 22 * lam)
    ax.semilogx(lam, train, color=BLUE, linewidth=2.2, label="Training error")
    ax.semilogx(lam, val, color=WARN, linewidth=2.2, label="Validation error")
    ax.axvline(lam[int(np.argmin(val))], color=GOOD, linestyle="--", linewidth=1.6)
    ax.text(lam[int(np.argmin(val))] * 1.25, 0.62, "best lambda", fontsize=8.5, color=GOOD)
    ax.set_xlabel("lambda (log scale)", fontsize=9, color=INK)
    ax.set_ylabel("Error", fontsize=9, color=INK)
    ax.set_yticks([])
    ax.set_title("Choosing lambda", fontsize=9.5, color=NAVY, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False)
    ax.grid(color=LINE, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    return fig


@figure("kfold_cross_validation")
def kfold_cross_validation() -> plt.Figure:
    """Five folds, each taking a turn as validation."""
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 6.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("5-Fold Cross-Validation", fontsize=12, color=NAVY, fontweight="bold")

    k = 5
    scores = [0.93, 0.89, 0.95, 0.91, 0.92]
    for row in range(k):
        y = 5.2 - row * 0.9
        ax.text(0.9, y, f"Round {row + 1}", ha="right", va="center", fontsize=9, color=INK)
        for fold in range(k):
            held_out = fold == row
            ax.add_patch(plt.Rectangle((1.2 + fold * 1.6, y - 0.3), 1.5, 0.6,
                                       facecolor="#F0A22E" if held_out else NAVY,
                                       edgecolor=PAPER, linewidth=1.6))
            ax.text(1.95 + fold * 1.6, y, "validate" if held_out else "train",
                    ha="center", va="center", fontsize=7.5,
                    color=INK if held_out else PAPER)
        ax.text(9.6, y, f"score {scores[row]:.2f}", va="center", fontsize=8.5, color=GREY)

    ax.text(5.2, 0.55, f"Average of the {k} scores = {np.mean(scores):.3f}  "
                       "-> one trustworthy estimate",
            ha="center", fontsize=9.5, color=NAVY, fontweight="bold",
            bbox=dict(facecolor="#F2F6FD", edgecolor=LINE, boxstyle="round,pad=0.45"))
    fig.tight_layout()
    return fig


@figure("regularization_methods_map")
def regularization_methods_map() -> plt.Figure:
    """Which regularization method to reach for."""
    fig, ax = plt.subplots(figsize=(9.8, 3.6))
    blank_axes(ax, (0, 10), (0, 4.2))
    panel_title(ax, "Choosing a Regularization Method", "")

    methods = [
        (1.7, "L1 (Lasso)", "SUM |theta|", "some become\nexactly 0", "feature selection\nneeded", "#FDE9D9"),
        (5.0, "L2 (Ridge)", "SUM theta^2", "shrink toward 0\nnever exactly", "keep all\nfeatures", "#D6EAF8"),
        (8.3, "Elastic Net", "|theta| + theta^2", "balance of\nboth", "correlated\nfeatures", "#CDEBD6"),
    ]
    for x, name, penalty, effect, when, colour in methods:
        node_box(ax, (x, 3.3), name, facecolor=colour, width=2.8, height=0.55, fontsize=10, bold=True)
        ax.text(x, 2.62, penalty, ha="center", fontsize=9, color=NAVY, fontweight="bold")
        ax.text(x, 1.95, effect, ha="center", fontsize=8.2, color=INK)
        ax.text(x, 1.15, f"use when:\n{when}", ha="center", fontsize=8.2, color=GREY)

    ax.text(5.0, 0.25, "Always standardize features first -- penalties compare coefficient sizes",
            ha="center", fontsize=8.5, color=WARN, style="italic")
    fig.tight_layout()
    return fig
