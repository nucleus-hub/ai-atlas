"""Figures for Topic 06 -- Overfitting, Underfitting & the Bias-Variance Trade-off."""

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


def _sample(seed: int = 6, n: int = 22) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    y = np.sin(2.2 * np.pi * x) * 0.85 + rng.normal(0, 0.26, n)
    return x, y


@figure("underfit_goodfit_overfit")
def underfit_goodfit_overfit() -> plt.Figure:
    """The canonical three-panel fit comparison."""
    x, y = _sample()
    grid = np.linspace(0, 1, 300)

    panels = [
        (1, "Underfitting (High Bias)", "too simple -- misses the pattern", WARN),
        (4, "Good Fit (Just Right)", "captures the real signal", GOOD),
        (18, "Overfitting (High Variance)", "learns the noise as if it mattered", WARN),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    for ax, (degree, title, sub, colour) in zip(axes, panels):
        coeffs = np.polyfit(x, y, degree)
        ax.scatter(x, y, s=30, color=NAVY, zorder=4, alpha=0.75)
        ax.plot(grid, np.polyval(coeffs, grid), color=colour, linewidth=2.4)
        ax.set_title(f"{title}\n{sub}", fontsize=10, color=NAVY, fontweight="bold")
        ax.set_ylim(-2.0, 2.0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    axes[0].text(0.5, -1.75, "high train error\nhigh test error", ha="center", fontsize=8.5, color=WARN)
    axes[1].text(0.5, -1.75, "low train error\nlow test error", ha="center", fontsize=8.5, color=GOOD)
    axes[2].text(0.5, -1.75, "very low train error\nhigh test error", ha="center", fontsize=8.5, color=WARN)
    fig.tight_layout()
    return fig


@figure("bias_variance_tradeoff")
def bias_variance_tradeoff() -> plt.Figure:
    """Total error decomposed into bias squared, variance and irreducible error."""
    complexity = np.linspace(0.3, 10, 400)
    bias_sq = 3.4 * np.exp(-0.55 * complexity)
    variance = 0.045 * complexity**1.85
    irreducible = np.full_like(complexity, 0.45)
    total = bias_sq + variance + irreducible
    best = complexity[int(np.argmin(total))]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(complexity, bias_sq, color=BLUE, linewidth=2.0, linestyle="--", label="Bias$^2$")
    ax.plot(complexity, variance, color="#E07A3F", linewidth=2.0, linestyle="--", label="Variance")
    ax.plot(complexity, irreducible, color=GREY, linewidth=1.6, linestyle=":", label="Irreducible error")
    ax.plot(complexity, total, color=NAVY, linewidth=2.8, label="Total error")

    ax.axvline(best, color=GOOD, linewidth=1.6)
    ax.text(best + 0.15, 3.4, "sweet spot", fontsize=9, color=GOOD, fontweight="bold")
    ax.text(1.0, 3.5, "UNDERFITTING\nhigh bias", fontsize=9, color=WARN, ha="center")
    ax.text(8.8, 3.5, "OVERFITTING\nhigh variance", fontsize=9, color=WARN, ha="center")

    ax.set_xlabel("Model complexity", fontsize=10, color=INK)
    ax.set_ylabel("Error", fontsize=10, color=INK)
    ax.set_ylim(0, 4.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Total Error = Bias$^2$ + Variance + Irreducible Error",
                 fontsize=12, color=NAVY, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper center", ncol=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


@figure("training_validation_curves")
def training_validation_curves() -> plt.Figure:
    """How to actually detect each scenario from two error curves."""
    complexity = np.linspace(0.3, 10, 300)
    train = 2.9 * np.exp(-0.52 * complexity) + 0.12
    validation = 2.9 * np.exp(-0.52 * complexity) + 0.12 + 0.042 * complexity**1.9
    best = complexity[int(np.argmin(validation))]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(complexity, train, color=BLUE, linewidth=2.6, label="Training error")
    ax.plot(complexity, validation, color=WARN, linewidth=2.6, label="Validation error")
    ax.axvline(best, color=GOOD, linestyle="--", linewidth=1.6)

    ax.axvspan(0.3, best - 2.2, color="#FDEBD0", alpha=0.55)
    ax.axvspan(best + 1.6, 10, color="#FBE3E0", alpha=0.55)
    ax.text(1.3, 2.6, "UNDERFIT\nboth errors high", fontsize=9, color=INK, ha="center")
    ax.text(best, 2.9, "stop here", fontsize=9, color=GOOD, ha="center", fontweight="bold")
    ax.text(8.4, 2.6, "OVERFIT\ntrain low, validation rising", fontsize=9, color=INK, ha="center")

    ax.set_xlabel("Model complexity", fontsize=10, color=INK)
    ax.set_ylabel("Error", fontsize=10, color=INK)
    ax.set_ylim(0, 3.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Diagnosing the Problem From Two Curves", fontsize=12, color=NAVY, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="center right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


@figure("variance_from_resampling")
def variance_from_resampling() -> plt.Figure:
    """Why it is called variance: refit a complex model on different samples."""
    grid = np.linspace(0, 1, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.9))

    for seed in range(6):
        x, y = _sample(seed=seed + 30, n=18)
        ax1.plot(grid, np.polyval(np.polyfit(x, y, 1), grid), color=BLUE, alpha=0.65, linewidth=1.7)
        ax2.plot(grid, np.polyval(np.polyfit(x, y, 14), grid), color=WARN, alpha=0.65, linewidth=1.7)

    truth = np.sin(2.2 * np.pi * grid) * 0.85
    for ax, title in ((ax1, "Simple model: high bias, low variance"),
                      (ax2, "Complex model: low bias, high variance")):
        ax.plot(grid, truth, color=NAVY, linewidth=2.4, linestyle="--", label="true pattern")
        ax.set_ylim(-2.2, 2.2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10.5, color=NAVY, fontweight="bold")
        ax.legend(fontsize=8, frameon=False, loc="lower center")
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    ax1.text(0.5, 1.85, "six refits, nearly identical\nand consistently wrong",
             ha="center", fontsize=8.5, color=GREY)
    ax2.text(0.5, 1.85, "six refits, wildly different\nfrom each other", ha="center", fontsize=8.5, color=GREY)
    fig.suptitle("Each line is the same model refit on a different data sample",
                 fontsize=11.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("fixing_overfitting")
def fixing_overfitting() -> plt.Figure:
    """The six practical remedies, mapped to the problem each one treats."""
    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    blank_axes(ax, (0, 10), (0, 4.6))
    panel_title(ax, "How to Fix It", "Six practical remedies from the session")

    fixes = [
        (1.8, 3.3, "More Data", "reduces variance"),
        (5.0, 3.3, "Simplify Model", "less depth, fewer features"),
        (8.2, 3.3, "Regularization", "penalise large weights"),
        (1.8, 1.5, "Feature Engineering", "relevant features, less noise"),
        (5.0, 1.5, "Cross-Validation", "validate model stability"),
        (8.2, 1.5, "Early Stopping", "stop when validation worsens"),
    ]
    for x, y, name, sub in fixes:
        node_box(ax, (x, y), name, facecolor=MIST, width=2.8, height=0.6, fontsize=9.5, bold=True)
        ax.text(x, y - 0.58, sub, ha="center", fontsize=8, color=GREY)

    ax.text(5.0, 0.25, "Optimization techniques:  Grid Search  |  Random Search  |  "
                       "Bayesian Optimization  |  Gradient Descent",
            ha="center", fontsize=8.5, color=NAVY,
            bbox=dict(facecolor="#F2F6FD", edgecolor=LINE, boxstyle="round,pad=0.4"))
    fig.tight_layout()
    return fig
