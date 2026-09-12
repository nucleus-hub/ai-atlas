"""Figures for note 06 -- Overfitting, Underfitting & Bias-Variance.

The polynomial fits, the complexity curve and the empirical bias-variance
decomposition are all computed live. The learning curves are reproduced from
the Wine dataset with the practical's pipeline (random_state=42), shuffled
first because ``load_wine`` returns rows sorted by class.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from .core import (
    BLUE,
    GOOD,
    GREY,
    INK,
    LINE,
    MIST,
    NAVY,
    PAPER,
    SKY,
    WARN,
    figure,
    panel_title,
)

TRUE_FUNCTION = lambda x: np.sin(1.5 * np.pi * x)  # noqa: E731


def _sample(seed: int = 3, n: int = 12, noise: float = 0.2):
    """A small noisy sample from the true curve -- small enough to overfit."""
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 1, n))
    return x, TRUE_FUNCTION(x) + rng.normal(0, noise, n)


def _fit(x, y, degree: int):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression()).fit(x[:, None], y)


@figure("underfit_goodfit_overfit")
def underfit_goodfit_overfit() -> plt.Figure:
    """The three-panel diagnosis from slide 63, fitted for real."""
    x, y = _sample()
    grid = np.linspace(0, 1, 300)

    panels = [
        (1, "Underfitting", "High Bias -- model too simple", WARN),
        (3, "Good Fit", "Captures the underlying pattern", GOOD),
        (11, "Overfitting", "High Variance -- learns the noise", WARN),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3), sharey=True)

    for ax, (degree, title, subtitle, colour) in zip(axes, panels):
        model = _fit(x, y, degree)
        ax.plot(grid, TRUE_FUNCTION(grid), color=GREY, linewidth=2,
                linestyle="--", label="True pattern", zorder=2)
        ax.plot(grid, model.predict(grid[:, None]), color=colour, linewidth=2.6,
                label=f"Model (degree {degree})", zorder=3)
        ax.scatter(x, y, color=NAVY, s=52, edgecolor=PAPER, linewidth=1.2,
                   zorder=5, label="Training data")

        train_mse = np.mean((y - model.predict(x[:, None])) ** 2)
        test_mse = np.mean((TRUE_FUNCTION(grid) - model.predict(grid[:, None])) ** 2)
        ax.text(
            0.5, -1.78,
            f"train MSE {train_mse:.3f}   |   test MSE {test_mse:.3f}",
            ha="center", fontsize=9, color=INK, fontweight="bold",
            bbox=dict(facecolor=MIST if colour is GOOD else "#FBE9E7",
                      edgecolor=colour, boxstyle="round,pad=0.3"),
        )

        ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, fontweight="bold", color=colour)
        ax.set_ylim(-2.1, 2.1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2, color=LINE)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Same 12 data points, three model complexities",
        fontsize=12.5, fontweight="bold", color=NAVY,
    )
    fig.tight_layout()
    return fig


@figure("complexity_curve")
def complexity_curve() -> plt.Figure:
    """Training error always falls; test error turns back up."""
    x, y = _sample()
    grid = np.linspace(0, 1, 300)
    truth = TRUE_FUNCTION(grid)

    degrees = list(range(1, 12))
    train_errors, test_errors = [], []
    for degree in degrees:
        model = _fit(x, y, degree)
        train_errors.append(np.mean((y - model.predict(x[:, None])) ** 2))
        test_errors.append(np.mean((truth - model.predict(grid[:, None])) ** 2))

    fig, ax = plt.subplots(figsize=(8, 4.8))

    ax.plot(degrees, train_errors, marker="o", color=BLUE, linewidth=2.2,
            markersize=7, label="Training error -- always decreases")
    ax.plot(degrees, test_errors, marker="s", color=WARN, linewidth=2.2,
            markersize=7, label="Test error -- falls, then rises")

    best = int(np.argmin(test_errors))
    ax.scatter([degrees[best]], [test_errors[best]], s=280, facecolor="none",
               edgecolor=GOOD, linewidth=2.5, zorder=6)
    ax.annotate(
        f"Sweet spot: degree {degrees[best]}\ntest MSE {test_errors[best]:.3f}",
        xy=(degrees[best], test_errors[best]),
        xytext=(degrees[best] + 1.4, 0.55),
        fontsize=9.5, color=GOOD, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GOOD, linewidth=1.5),
    )

    ax.axvspan(0.5, degrees[best] - 0.5, color="#FBE9E7", alpha=0.55, zorder=0)
    ax.axvspan(degrees[best] + 0.5, 11.5, color="#FEF5E7", alpha=0.55, zorder=0)
    ax.text(1.6, 8, "UNDERFITTING\nhigh bias", fontsize=9, color=WARN,
            ha="center", fontweight="bold")
    ax.text(9.0, 8, "OVERFITTING\nhigh variance", fontsize=9, color="#B9770E",
            ha="center", fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlabel("Model complexity  (polynomial degree)", fontsize=10, color=INK)
    ax.set_ylabel("Mean Squared Error  (log scale)", fontsize=10, color=INK)
    ax.set_xticks(degrees)
    ax.set_xlim(0.5, 11.5)
    ax.grid(alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "The Complexity Curve",
                "The gap between the two lines IS the variance")
    fig.tight_layout()
    return fig


# -- Empirical decomposition: 200 resampled training sets per degree ------
BV_DEGREES = [1, 2, 3, 4, 5, 7, 9]
BV_BIAS2 = [0.1654, 0.0272, 0.0026, 0.0004, 0.0004, 0.0119, 0.0244]
BV_VARIANCE = [0.0359, 0.0228, 0.0299, 0.0584, 0.4825, 1.4005, 36.6429]
BV_IRREDUCIBLE = 0.0625


@figure("bias_variance_tradeoff")
def bias_variance_tradeoff() -> plt.Figure:
    """The classic decomposition, measured rather than sketched."""
    fig, ax = plt.subplots(figsize=(8.4, 4.8))

    bias2 = np.array(BV_BIAS2)
    variance = np.array(BV_VARIANCE)
    total = bias2 + variance + BV_IRREDUCIBLE
    x = np.arange(len(BV_DEGREES))

    ax.plot(x, bias2, marker="o", color=BLUE, linewidth=2.2, label="Bias$^2$ -- falls with complexity")
    ax.plot(x, variance, marker="s", color=WARN, linewidth=2.2, label="Variance -- rises with complexity")
    ax.axhline(BV_IRREDUCIBLE, color=GREY, linestyle=":", linewidth=1.8,
               label=f"Irreducible error = {BV_IRREDUCIBLE:.4f}")
    ax.plot(x, total, marker="D", color=NAVY, linewidth=2.6, label="Total error = sum of all three")

    best = int(np.argmin(total))
    ax.scatter([x[best]], [total[best]], s=300, facecolor="none",
               edgecolor=GOOD, linewidth=2.5, zorder=6)
    ax.annotate(
        f"Minimum total error\ndegree {BV_DEGREES[best]}",
        xy=(x[best], total[best]),
        xytext=(x[best] + 0.55, 0.6),
        fontsize=9.5, color=GOOD, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=GOOD, linewidth=1.5),
    )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(BV_DEGREES)
    ax.set_xlabel("Model complexity  (polynomial degree)", fontsize=10, color=INK)
    ax.set_ylabel("Error contribution  (log scale)", fontsize=10, color=INK)
    ax.grid(alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(
        ax,
        "Total Error = Bias$^2$ + Variance + Irreducible Error",
        "Measured over 200 resampled training sets -- not a sketch",
    )
    fig.tight_layout()
    return fig


@figure("variance_illustration")
def variance_illustration() -> plt.Figure:
    """What high bias and high variance look like across resampled datasets."""
    rng = np.random.default_rng(11)
    grid = np.linspace(0, 1, 200)
    truth = TRUE_FUNCTION(grid)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)

    for ax, degree, title, colour in (
        (axes[0], 1, "Degree 1 -- HIGH BIAS, low variance", WARN),
        (axes[1], 11, "Degree 11 -- low bias, HIGH VARIANCE", "#B9770E"),
    ):
        fits = []
        for _ in range(14):
            xs = np.sort(rng.uniform(0, 1, 12))
            ys = TRUE_FUNCTION(xs) + rng.normal(0, 0.2, 12)
            fits.append(_fit(xs, ys, degree).predict(grid[:, None]))
            ax.plot(grid, fits[-1], color=colour, linewidth=0.9, alpha=0.45, zorder=2)

        ax.plot(grid, np.mean(fits, axis=0), color=colour, linewidth=2.8,
                zorder=4, label="Average of the 14 fits")
        ax.plot(grid, truth, color=NAVY, linewidth=2.4, linestyle="--",
                zorder=5, label="True pattern")

        ax.set_title(title, fontsize=10.5, fontweight="bold", color=colour)
        ax.set_ylim(-2.6, 2.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2, color=LINE)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    axes[0].text(0.5, -2.35, "Every fit is similar -- and all of them are wrong",
                 ha="center", fontsize=9, color=INK, style="italic")
    axes[1].text(0.5, -2.35, "The average is right -- but no single fit is",
                 ha="center", fontsize=9, color=INK, style="italic")

    fig.suptitle(
        "14 models, each trained on a different sample of the same problem",
        fontsize=12, fontweight="bold", color=NAVY,
    )
    fig.tight_layout()
    return fig


# -- Wine learning curves (shuffled first; load_wine is sorted by class) --
LC_SIZES = [21, 45, 69, 93, 117, 142]
LC_OVERFIT_TRAIN = [1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]
LC_OVERFIT_CV = [0.7924, 0.7757, 0.8368, 0.8822, 0.9275, 0.9214]
LC_UNDERFIT_TRAIN = [0.7524, 0.7378, 0.7275, 0.6946, 0.6889, 0.6873]
LC_UNDERFIT_CV = [0.6686, 0.6686, 0.6570, 0.6059, 0.6057, 0.5835]


@figure("learning_curves")
def learning_curves() -> plt.Figure:
    """Does more data help? Only if the problem is variance."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    for ax, train, cv, title, colour, verdict in (
        (axes[0], LC_OVERFIT_TRAIN, LC_OVERFIT_CV,
         "Unpruned tree (overfitting)", WARN,
         "Gap closes 0.208 -> 0.079\nMORE DATA HELPS"),
        (axes[1], LC_UNDERFIT_TRAIN, LC_UNDERFIT_CV,
         "Depth-1 stump (underfitting)", "#B9770E",
         "Both lines stay low\nMORE DATA WON'T HELP"),
    ):
        ax.plot(LC_SIZES, train, marker="o", color=BLUE, linewidth=2.2, label="Training accuracy")
        ax.plot(LC_SIZES, cv, marker="s", color=colour, linewidth=2.2, label="Cross-validated accuracy")
        ax.fill_between(LC_SIZES, cv, train, color=MIST, alpha=0.45)

        mid = len(LC_SIZES) // 2
        ax.annotate(
            "the gap\n= variance",
            xy=(LC_SIZES[mid], (train[mid] + cv[mid]) / 2),
            xytext=(LC_SIZES[mid] - 40, 0.42),
            fontsize=8.5, color=GREY,
            arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.1),
        )
        ax.text(0.70, 0.055, verdict, transform=ax.transAxes, ha="center",
                fontsize=9.5, fontweight="bold", color=colour,
                bbox=dict(facecolor=PAPER, edgecolor=colour, boxstyle="round,pad=0.35"))

        ax.set_title(title, fontsize=10.5, fontweight="bold", color=colour)
        ax.set_xlabel("Training set size", fontsize=9.5, color=INK)
        ax.set_ylim(0.33, 1.16)
        ax.grid(alpha=0.25, color=LINE)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, 1.005))
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    axes[0].set_ylabel("Accuracy", fontsize=9.5, color=INK)
    fig.suptitle(
        'Learning curves diagnose WHICH problem you have (Wine dataset)',
        fontsize=12, fontweight="bold", color=NAVY,
    )
    fig.tight_layout()
    return fig
