"""Figures for note 02 -- Random Forest & Ensembles.

Wine numbers are reproduced by re-running the practical notebook's exact
pipeline (``random_state=42``); the bootstrap/OOB figure is computed from the
(1 - 1/n)^n limit rather than quoting "~37%" blindly.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

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
    blank_axes,
    connect,
    figure,
    node_box,
    panel_title,
)


@figure("bagging_diagram")
def bagging_diagram() -> plt.Figure:
    """The Random Forest pipeline: bootstrap -> many trees -> vote."""
    fig, ax = plt.subplots(figsize=(10, 4.8))
    blank_axes(ax, xlim=(0, 12), ylim=(0, 5.6))

    node_box(
        ax,
        (1.4, 2.8),
        "Training\nData\n\nN samples\nM features",
        facecolor=MIST,
        edgecolor=NAVY,
        width=1.9,
        height=2.2,
        fontsize=9,
        bold=True,
    )

    sample_y = [4.4, 2.8, 1.2]
    for i, y in enumerate(sample_y, start=1):
        node_box(
            ax,
            (4.3, y),
            f"Bootstrap\nsample {i}",
            facecolor=PAPER,
            edgecolor=SKY,
            width=1.9,
            height=0.95,
            fontsize=8.5,
        )
        connect(ax, (2.4, 2.8), (3.32, y))

        node_box(ax, (7.2, y), f"Tree {i}", facecolor=PAPER, edgecolor=BLUE, width=1.6, height=0.8, fontsize=9)
        connect(ax, (5.28, y), (6.38, y))

        vote = "Yes" if i != 2 else "No"
        ax.text(
            8.45,
            y + 0.32,
            f'votes "{vote}"',
            fontsize=8.5,
            color=GREY,
            va="center",
            ha="center",
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5),
            zorder=6,
        )

    ax.text(4.3, 0.35, "sampling WITH replacement", ha="center", fontsize=8, color=GREY, style="italic")
    ax.text(7.2, 0.35, "random feature subset at each split", ha="center", fontsize=8, color=GREY, style="italic")

    node_box(
        ax,
        (10.9, 2.8),
        "Majority\nvote\n\nFinal:\n\"Yes\"",
        facecolor="#E4F3E9",
        edgecolor=GOOD,
        width=1.8,
        height=2.2,
        fontsize=9,
        bold=True,
    )
    for y in sample_y:
        connect(ax, (8.02, y), (9.98, 2.8))

    panel_title(
        ax,
        "Random Forest = Bagging + Random Feature Selection",
        "Many deliberately different trees, one aggregated answer",
    )
    fig.tight_layout()
    return fig


@figure("bootstrap_oob")
def bootstrap_oob() -> plt.Figure:
    """Why a bootstrap sample leaves out roughly 37% of the rows."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: the convergence of (1 - 1/n)^n to 1/e.
    ax = axes[0]
    n = np.arange(2, 200)
    oob = (1 - 1 / n) ** n
    ax.plot(n, oob * 100, color=BLUE, linewidth=2.2)
    ax.axhline(np.exp(-1) * 100, color=WARN, linestyle="--", linewidth=1.4)
    ax.text(120, np.exp(-1) * 100 + 1.4, "1/e  =  36.8%", fontsize=9, color=WARN, fontweight="bold")
    ax.set_xlabel("Number of rows in the dataset  (n)", fontsize=9.5, color=INK)
    ax.set_ylabel("Share of rows left out  (%)", fontsize=9.5, color=INK)
    ax.set_ylim(30, 42)
    ax.grid(alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("Each bootstrap sample misses ~37% of rows", fontsize=10.5, fontweight="bold", color=NAVY)

    # Right: the in-bag / out-of-bag split as a donut.
    ax = axes[1]
    wedges, _ = ax.pie(
        [63.2, 36.8],
        colors=[BLUE, MIST],
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor=PAPER, linewidth=2),
    )
    ax.text(0, 0.12, "~63%", ha="center", fontsize=15, fontweight="bold", color=NAVY)
    ax.text(0, -0.14, "in bag\n(trains the tree)", ha="center", fontsize=8.5, color=INK)
    ax.annotate(
        "~37% out-of-bag\nnever seen by this tree ->\nfree validation set",
        xy=(0.78, 0.62),
        xytext=(1.15, 1.02),
        fontsize=8.5,
        color=INK,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.1),
    )
    ax.set_title("One tree's view of the data", fontsize=10.5, fontweight="bold", color=NAVY)

    fig.suptitle(
        "Out-of-Bag samples: validation for free",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig


# -- Reproduced from the notebook pipeline (random_state=42, max_depth=6) --
N_TREES = [1, 5, 10, 25, 50, 100, 200]
N_TREES_ACCURACY = [0.8889, 0.9333, 0.9333, 1.0, 1.0, 1.0, 1.0]

FOREST_IMPORTANCE = {
    "color_intensity": 0.1390,
    "proline": 0.1307,
    "alcohol_flavanoid_ratio": 0.1284,
    "alcohol": 0.1240,
    "flavanoids": 0.1130,
    "hue": 0.0815,
    "od280/od315_of_diluted_wines": 0.0741,
    "color_hue_interaction": 0.0597,
}

# Single tree (max_depth=4) vs forest (200 trees, max_depth=6) on the same split.
TREE_IMPORTANCE = {
    "flavanoids": 0.4108,
    "color_intensity": 0.4033,
    "proline": 0.1003,
    "ash": 0.0436,
    "od280/od315_of_diluted_wines": 0.0224,
    "alcohol": 0.0196,
}


@figure("n_estimators_curve")
def n_estimators_curve() -> plt.Figure:
    """Test accuracy against forest size -- rises fast, then flattens."""
    fig, ax = plt.subplots(figsize=(7.4, 4.3))

    ax.plot(N_TREES, np.array(N_TREES_ACCURACY) * 100, marker="o", color=BLUE, linewidth=2.2, markersize=7)
    for n, acc in zip(N_TREES, N_TREES_ACCURACY):
        ax.annotate(
            f"{acc * 100:.1f}%",
            xy=(n, acc * 100),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK,
        )

    ax.annotate(
        "One tree alone: 88.9%\n(this is just a Decision Tree)",
        xy=(1, 88.89),
        xytext=(18, 90.4),
        fontsize=8.5,
        color=WARN,
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.1),
    )
    ax.annotate(
        "Plateau -- more trees never hurt\naccuracy, they just cost time",
        xy=(100, 100),
        xytext=(58, 94.6),
        fontsize=8.5,
        color=GOOD,
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.1),
    )

    ax.set_xlabel("Number of trees  (n_estimators)", fontsize=10, color=INK)
    ax.set_ylabel("Test accuracy  (%)", fontsize=10, color=INK)
    ax.set_ylim(86, 102)
    ax.grid(alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Does the Number of Trees Matter?", "Wine dataset -- averaging quickly stabilises the prediction")
    fig.tight_layout()
    return fig


@figure("forest_vs_tree_importance")
def forest_vs_tree_importance() -> plt.Figure:
    """A single tree concentrates importance; a forest spreads it out."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharex=True)

    for ax, data, title, colour in (
        (axes[0], TREE_IMPORTANCE, "Single Decision Tree (depth 4)", WARN),
        (axes[1], FOREST_IMPORTANCE, "Random Forest (200 trees)", BLUE),
    ):
        order = sorted(data, key=data.get)
        values = [data[f] for f in order]
        ax.barh(order, values, color=colour, edgecolor=colour, linewidth=1.1, alpha=0.85)
        for i, value in enumerate(values):
            ax.text(value + 0.008, i, f"{value:.3f}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, 0.48)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=NAVY)
        ax.grid(axis="x", alpha=0.25, color=LINE)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    axes[0].set_xlabel("Importance", fontsize=9.5, color=INK)
    axes[1].set_xlabel("Importance", fontsize=9.5, color=INK)
    fig.suptitle(
        "The tree bets everything on 2 features; the forest spreads its weight across many",
        fontsize=11.5,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig


@figure("variance_reduction")
def variance_reduction() -> plt.Figure:
    """Why averaging many noisy trees produces a steadier answer."""
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    x = np.linspace(0, 10, 300)
    truth = np.sin(x)

    # Left: individual trees wobble wildly around the truth.
    ax = axes[0]
    for _ in range(12):
        noisy = truth + rng.normal(0, 0.42, size=x.size)
        ax.plot(x, noisy, color=SKY, linewidth=0.8, alpha=0.5)
    ax.plot(x, truth, color=NAVY, linewidth=2.6, label="True pattern")
    ax.set_title("12 individual trees -- high variance", fontsize=10.5, fontweight="bold", color=WARN)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # Right: their average hugs the truth.
    ax = axes[1]
    stack = np.array([truth + rng.normal(0, 0.42, size=x.size) for _ in range(12)])
    ax.plot(x, truth, color=NAVY, linewidth=2.6, label="True pattern")
    ax.plot(x, stack.mean(axis=0), color=GOOD, linewidth=2.2, label="Average of the 12")
    ax.set_title("Their average -- low variance", fontsize=10.5, fontweight="bold", color=GOOD)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2, color=LINE)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle(
        "Errors that point in random directions cancel out when you average them",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig
