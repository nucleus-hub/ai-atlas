"""Figures for Topic 02 -- Ensemble Methods & Random Forest."""

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
    SKY,
    WARN,
    blank_axes,
    connect,
    figure,
    node_box,
    panel_title,
)


@figure("single_tree_variance")
def single_tree_variance() -> plt.Figure:
    """Two trees from nearly identical data, with completely different structures."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))

    trees = [
        ("Trained on sample A", "Outlook", ["Humidity", "Windy"]),
        ("Trained on sample B\n(two rows changed)", "Humidity", ["Outlook", "Temperature"]),
    ]
    for ax, (title, root, kids) in zip(axes, trees):
        blank_axes(ax, (0, 6), (0, 5))
        ax.set_title(title, fontsize=10.5, color=NAVY, fontweight="bold")
        r = (3.0, 4.1)
        left, right = (1.4, 2.3), (4.6, 2.3)
        connect(ax, r, left)
        connect(ax, r, right)
        node_box(ax, r, root, facecolor=NAVY, textcolor=PAPER, edgecolor=NAVY, width=2.0, bold=True)
        node_box(ax, left, kids[0], facecolor=MIST, width=1.9)
        node_box(ax, right, kids[1], facecolor=MIST, width=1.9)
        for x, parent in ((0.8, left), (2.0, left), (4.0, right), (5.2, right)):
            leaf = (x, 0.7)
            connect(ax, parent, leaf)
            label = "Yes" if x in (2.0, 4.0) else "No"
            colour = GOOD if label == "Yes" else WARN
            node_box(ax, leaf, label, edgecolor=colour, textcolor=colour, width=1.0, height=0.5, fontsize=8)

    fig.suptitle(
        "High Variance: a small data change rebuilds the whole tree",
        fontsize=12,
        color=NAVY,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


@figure("ensemble_families")
def ensemble_families() -> plt.Figure:
    """Bagging trains in parallel, boosting in sequence, stacking blends model types."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))

    # Bagging -- parallel, identical model type.
    ax = axes[0]
    blank_axes(ax, (0, 6), (0, 5))
    ax.set_title("Bagging  (Random Forest)", fontsize=10.5, color=NAVY, fontweight="bold")
    node_box(ax, (3.0, 4.3), "Dataset", facecolor=NAVY, textcolor=PAPER, edgecolor=NAVY, width=2.0, bold=True)
    for x in (1.1, 3.0, 4.9):
        connect(ax, (3.0, 4.3), (x, 2.6))
        node_box(ax, (x, 2.6), "Tree", facecolor=MIST, width=1.4, height=0.55, fontsize=8.5)
        connect(ax, (x, 2.6), (3.0, 1.0))
    node_box(ax, (3.0, 1.0), "Vote / Average", facecolor="#CDEBD6", edgecolor=GOOD, width=2.6, fontsize=8.5)
    ax.text(3.0, 0.25, "trained in parallel -> cuts VARIANCE", ha="center", fontsize=8, color=GREY, style="italic")

    # Boosting -- sequential.
    ax = axes[1]
    blank_axes(ax, (0, 6), (0, 5))
    ax.set_title("Boosting  (XGBoost, AdaBoost)", fontsize=10.5, color=NAVY, fontweight="bold")
    for i, y in enumerate((4.2, 2.9, 1.6)):
        node_box(ax, (3.0, y), f"Tree {i + 1}", facecolor=MIST, width=1.7, height=0.55, fontsize=8.5)
        if i:
            ax.annotate(
                "",
                xy=(3.0, y + 0.28),
                xytext=(3.0, y + 1.02),
                arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.6),
            )
            ax.text(3.35, y + 0.65, "fix the errors", fontsize=7.5, color=WARN, va="center")
    node_box(ax, (3.0, 0.55), "Final model", facecolor="#CDEBD6", edgecolor=GOOD, width=2.3, height=0.5, fontsize=8.5)
    ax.text(3.0, 0.0, "trained in sequence -> cuts BIAS", ha="center", fontsize=8, color=GREY, style="italic")

    # Stacking -- different learner types.
    ax = axes[2]
    blank_axes(ax, (0, 6), (0, 5))
    ax.set_title("Stacking", fontsize=10.5, color=NAVY, fontweight="bold")
    for x, name in ((1.1, "Tree"), (3.0, "SVM"), (4.9, "KNN")):
        node_box(ax, (x, 3.6), name, facecolor=MIST, width=1.4, height=0.55, fontsize=8.5)
        connect(ax, (x, 3.6), (3.0, 1.9))
    node_box(ax, (3.0, 1.9), "Meta-learner", facecolor=SKY, edgecolor=BLUE, width=2.4, fontsize=8.5)
    connect(ax, (3.0, 1.9), (3.0, 0.8))
    node_box(ax, (3.0, 0.8), "Prediction", facecolor="#CDEBD6", edgecolor=GOOD, width=2.1, height=0.5, fontsize=8.5)
    ax.text(3.0, 0.15, "blends DIFFERENT model types", ha="center", fontsize=8, color=GREY, style="italic")

    fig.tight_layout()
    return fig


@figure("random_forest_pipeline")
def random_forest_pipeline() -> plt.Figure:
    """The five-step Random Forest flow."""
    fig, ax = plt.subplots(figsize=(11.2, 2.9))
    blank_axes(ax, (0, 11.4), (0, 3))
    panel_title(ax, "Random Forest in Five Steps", "")

    steps = [
        ("1\nInput Dataset", "N samples, M features"),
        ("2\nBootstrap Sampling", "random sampling\nwith replacement"),
        ("3\nBuild Trees", "one tree per\nbootstrap sample"),
        ("4\nAggregate", "voting or\naveraging"),
        ("5\nFinal Prediction", "majority vote\n/ mean"),
    ]
    xs = np.linspace(1.2, 10.2, 5)
    for i, (x, (title, sub)) in enumerate(zip(xs, steps)):
        colour = NAVY if i in (0, 4) else MIST
        text_colour = PAPER if i in (0, 4) else INK
        node_box(
            ax, (x, 2.0), title, facecolor=colour, textcolor=text_colour, edgecolor=NAVY if i in (0, 4) else BLUE,
            width=1.85, height=0.95, fontsize=8.5, bold=True,
        )
        ax.text(x, 1.0, sub, ha="center", va="center", fontsize=7.5, color=GREY)
        if i < 4:
            ax.annotate(
                "",
                xy=(xs[i + 1] - 0.95, 2.0),
                xytext=(x + 0.95, 2.0),
                arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.6),
            )
    fig.tight_layout()
    return fig


@figure("bootstrap_sampling")
def bootstrap_sampling() -> plt.Figure:
    """Sampling with replacement: duplicates appear, some rows vanish."""
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    blank_axes(ax, (0, 9), (0, 5))
    panel_title(ax, "Bagging: Each Tree Sees a Different Bootstrap Sample", "Sampling with replacement")

    original = ["A", "B", "C", "D", "E", "F"]
    samples = [
        (["A", "B", "B", "D", "E", "E"], {"B", "E"}, {"C", "F"}),
        (["A", "A", "C", "C", "D", "F"], {"A", "C"}, {"B", "E"}),
        (["B", "C", "D", "D", "F", "F"], {"D", "F"}, {"A", "E"}),
    ]

    for i, letter in enumerate(original):
        node_box(ax, (2.0 + i * 0.85, 4.2), letter, facecolor=NAVY, textcolor=PAPER, edgecolor=NAVY,
                 width=0.72, height=0.52, fontsize=9, bold=True)
    ax.text(1.3, 4.2, "Data", ha="right", va="center", fontsize=9, color=INK, fontweight="bold")

    for row, (sample, dupes, missing) in enumerate(samples):
        y = 2.9 - row * 0.95
        ax.text(1.3, y, f"Tree {row + 1}", ha="right", va="center", fontsize=9, color=INK)
        for i, letter in enumerate(sample):
            colour = "#FDE9D9" if letter in dupes else MIST
            node_box(ax, (2.0 + i * 0.85, y), letter, facecolor=colour, width=0.72, height=0.52, fontsize=9)
        ax.text(7.5, y, f"missing: {', '.join(sorted(missing))}", va="center", fontsize=8, color=WARN)

    ax.text(4.5, 0.35, "orange = drawn more than once     |     ~37% of rows miss any given tree",
            ha="center", fontsize=8.5, color=GREY, style="italic")
    fig.tight_layout()
    return fig


@figure("random_feature_selection")
def random_feature_selection() -> plt.Figure:
    """Each split only gets to consider a random subset of the features."""
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    blank_axes(ax, (0, 9), (0, 4.6))
    panel_title(ax, "Random Feature Selection", "Each split chooses from a random subset, not all features")

    features = ["Outlook", "Temperature", "Humidity", "Windy"]
    offered = [
        ("Split A", {"Temperature", "Humidity"}),
        ("Split B", {"Outlook", "Windy"}),
        ("Split C", {"Humidity", "Windy"}),
    ]

    for i, name in enumerate(features):
        ax.text(2.4 + i * 1.75, 3.85, name, ha="center", fontsize=8.5, color=INK, fontweight="bold")

    for row, (label, subset) in enumerate(offered):
        y = 2.9 - row * 0.85
        ax.text(1.4, y, label, ha="right", va="center", fontsize=9, color=INK)
        for i, name in enumerate(features):
            live = name in subset
            node_box(
                ax,
                (2.4 + i * 1.75, y),
                "considered" if live else "-",
                facecolor=MIST if live else PAPER,
                edgecolor=BLUE if live else LINE,
                textcolor=INK if live else GREY,
                width=1.5,
                height=0.5,
                fontsize=7.5,
            )
    ax.text(4.8, 0.3, "Bagging varies the ROWS. Random feature selection varies the COLUMNS.",
            ha="center", fontsize=8.5, color=GREY, style="italic")
    fig.tight_layout()
    return fig


@figure("forest_majority_vote")
def forest_majority_vote() -> plt.Figure:
    """Five trees vote 4-1; the single wrong tree is outvoted."""
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    blank_axes(ax, (0, 9), (0, 4.4))
    panel_title(ax, "Aggregation: One Wrong Tree Gets Outvoted", "Majority vote across five trees")

    votes = ["Yes", "Yes", "No", "Yes", "Yes"]
    for i, vote in enumerate(votes):
        x = 1.3 + i * 1.6
        right = vote == "Yes"
        node_box(ax, (x, 3.3), f"Tree {i + 1}", facecolor=MIST, width=1.3, height=0.5, fontsize=8.5)
        node_box(
            ax, (x, 2.4), vote,
            facecolor="#CDEBD6" if right else "#FBE3E0",
            edgecolor=GOOD if right else WARN,
            textcolor=GOOD if right else WARN,
            width=1.3, height=0.5, fontsize=9, bold=True,
        )
        connect(ax, (x, 2.4), (4.5, 1.2))

    node_box(ax, (4.5, 1.2), "Majority vote: Yes  (4 of 5)   ->   probability 0.80",
             facecolor=NAVY, textcolor=PAPER, edgecolor=NAVY, width=6.2, height=0.62, fontsize=9, bold=True)
    ax.text(4.5, 0.35, "Tree 3 was wrong. It did not matter -- that is the entire point.",
            ha="center", fontsize=8.5, color=GREY, style="italic")
    fig.tight_layout()
    return fig


@figure("oob_split")
def oob_split() -> plt.Figure:
    """The 63/37 bootstrap partition that funds free validation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0), gridspec_kw={"width_ratios": [1, 1.35]})

    ax1.pie(
        [63, 37],
        labels=["In bootstrap sample\n~63%", "Out-of-bag\n~37%"],
        colors=[NAVY, "#FDE9D9"],
        autopct="%1.0f%%",
        startangle=90,
        textprops=dict(fontsize=9, color=INK),
        wedgeprops=dict(edgecolor=PAPER, linewidth=2),
    )
    ax1.set_title("Every tree ignores about a third of the data", fontsize=10.5, color=NAVY, fontweight="bold")

    n = np.arange(1, 201)
    prob = (1 - 1 / n) ** n
    ax2.plot(n, prob, color=BLUE, linewidth=2.4)
    ax2.axhline(1 / np.e, color=WARN, linestyle="--", linewidth=1.4)
    ax2.text(120, 1 / np.e + 0.012, "1/e = 0.368", fontsize=9, color=WARN)
    ax2.set_xlabel("Dataset size  n", fontsize=10, color=INK)
    ax2.set_ylabel("P(a given row is left out)", fontsize=10, color=INK)
    ax2.set_ylim(0, 0.55)
    ax2.set_title("(1 - 1/n)^n converges to 1/e", fontsize=10.5, color=NAVY, fontweight="bold")
    ax2.grid(color=LINE, linewidth=0.8)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)

    fig.suptitle("Out-of-Bag Error: a free, unbiased validation set", fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("trees_vs_accuracy")
def trees_vs_accuracy() -> plt.Figure:
    """More trees plateau rather than overfit."""
    trees = np.arange(1, 301)
    rng = np.random.default_rng(3)
    accuracy = 0.955 - 0.13 * np.exp(-trees / 35) + rng.normal(0, 0.0016, trees.size)

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.plot(trees, accuracy, color=BLUE, linewidth=1.9)
    ax.axhline(0.955, color=GREY, linestyle=":", linewidth=1.2)
    ax.axvspan(1, 30, color="#FBE3E0", alpha=0.55)
    ax.axvspan(100, 300, color="#E8F3EC", alpha=0.7)
    ax.text(15, 0.872, "too few trees\nnoisy, unstable", ha="center", fontsize=8.5, color=WARN)
    ax.text(200, 0.905, "plateau: more trees cost time,\nnot accuracy -- and never overfit",
            ha="center", fontsize=8.5, color=GOOD)

    ax.set_xlabel("n_estimators (number of trees)", fontsize=10, color=INK)
    ax.set_ylabel("Validation accuracy", fontsize=10, color=INK)
    ax.set_ylim(0.85, 0.98)
    ax.set_title("Adding Trees Plateaus -- It Never Overfits", fontsize=12, color=NAVY, fontweight="bold")
    ax.grid(color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


@figure("tree_vs_forest_comparison")
def tree_vs_forest_comparison() -> plt.Figure:
    """Side-by-side profile of a single tree against an ensemble."""
    aspects = ["Variance", "Overfitting\nrisk", "Stability", "Accuracy", "Interpret-\nability", "Training\nspeed"]
    tree = [0.90, 0.85, 0.30, 0.55, 0.95, 0.95]
    forest = [0.25, 0.20, 0.88, 0.90, 0.35, 0.40]

    y = np.arange(len(aspects))
    height = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.barh(y + height / 2, tree, height, label="Single Decision Tree", color=MIST, edgecolor=BLUE)
    ax.barh(y - height / 2, forest, height, label="Random Forest", color=NAVY, edgecolor=NAVY)

    ax.set_yticks(y)
    ax.set_yticklabels(aspects, fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.set_title("Where Each Model Wins  (longer bar = more of that trait)",
                 fontsize=11.5, color=NAVY, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(axis="x", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig
