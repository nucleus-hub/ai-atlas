"""Figures for note 01 -- Decision Trees.

Every number here is either computed live from the slides' 14-row Play Tennis
dataset, or reproduced by re-running the practical notebook's exact pipeline
(``random_state=42``) against scikit-learn's Wine dataset. Nothing is
hand-typed from the slide images.
"""

from __future__ import annotations

import math

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
    WARN,
    blank_axes,
    connect,
    figure,
    node_box,
    panel_title,
)

# -- The slides' Play Tennis dataset, as (Outlook, Temperature, Humidity, Windy, Play) --
PLAY_TENNIS = [
    ("Sunny", "Hot", "High", False, "No"),
    ("Sunny", "Hot", "High", True, "No"),
    ("Overcast", "Hot", "High", False, "Yes"),
    ("Rainy", "Mild", "High", False, "Yes"),
    ("Rainy", "Cool", "Normal", False, "Yes"),
    ("Rainy", "Cool", "Normal", True, "No"),
    ("Overcast", "Cool", "Normal", True, "Yes"),
    ("Sunny", "Mild", "High", False, "No"),
    ("Sunny", "Cool", "Normal", False, "Yes"),
    ("Rainy", "Mild", "Normal", False, "Yes"),
    ("Sunny", "Mild", "Normal", True, "Yes"),
    ("Overcast", "Mild", "High", True, "Yes"),
    ("Overcast", "Hot", "Normal", False, "Yes"),
    ("Rainy", "Mild", "High", True, "No"),
]
COLUMNS = ["Outlook", "Temperature", "Humidity", "Windy"]


def entropy(counts: list[int]) -> float:
    """H(S) = -sum p_i log2(p_i) -- 0 for a pure node."""
    total = sum(counts)
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def gini(counts: list[int]) -> float:
    """Gini(S) = 1 - sum p_i^2 -- 0 for a pure node."""
    total = sum(counts)
    return 1 - sum((c / total) ** 2 for c in counts)


def _counts(rows: list[tuple]) -> list[int]:
    yes = sum(1 for r in rows if r[-1] == "Yes")
    return [yes, len(rows) - yes]


def branches(feature: str) -> dict[str, list[tuple]]:
    """Partition the dataset by the values of ``feature``."""
    index = COLUMNS.index(feature)
    groups: dict[str, list[tuple]] = {}
    for row in PLAY_TENNIS:
        groups.setdefault(row[index], []).append(row)
    return groups


def gain(feature: str, measure) -> float:
    """Information Gain (entropy) or Gini Gain, depending on ``measure``."""
    parent = measure(_counts(PLAY_TENNIS))
    total = len(PLAY_TENNIS)
    weighted = sum(len(rows) / total * measure(_counts(rows)) for rows in branches(feature).values())
    return parent - weighted


@figure("entropy_curve")
def entropy_curve() -> plt.Figure:
    """Entropy vs. probability -- peaks at 1 bit when the node is 50/50."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    p = np.linspace(0.0001, 0.9999, 500)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    ax.plot(p, h, color=BLUE, linewidth=2.4)
    ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1.2)
    ax.plot([0.5], [1.0], "o", color=WARN, markersize=8, zorder=5)
    ax.annotate(
        "Maximum uncertainty\np = 0.5  ->  H = 1 bit",
        xy=(0.5, 1.0),
        xytext=(0.62, 0.62),
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.2),
    )
    for x in (0.0, 1.0):
        ax.plot([x], [0.0], "o", color=GOOD, markersize=8, zorder=5)
    ax.annotate(
        "Pure node\nH = 0",
        xy=(0.0, 0.0),
        xytext=(0.07, 0.26),
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.2),
    )

    ax.set_xlabel("Probability of Class 1  (p)", fontsize=10, color=INK)
    ax.set_ylabel("Entropy  H(S)   (bits)", fontsize=10, color=INK)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, color=LINE)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Entropy vs. Probability", "Uncertainty is highest when the classes are perfectly mixed")
    fig.tight_layout()
    return fig


@figure("gini_vs_entropy")
def gini_vs_entropy() -> plt.Figure:
    """Both impurity measures side by side -- same shape, different scale."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    p = np.linspace(0.0001, 0.9999, 500)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    g = 1 - (p**2 + (1 - p) ** 2)

    ax.plot(p, h, color=BLUE, linewidth=2.4, label="Entropy (base 2) -- max 1.0")
    ax.plot(p, g, color=WARN, linewidth=2.4, label="Gini Impurity -- max 0.5")
    ax.fill_between(p, g, h, color=MIST, alpha=0.35)

    ax.set_xlabel("Probability of Class 1  (p)", fontsize=10, color=INK)
    ax.set_ylabel("Impurity", fontsize=10, color=INK)
    ax.set_ylim(0, 1.15)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25, color=LINE)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Gini Impurity vs. Entropy", "Both are 0 at pure nodes and peak at p = 0.5 -- Gini just peaks lower")
    fig.tight_layout()
    return fig


@figure("information_gain_comparison")
def information_gain_comparison() -> plt.Figure:
    """The slides' Step 6 -- IG and Gini Gain for all four features, computed live."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, measure, title, colour in (
        (axes[0], entropy, "Information Gain (Entropy)", BLUE),
        (axes[1], gini, "Gini Gain", WARN),
    ):
        scores = {f: gain(f, measure) for f in COLUMNS}
        order = sorted(scores, key=scores.get)
        values = [scores[f] for f in order]
        colours = [colour if f == "Outlook" else MIST for f in order]

        bars = ax.barh(order, values, color=colours, edgecolor=colour, linewidth=1.2)
        for bar, value in zip(bars, values):
            ax.text(
                value + max(values) * 0.03,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=9,
                color=INK,
                fontweight="bold" if value == max(values) else "normal",
            )
        ax.set_xlim(0, max(values) * 1.28)
        ax.set_xlabel(title, fontsize=10, color=INK)
        ax.grid(axis="x", alpha=0.25, color=LINE)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    axes[0].set_title("Entropy route", fontsize=11, fontweight="bold", color=NAVY)
    axes[1].set_title("Gini route", fontsize=11, fontweight="bold", color=NAVY)
    fig.suptitle(
        "Outlook wins under both measures -- so it becomes the root node",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig


@figure("play_tennis_tree")
def play_tennis_tree() -> plt.Figure:
    """The final tree from the slides' worked example."""
    fig, ax = plt.subplots(figsize=(9, 5.2))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 6))

    root = (5.0, 5.2)
    sunny, overcast, rainy = (2.0, 3.3), (5.0, 3.3), (8.0, 3.3)
    leaves = {
        "high": (0.9, 1.3),
        "normal": (3.1, 1.3),
        "strong": (6.9, 1.3),
        "weak": (9.1, 1.3),
    }

    node_box(ax, root, "Outlook?", facecolor=MIST, edgecolor=NAVY, width=1.9, height=0.62, bold=True)
    node_box(ax, sunny, "Humidity?", facecolor=PAPER, edgecolor=BLUE, width=1.9, height=0.6)
    node_box(ax, overcast, "Yes", facecolor="#E4F3E9", edgecolor=GOOD, width=1.4, height=0.6, bold=True)
    node_box(ax, rainy, "Windy?", facecolor=PAPER, edgecolor=BLUE, width=1.9, height=0.6)

    node_box(ax, leaves["high"], "No", facecolor="#FBE9E7", edgecolor=WARN, width=1.3, height=0.55, bold=True)
    node_box(ax, leaves["normal"], "Yes", facecolor="#E4F3E9", edgecolor=GOOD, width=1.3, height=0.55, bold=True)
    node_box(ax, leaves["strong"], "No", facecolor="#FBE9E7", edgecolor=WARN, width=1.3, height=0.55, bold=True)
    node_box(ax, leaves["weak"], "Yes", facecolor="#E4F3E9", edgecolor=GOOD, width=1.3, height=0.55, bold=True)

    connect(ax, root, sunny, "Sunny")
    connect(ax, root, overcast, "Overcast")
    connect(ax, root, rainy, "Rainy")
    connect(ax, sunny, leaves["high"], "High")
    connect(ax, sunny, leaves["normal"], "Normal")
    connect(ax, rainy, leaves["strong"], "Strong")
    connect(ax, rainy, leaves["weak"], "Weak")

    ax.text(
        5.0,
        0.35,
        'New day: Outlook = Sunny, Humidity = High   ->   follow Sunny, then High   ->   predict "No"',
        ha="center",
        fontsize=9.5,
        color=INK,
        style="italic",
    )
    ax.text(
        5.0,
        2.68,
        "pure node -- 4 Yes, 0 No\nno further question needed",
        fontsize=8,
        color=GREY,
        ha="center",
        va="top",
    )
    panel_title(ax, "The Resulting Decision Tree", "Play Tennis -- grown by repeatedly maximising Information Gain")
    fig.tight_layout()
    return fig


@figure("outlook_split")
def outlook_split() -> plt.Figure:
    """Step 2 of the worked example -- the winning split, branch by branch."""
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 5.4))

    parent_counts = _counts(PLAY_TENNIS)
    parent_h = entropy(parent_counts)

    node_box(
        ax,
        (5.0, 4.5),
        f"All 14 days   ({parent_counts[0]} Yes, {parent_counts[1]} No)\nH(S) = {parent_h:.3f} bits",
        facecolor=MIST,
        edgecolor=NAVY,
        width=4.6,
        height=0.82,
        fontsize=10,
        bold=True,
    )

    groups = branches("Outlook")
    positions = {"Sunny": 1.8, "Overcast": 5.0, "Rainy": 8.2}
    for name, x in positions.items():
        rows = groups[name]
        counts = _counts(rows)
        h = abs(entropy(counts))  # abs() tidies the -0.0 that a pure node produces
        pure = h == 0
        node_box(
            ax,
            (x, 2.5),
            f"{name}  ({len(rows)} days)\n{counts[0]} Yes, {counts[1]} No\nH = {h:.2f}",
            facecolor="#E4F3E9" if pure else PAPER,
            edgecolor=GOOD if pure else BLUE,
            width=2.6,
            height=1.05,
            fontsize=9.5,
        )
        connect(ax, (5.0, 4.09), (x, 3.03), f"{len(rows)}/14")

    weighted = sum(len(r) / 14 * entropy(_counts(r)) for r in groups.values())
    ax.text(
        5.0,
        1.25,
        f"Weighted entropy after the split = {weighted:.3f}",
        ha="center",
        fontsize=10,
        color=INK,
    )
    ax.text(
        5.0,
        0.62,
        f"Information Gain (Outlook) = {parent_h:.3f} - {weighted:.3f} = {parent_h - weighted:.3f}",
        ha="center",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
    )
    panel_title(ax, 'Splitting on "Outlook"', "Entropy is measured inside each branch, then averaged by branch size")
    fig.tight_layout()
    return fig


@figure("tree_anatomy")
def tree_anatomy() -> plt.Figure:
    """Root / decision / leaf vocabulary on a generic tree."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    blank_axes(ax, xlim=(0, 10), ylim=(0, 5.2))

    root = (5.0, 4.3)
    left, right = (2.6, 2.6), (7.4, 2.6)
    leaves = [(1.1, 0.9), (4.1, 0.9), (6.1, 0.9), (8.9, 0.9)]

    node_box(ax, root, "Salary > $80k?", facecolor=MIST, edgecolor=NAVY, width=2.4, height=0.62, bold=True)
    node_box(ax, left, "Commute < 1h?", facecolor=PAPER, edgecolor=BLUE, width=2.4, height=0.6)
    node_box(ax, right, "Free coffee?", facecolor=PAPER, edgecolor=BLUE, width=2.4, height=0.6)
    for pos, label in zip(leaves, ["Decline", "Accept", "Decline", "Accept"]):
        good = label == "Accept"
        node_box(
            ax,
            pos,
            label,
            facecolor="#E4F3E9" if good else "#FBE9E7",
            edgecolor=GOOD if good else WARN,
            width=1.6,
            height=0.55,
            bold=True,
        )

    connect(ax, root, left, "No")
    connect(ax, root, right, "Yes")
    connect(ax, left, leaves[0], "No")
    connect(ax, left, leaves[1], "Yes")
    connect(ax, right, leaves[2], "No")
    connect(ax, right, leaves[3], "Yes")

    ax.annotate(
        "Root node\nthe whole dataset",
        xy=(6.25, 4.3),
        xytext=(8.1, 4.55),
        fontsize=8.5,
        color=NAVY,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.1),
    )
    ax.annotate(
        "Decision node\nasks a question",
        xy=(1.4, 2.6),
        xytext=(1.15, 3.5),
        fontsize=8.5,
        color=NAVY,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.1),
    )
    ax.annotate(
        "Leaf node\nfinal prediction",
        xy=(4.1, 1.22),
        xytext=(4.15, 2.15),
        fontsize=8.5,
        color=NAVY,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.1),
    )
    panel_title(ax, "Anatomy of a Decision Tree", "A flow-chart of simple questions ending in a prediction")
    fig.tight_layout()
    return fig


# -- Wine dataset numbers, reproduced from the notebook pipeline (random_state=42) --
WINE_DEPTHS = list(range(1, 16))
WINE_TRAIN_ACC = [0.6617, 0.9398, 0.9925, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
WINE_TEST_ACC = [0.600, 0.8667, 0.9778, 0.9778, 0.9778, 0.9778, 0.9778, 0.9778, 0.9778,
                 0.9778, 0.9778, 0.9778, 0.9778, 0.9778, 0.9778]

# Feature importances of the fitted max_depth=4 tree (only the 6 features it actually used).
WINE_IMPORTANCE = {
    "flavanoids": 0.4108,
    "color_intensity": 0.4033,
    "proline": 0.1003,
    "ash": 0.0436,
    "od280/od315_of_diluted_wines": 0.0224,
    "alcohol": 0.0196,
}


@figure("tree_depth_tradeoff")
def tree_depth_tradeoff() -> plt.Figure:
    """Training vs test accuracy as the Wine tree is allowed to grow deeper."""
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    ax.plot(WINE_DEPTHS, WINE_TRAIN_ACC, marker="o", color=BLUE, linewidth=2, label="Training accuracy")
    ax.plot(WINE_DEPTHS, WINE_TEST_ACC, marker="o", color=WARN, linewidth=2, label="Test accuracy")

    ax.axvspan(0.5, 2.5, color="#FBE9E7", alpha=0.6, zorder=0)
    ax.text(1.5, 0.53, "Underfitting\n(too simple)", ha="center", fontsize=8.5, color=WARN)
    ax.axvspan(2.5, 4.5, color="#E4F3E9", alpha=0.6, zorder=0)
    ax.text(3.5, 0.53, "Good fit", ha="center", fontsize=8.5, color=GOOD)

    ax.annotate(
        "Training accuracy hits 1.00 and stays there --\nthe tree has memorised every training wine",
        xy=(9, 1.0),
        xytext=(6.4, 0.74),
        fontsize=8.5,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.1),
    )

    ax.set_xlabel("Max tree depth", fontsize=10, color=INK)
    ax.set_ylabel("Accuracy", fontsize=10, color=INK)
    ax.set_ylim(0.45, 1.06)
    ax.set_xticks(WINE_DEPTHS)
    ax.grid(alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Underfitting vs. Overfitting as Depth Increases", "Wine dataset -- the gap between the two lines is variance")
    fig.tight_layout()
    return fig


@figure("tree_feature_importance")
def tree_feature_importance() -> plt.Figure:
    """Which features the fitted Wine tree actually relied on."""
    fig, ax = plt.subplots(figsize=(7.6, 4.0))

    order = sorted(WINE_IMPORTANCE, key=WINE_IMPORTANCE.get)
    values = [WINE_IMPORTANCE[f] for f in order]
    colours = [BLUE if v > 0.1 else MIST for v in values]

    bars = ax.barh(order, values, color=colours, edgecolor=BLUE, linewidth=1.1)
    for bar, value in zip(bars, values):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9, color=INK)

    ax.set_xlim(0, 0.5)
    ax.set_xlabel("Importance  (total impurity decrease, normalised)", fontsize=10, color=INK)
    ax.grid(axis="x", alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(
        ax,
        "Decision Tree: Feature Importance (Wine)",
        "Only 6 of the 15 available features were used at all -- the rest scored exactly 0",
    )
    fig.tight_layout()
    return fig
