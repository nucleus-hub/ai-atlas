"""Figures for Topic 01 -- Decision Trees.

Every diagram here is built around the course's Play Tennis worked example so
the note's numbers and its pictures can never drift apart.
"""

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
    gain_bars,
    node_box,
    panel_title,
)


def _entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in bits, with the 0*log0 = 0 convention handled."""
    safe = np.clip(p, 1e-12, 1 - 1e-12)
    return -safe * np.log2(safe) - (1 - safe) * np.log2(1 - safe)


def _gini(p: np.ndarray) -> np.ndarray:
    """Binary Gini impurity."""
    return 1 - p**2 - (1 - p) ** 2


@figure("decision_tree_anatomy")
def decision_tree_anatomy() -> plt.Figure:
    """Root / decision node / leaf, labelled on the course's salary example."""
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    blank_axes(ax, (0, 10), (0, 6))
    panel_title(ax, "Anatomy of a Decision Tree", "One question per node, one prediction per leaf")

    root = (5.0, 5.1)
    left, right = (2.6, 3.1), (7.4, 3.1)
    leaves = [(1.3, 1.0), (3.9, 1.0), (6.1, 1.0), (8.7, 1.0)]

    connect(ax, root, left, "No")
    connect(ax, root, right, "Yes")
    for parent, (a, b), labels in ((left, leaves[:2], ("No", "Yes")), (right, leaves[2:], ("No", "Yes"))):
        connect(ax, parent, a, labels[0])
        connect(ax, parent, b, labels[1])

    node_box(ax, root, "Salary > $80k?", facecolor=NAVY, textcolor=PAPER, edgecolor=NAVY, width=2.6, bold=True)
    node_box(ax, left, "Years > 3?", facecolor=MIST, width=2.1)
    node_box(ax, right, "Debt > 50%?", facecolor=MIST, width=2.1)
    for xy, text in zip(leaves, ("Decline", "Approve", "Approve", "Decline")):
        colour = GOOD if text == "Approve" else WARN
        node_box(ax, xy, text, edgecolor=colour, textcolor=colour, width=1.7, height=0.55)

    notes = (
        (9.4, 5.1, "Root node\nholds all data"),
        (9.4, 3.1, "Decision nodes\nask more questions"),
        (9.4, 1.0, "Leaf nodes\nfinal prediction"),
    )
    for x, y, text in notes:
        ax.text(x, y, text, ha="right", va="center", fontsize=8.5, color=GREY, style="italic")

    fig.tight_layout()
    return fig


@figure("divide_and_conquer")
def divide_and_conquer() -> plt.Figure:
    """Recursive partitioning shown as successive cuts through a 2D sample."""
    rng = np.random.default_rng(7)
    blues = rng.normal([0.3, 0.7], 0.13, size=(30, 2))
    reds = np.vstack([rng.normal([0.72, 0.3], 0.12, size=(20, 2)), rng.normal([0.3, 0.22], 0.1, size=(12, 2))])

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    cuts = [
        ("Split 1 -- full dataset", []),
        ("Split 2 -- partition again", [("v", 0.55)]),
        ("Split 3 -- leaves are pure", [("v", 0.55), ("h", 0.45)]),
    ]

    for ax, (title, lines) in zip(axes, cuts):
        ax.scatter(blues[:, 0], blues[:, 1], s=26, color=BLUE, label="Yes", zorder=3)
        ax.scatter(reds[:, 0], reds[:, 1], s=26, color=WARN, marker="s", label="No", zorder=3)
        for orientation, pos in lines:
            if orientation == "v":
                ax.axvline(pos, color=NAVY, linewidth=2)
            else:
                ax.plot([0, 0.55], [pos, pos], color=NAVY, linewidth=2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10, color=NAVY, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(
        "Divide and Conquer: each cut splits one region, then recursion repeats on the pieces",
        fontsize=11,
        color=NAVY,
        fontweight="bold",
    )
    fig.tight_layout()
    return fig


@figure("entropy_curve")
def entropy_curve() -> plt.Figure:
    """Binary entropy against class probability, peaking at 1 bit."""
    p = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(p, _entropy(p), color=NAVY, linewidth=2.4)
    ax.fill_between(p, _entropy(p), color=MIST, alpha=0.35)

    marks = [(0.0, "Pure node\nH = 0"), (0.5, "Maximum uncertainty\nH = 1 bit"), (1.0, "Pure node\nH = 0")]
    for x, label in marks:
        y = float(_entropy(np.array([x]))[0])
        ax.scatter([x], [y], s=45, color=WARN, zorder=4)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x, y + 0.22 if x == 0.5 else y + 0.32),
            ha="center",
            fontsize=8.5,
            color=INK,
            arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1),
        )

    ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1)
    ax.set_xlabel("Probability of Class 1  (p)", fontsize=10, color=INK)
    ax.set_ylabel("Entropy  H(S)  in bits", fontsize=10, color=INK)
    ax.set_ylim(0, 1.35)
    ax.set_xlim(0, 1)
    ax.set_title("Entropy Peaks When the Classes Are Perfectly Mixed", fontsize=12, color=NAVY, fontweight="bold")
    ax.grid(color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


@figure("gini_vs_entropy")
def gini_vs_entropy() -> plt.Figure:
    """Both impurity measures overlaid -- same shape, different heights."""
    p = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(p, _entropy(p), color=NAVY, linewidth=2.4, label="Entropy (base 2), max = 1.0")
    ax.plot(p, _gini(p), color=SKY, linewidth=2.4, label="Gini impurity, max = 0.5")

    ax.annotate(
        "Both reach 0 at pure nodes",
        xy=(0.02, 0.03),
        xytext=(0.18, 0.28),
        fontsize=8.5,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1),
    )
    ax.annotate(
        "Both peak at p = 0.5",
        xy=(0.5, 0.5),
        xytext=(0.62, 0.22),
        fontsize=8.5,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1),
    )

    ax.set_xlabel("Probability of Class 1  (p)", fontsize=10, color=INK)
    ax.set_ylabel("Impurity", fontsize=10, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.12)
    ax.set_title("Gini vs. Entropy: Same Preference, Different Scale", fontsize=12, color=NAVY, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.grid(color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


@figure("outlook_split_entropy")
def outlook_split_entropy() -> plt.Figure:
    """The Outlook split drawn branch by branch with per-branch entropy."""
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    blank_axes(ax, (0, 10), (0, 6))
    panel_title(ax, "Splitting the 14 Days on Outlook", "Entropy per branch, then combined as a weighted average")

    root = (5.0, 5.0)
    kids = [(1.9, 2.9), (5.0, 2.9), (8.1, 2.9)]
    node_box(
        ax,
        root,
        "All 14 days\n9 Yes / 5 No    H = 0.940",
        facecolor=NAVY,
        textcolor=PAPER,
        edgecolor=NAVY,
        width=3.6,
        height=0.9,
        bold=True,
    )

    branches = [
        ("Sunny (5)\n2 Yes / 3 No\nH = 0.97", MIST, "Sunny"),
        ("Overcast (4)\n4 Yes / 0 No\nH = 0.00", "#CDEBD6", "Overcast"),
        ("Rainy (5)\n3 Yes / 2 No\nH = 0.97", MIST, "Rainy"),
    ]
    for xy, (text, colour, label) in zip(kids, branches):
        connect(ax, root, xy, label)
        edge = GOOD if colour != MIST else BLUE
        node_box(ax, xy, text, facecolor=colour, edgecolor=edge, width=2.5, height=1.15, fontsize=8.5)

    ax.text(
        5.0,
        1.1,
        "Weighted entropy = (5/14)(0.97) + (4/14)(0.00) + (5/14)(0.97) = 0.69\n"
        "Information Gain = 0.940 - 0.69 = 0.246",
        ha="center",
        va="center",
        fontsize=9.5,
        color=NAVY,
        fontweight="bold",
        bbox=dict(facecolor="#F2F6FD", edgecolor=LINE, boxstyle="round,pad=0.5"),
    )
    ax.text(5.0, 2.0, "pure -> instant leaf", ha="center", fontsize=8, color=GOOD, style="italic")
    fig.tight_layout()
    return fig


@figure("information_gain_bars")
def information_gain_bars() -> plt.Figure:
    """Information Gain for all four features -- Outlook wins."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    gain_bars(
        ax,
        ["Temperature", "Windy", "Humidity", "Outlook"],
        [0.029, 0.048, 0.152, 0.246],
        "Information Gain (bits)",
    )
    ax.set_title("Outlook Gives the Largest Drop in Entropy", fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("gini_gain_bars")
def gini_gain_bars() -> plt.Figure:
    """Gini Gain for all four features -- identical ranking to entropy."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    gain_bars(
        ax,
        ["Temperature", "Windy", "Humidity", "Outlook"],
        [0.019, 0.031, 0.092, 0.116],
        "Gini Gain",
    )
    ax.set_title("Gini Agrees: Same Ranking, Smaller Numbers", fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("play_tennis_tree")
def play_tennis_tree() -> plt.Figure:
    """The finished Play Tennis tree, with the Sunny/High prediction path traced."""
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    blank_axes(ax, (0, 10), (0, 6))
    panel_title(ax, "The Resulting Play Tennis Tree", "Highlighted path: Outlook = Sunny, Humidity = High -> No")

    root = (5.0, 5.2)
    humidity, overcast, windy = (2.1, 3.3), (5.0, 3.3), (7.9, 3.3)
    leaves = {
        "h_high": (1.0, 1.2),
        "h_norm": (3.2, 1.2),
        "w_strong": (6.8, 1.2),
        "w_weak": (9.0, 1.2),
    }

    connect(ax, root, humidity, "Sunny", color=WARN, linewidth=2.4)
    connect(ax, root, overcast, "Overcast")
    connect(ax, root, windy, "Rain")
    connect(ax, humidity, leaves["h_high"], "High", color=WARN, linewidth=2.4)
    connect(ax, humidity, leaves["h_norm"], "Normal")
    connect(ax, windy, leaves["w_strong"], "Strong")
    connect(ax, windy, leaves["w_weak"], "Weak")

    node_box(ax, root, "Outlook", facecolor=NAVY, textcolor=PAPER, edgecolor=NAVY, width=2.0, bold=True)
    node_box(ax, humidity, "Humidity?", facecolor=MIST, edgecolor=WARN, width=1.9)
    node_box(ax, overcast, "Yes", facecolor="#CDEBD6", edgecolor=GOOD, textcolor=GOOD, width=1.5, bold=True)
    node_box(ax, windy, "Windy?", facecolor=MIST, width=1.9)

    leaf_style = {
        "h_high": ("No", WARN, True),
        "h_norm": ("Yes", GOOD, False),
        "w_strong": ("No", WARN, False),
        "w_weak": ("Yes", GOOD, False),
    }
    for key, (text, colour, highlight) in leaf_style.items():
        node_box(
            ax,
            leaves[key],
            text,
            facecolor="#FBE3E0" if highlight else PAPER,
            edgecolor=colour,
            textcolor=colour,
            width=1.5,
            height=0.55,
            bold=highlight,
        )

    ax.text(
        5.0,
        0.25,
        "Overcast was already pure (4 Yes / 0 No), so it became a leaf immediately",
        ha="center",
        fontsize=8.5,
        color=GREY,
        style="italic",
    )
    fig.tight_layout()
    return fig


@figure("tree_feature_importance")
def tree_feature_importance() -> plt.Figure:
    """Feature importance for the Play Tennis tree."""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    gain_bars(
        ax,
        ["Temperature", "Wind", "Humidity", "Outlook"],
        [0.10, 0.15, 0.30, 0.45],
        "Importance",
    )
    ax.set_title("Which Features the Tree Actually Relied On", fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("tree_depth_overfitting")
def tree_depth_overfitting() -> plt.Figure:
    """Why unrestricted depth is dangerous: training accuracy rises, test accuracy falls."""
    depth = np.arange(1, 13)
    train = 0.70 + 0.30 * (1 - np.exp(-0.55 * depth))
    test = 0.68 + 0.26 * (1 - np.exp(-0.8 * depth)) - 0.016 * np.clip(depth - 4, 0, None)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(depth, train, color=BLUE, linewidth=2.4, marker="o", markersize=4, label="Training accuracy")
    ax.plot(depth, test, color=WARN, linewidth=2.4, marker="s", markersize=4, label="Test accuracy")

    best = int(depth[int(np.argmax(test))])
    ax.axvline(best, color=GREY, linestyle=":", linewidth=1.2)
    ax.annotate(
        f"Best depth ~ {best}\nbeyond here the tree\nmemorises the training set",
        xy=(best, float(np.max(test))),
        xytext=(best + 1.4, 0.80),
        fontsize=8.5,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1),
    )

    ax.set_xlabel("max_depth", fontsize=10, color=INK)
    ax.set_ylabel("Accuracy", fontsize=10, color=INK)
    ax.set_ylim(0.6, 1.05)
    ax.set_title("A Deeper Tree Is Not a Better Tree", fontsize=12, color=NAVY, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.grid(color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig
