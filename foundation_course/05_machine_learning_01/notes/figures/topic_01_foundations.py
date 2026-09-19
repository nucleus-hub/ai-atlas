"""Figures for note 01 -- Machine Learning Foundations & Types."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import BLUE, GREY, INK, NAVY, PAPER, blank_axes, figure, panel_title


@figure("ai_ml_dl_genai")
def _ai_ml_dl_genai() -> plt.Figure:
    """Nested circles showing AI > ML > Deep Learning > GenAI."""
    # Each caption sits in the clear band between its own circle and the next
    # one inward, so no text ever lands on a boundary line.
    layers = [
        # (label, subtitle, radius, facecolor, edgecolor, label_y)
        ("Artificial Intelligence", "building intelligent systems", 1.00, "#EAF1FF", NAVY, 0.885),
        ("Machine Learning", "learns patterns from data", 0.76, "#CFE0FF", BLUE, 0.650),
        ("Deep Learning", "neural networks, many layers", 0.53, "#9FC3FF", BLUE, 0.420),
        ("GenAI", "generates new content", 0.31, "#5B9BFF", NAVY, 0.045),
    ]

    fig, ax = plt.subplots(figsize=(7.2, 7.2), facecolor=PAPER)

    for label, subtitle, radius, face, edge, label_y in layers:
        ax.add_patch(
            plt.Circle((0, 0), radius, facecolor=face, edgecolor=edge, linewidth=2.0, zorder=1)
        )
        ax.text(0, label_y, label, ha="center", va="center", fontsize=12,
                fontweight="bold", color=NAVY, zorder=2)
        ax.text(0, label_y - 0.062, subtitle, ha="center", va="center", fontsize=8.5,
                style="italic", color=INK, zorder=2)

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.22)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Where Machine Learning Fits", fontsize=15, fontweight="bold", color=NAVY, pad=14)
    ax.text(
        0, -1.06,
        "Every GenAI system is Deep Learning; every DL system is ML; every ML system is AI.",
        ha="center", va="center", fontsize=9, color=INK,
    )
    fig.tight_layout()
    return fig


@figure("ml_types")
def _ml_types() -> plt.Figure:
    """Three-panel comparison of supervised, unsupervised and reinforcement learning."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=PAPER)

    panels = [
        ("Supervised", "learns with an answer key", "Labelled examples\n(features + correct answer)"),
        ("Unsupervised", "finds structure with none", "Unlabelled data\n(no correct answer)"),
        ("Reinforcement", "learns from reward & penalty", "Interaction with\nan environment"),
    ]

    group_a = ([1, 1.6, 1.2, 2.0, 1.4], [3.0, 3.6, 2.6, 3.2, 4.0])
    group_b = ([3.2, 3.8, 3.4, 4.1, 3.6], [1.2, 1.8, 0.9, 1.5, 2.2])

    # -- Supervised: two labelled classes, cleanly separated ----------------
    ax = axes[0]
    ax.scatter(*group_a, s=110, color=BLUE, label="Class A")
    ax.scatter(*group_b, s=110, color=NAVY, marker="s", label="Class B")
    ax.legend(loc="upper right", fontsize=8, frameon=False)

    # -- Unsupervised: same points, no labels, discovered rings -------------
    ax = axes[1]
    ax.scatter(*group_a, s=110, color=GREY)
    ax.scatter(*group_b, s=110, color=GREY)
    for cx, cy in [(1.44, 3.28), (3.62, 1.52)]:
        ax.add_patch(plt.Circle((cx, cy), 0.95, facecolor="none", edgecolor=BLUE,
                                linestyle="--", linewidth=1.8))
    ax.text(1.44, 4.55, "discovered group", ha="center", fontsize=8, color=BLUE, style="italic")

    # -- Reinforcement: the agent/environment feedback loop -----------------
    # Boxes stacked vertically with one arrow down each side, so neither arrow
    # nor its label crosses the other.
    ax = axes[2]
    ax.add_patch(plt.Rectangle((1.35, 3.35), 2.3, 0.9, facecolor="#CFE0FF", edgecolor=NAVY, linewidth=1.8))
    ax.text(2.5, 3.80, "Agent", ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)
    ax.add_patch(plt.Rectangle((1.35, 0.85), 2.3, 0.9, facecolor="#CFE0FF", edgecolor=NAVY, linewidth=1.8))
    ax.text(2.5, 1.30, "Environment", ha="center", va="center", fontsize=11, fontweight="bold", color=NAVY)

    ax.annotate("", xy=(3.30, 1.75), xytext=(3.30, 3.35),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2, connectionstyle="arc3,rad=-0.42"))
    ax.text(4.35, 2.55, "action", fontsize=9, color=BLUE, rotation=-90, ha="center", va="center")

    ax.annotate("", xy=(1.70, 3.35), xytext=(1.70, 1.75),
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2, connectionstyle="arc3,rad=-0.42"))
    ax.text(0.62, 2.55, "reward / penalty", fontsize=9, color=NAVY, rotation=90, ha="center", va="center")

    for ax, (title, subtitle, source) in zip(axes, panels):
        blank_axes(ax)
        panel_title(ax, title, subtitle)
        ax.set_xlabel(source, fontsize=9, color=INK, labelpad=8)

    fig.suptitle("The Three Types of Machine Learning", fontsize=15, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig


@figure("task_types")
def _task_types() -> plt.Figure:
    """Regression vs classification vs clustering, side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor=PAPER)

    # -- Regression: a best-fit line through a continuous target ------------
    ax = axes[0]
    ax.scatter([0.5, 1.2, 1.9, 2.6, 3.3, 4.0, 4.6], [1.0, 1.6, 2.0, 2.9, 3.2, 4.0, 4.4],
               s=90, color=BLUE, zorder=3)
    ax.plot([0.2, 4.9], [0.72, 4.68], color=NAVY, linewidth=2.2, zorder=2)
    panel_title(ax, "Regression", "predicts a number")
    ax.set_xlabel("Area (sq.ft)  →", fontsize=9, color=INK)
    ax.set_ylabel("Price  →", fontsize=9, color=INK)

    # -- Classification: a predefined boundary between known classes --------
    ax = axes[1]
    ax.scatter([1.0, 1.7, 1.3, 2.1, 1.5], [3.1, 3.7, 2.7, 3.4, 4.1], s=100, color=BLUE, label="Spam")
    ax.scatter([3.2, 3.9, 3.4, 4.2, 3.7], [1.2, 1.9, 0.9, 1.6, 2.3], s=100, color=NAVY,
               marker="s", label="Not spam")
    ax.plot([0.4, 4.6], [4.6, 0.4], color="#C0392B", linewidth=2.0, linestyle="--")
    ax.text(4.05, 4.15, "decision\nboundary", fontsize=8, color="#C0392B", ha="center")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    panel_title(ax, "Classification", "predicts a category")

    # -- Clustering: groups that emerge, named afterwards --------------------
    ax = axes[2]
    groups = [
        ([0.9, 1.5, 1.1, 1.8], [3.6, 4.0, 3.1, 3.7], (1.32, 3.60)),
        ([3.4, 4.0, 3.6, 4.2], [3.3, 3.8, 2.9, 3.4], (3.80, 3.35)),
        ([2.1, 2.7, 2.3, 2.9], [1.0, 1.4, 0.7, 1.1], (2.50, 1.05)),
    ]
    for i, (gx, gy, (cx, cy)) in enumerate(groups, start=1):
        ax.scatter(gx, gy, s=100, color=GREY)
        ax.add_patch(plt.Circle((cx, cy), 0.85, facecolor="none", edgecolor=BLUE,
                                linestyle="--", linewidth=1.8))
        ax.text(cx, cy + 1.02, f"Group {i}", ha="center", fontsize=8, color=BLUE, style="italic")
    panel_title(ax, "Clustering", "groups discovered from data")

    for ax in axes:
        blank_axes(ax)

    fig.suptitle("Regression vs Classification vs Clustering", fontsize=15,
                 fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig
