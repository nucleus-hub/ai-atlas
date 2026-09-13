"""Figures for Note 01 -- Introduction to Deep Learning & the Artificial Neuron.

Four diagrams:

* ``ai_ml_dl_nesting``               -- AI > ML > DL, with what makes DL "deep"
* ``biological_vs_artificial_neuron`` -- side-by-side with the mapping arrows
* ``neuron_anatomy``                 -- x -> w -> sum+b -> f -> yhat, real numbers
* ``weights_bias_effect``            -- how w rotates and b shifts a boundary
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

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
    figure,
    node_box,
    panel_title,
)


@figure("ai_ml_dl_nesting")
def ai_ml_dl_nesting() -> plt.Figure:
    """Three nested circles: AI contains ML contains DL."""
    fig, ax = plt.subplots(figsize=(9, 6.2))
    blank_axes(ax, xlim=(0, 10), ylim=(0.35, 7))
    for spine in ax.spines.values():
        spine.set_visible(False)

    rings = [
        (3.05, MIST, "Artificial Intelligence", "Any machine that mimics\nhuman-like capability", 6.28),
        (2.10, SKY, "Machine Learning", "Learns patterns from data,\nnot hand-written rules", 5.22),
        (1.05, BLUE, "Deep Learning", "", 0.0),
    ]
    for radius, colour, label, sub, label_y in rings:
        ax.add_patch(Circle((3.6, 3.5), radius, facecolor=colour, edgecolor=PAPER, linewidth=2.5, zorder=1))
        if label_y:
            ax.text(3.6, label_y, label, ha="center", va="center", fontsize=11,
                    fontweight="bold", color=NAVY, zorder=4)
            ax.text(3.6, label_y - 0.42, sub, ha="center", va="center", fontsize=8,
                    color=INK, style="italic", zorder=4)

    ax.text(3.6, 3.78, "Deep\nLearning", ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=PAPER, zorder=4)
    ax.text(3.6, 3.05, "multi-layer\nneural networks", ha="center", va="center",
            fontsize=7.2, color=PAPER, style="italic", zorder=4)

    # Call-out panel: what makes DL "deep".
    ax.add_patch(FancyBboxPatch((7.0, 2.0), 2.85, 3.0, boxstyle="round,pad=0.08,rounding_size=0.12",
                                facecolor=PAPER, edgecolor=BLUE, linewidth=1.6, zorder=3))
    ax.text(8.42, 4.72, "What makes it DEEP", ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=NAVY, zorder=4)
    ax.annotate("", xy=(6.95, 3.5), xytext=(4.72, 3.5),
                arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.8), zorder=2)

    lines = [
        ("Input layer", MIST),
        ("Hidden layer 1", SKY),
        ("Hidden layer 2", SKY),
        ("Hidden layer 3", SKY),
        ("Output layer", BLUE),
    ]
    for i, (name, colour) in enumerate(lines):
        y = 4.28 - i * 0.42
        ax.add_patch(FancyBboxPatch((7.35, y - 0.14), 2.15, 0.28,
                                    boxstyle="round,pad=0.01,rounding_size=0.06",
                                    facecolor=colour, edgecolor="none", zorder=4))
        ax.text(8.42, y, name, ha="center", va="center", fontsize=8,
                color=INK if colour is not BLUE else PAPER, zorder=5)

    ax.text(8.42, 2.25, '"Deep" = a COUNT of layers,\nnot a compliment',
            ha="center", va="center", fontsize=7.8, color=WARN,
            fontweight="bold", zorder=5)

    ax.set_title("Deep Learning sits inside Machine Learning, which sits inside AI",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig


@figure("biological_vs_artificial_neuron")
def biological_vs_artificial_neuron() -> plt.Figure:
    """Biological neuron beside the artificial one, with the mapping between."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Left: biological ---------------------------------------------------
    ax = axes[0]
    blank_axes(ax, xlim=(0, 10), ylim=(0, 7))
    panel_title(ax, "Biological Neuron", "the inspiration")

    ax.add_patch(Circle((3.6, 3.5), 1.05, facecolor=MIST, edgecolor=NAVY, linewidth=1.8, zorder=3))
    ax.add_patch(Circle((3.6, 3.5), 0.34, facecolor=SKY, edgecolor=NAVY, linewidth=1.0, zorder=4))
    ax.text(3.6, 2.18, "cell body\n(soma)", ha="center", va="center", fontsize=8.5,
            color=NAVY, fontweight="bold", zorder=5)

    for dy in (1.35, 0.55, -0.35, -1.15):
        ax.plot([0.75, 2.62], [3.5 + dy * 1.05, 3.5 + dy * 0.42], color=GREY, linewidth=1.5, zorder=2)
    ax.text(1.15, 5.6, "dendrites", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(1.15, 5.18, "(receive signals)", ha="center", fontsize=7.5, color=GREY, style="italic")

    ax.plot([4.62, 7.6], [3.5, 3.5], color=NAVY, linewidth=3.0, zorder=2)
    ax.text(6.0, 3.9, "axon", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(6.0, 3.05, "(carries the signal)", ha="center", fontsize=7.5, color=GREY, style="italic")

    for dy in (0.85, 0.0, -0.85):
        ax.plot([7.6, 8.9], [3.5, 3.5 + dy], color=NAVY, linewidth=1.8, zorder=2)
        ax.add_patch(Circle((8.98, 3.5 + dy), 0.16, facecolor=BLUE, edgecolor="none", zorder=3))
    ax.text(8.75, 5.05, "synapses", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(8.75, 4.63, "(strength varies)", ha="center", fontsize=7.5, color=GREY, style="italic")

    ax.text(5.0, 0.75, "Fires only if the combined signal crosses a threshold.",
            ha="center", fontsize=8.5, color=WARN, fontweight="bold")

    # ---- Right: artificial --------------------------------------------------
    ax = axes[1]
    blank_axes(ax, xlim=(0, 10), ylim=(0, 7))
    panel_title(ax, "Artificial Neuron", "the implementation")

    inputs = [(1.15, 4.9, "x1"), (1.15, 3.5, "x2"), (1.15, 2.1, "x3")]
    weights = ["w1", "w2", "w3"]
    for (x, y, label), w in zip(inputs, weights):
        ax.add_patch(Circle((x, y), 0.42, facecolor=MIST, edgecolor=NAVY, linewidth=1.5, zorder=3))
        ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold",
                color=NAVY, zorder=4)
        ax.annotate("", xy=(4.05, 3.5), xytext=(x + 0.45, y),
                    arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.5), zorder=2)
        mx, my = (x + 4.05) / 2, (y + 3.5) / 2
        ax.text(mx, my + 0.22, w, ha="center", va="center", fontsize=9,
                color=BLUE, fontweight="bold",
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.2), zorder=5)

    ax.text(1.15, 5.85, "inputs", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.text(2.62, 5.85, "weights", ha="center", fontsize=8.5, color=BLUE, fontweight="bold")

    ax.add_patch(Circle((4.55, 3.5), 0.78, facecolor=SKY, edgecolor=NAVY, linewidth=1.8, zorder=3))
    ax.text(4.55, 3.5, "Σ", ha="center", va="center", fontsize=20, fontweight="bold",
            color=PAPER, zorder=4)
    ax.text(4.55, 2.32, "weighted sum\nz = Σwx + b", ha="center", va="center", fontsize=8,
            color=NAVY, fontweight="bold", zorder=5)

    ax.add_patch(Circle((4.55, 5.55), 0.30, facecolor=PAPER, edgecolor=WARN, linewidth=1.5, zorder=3))
    ax.text(4.55, 5.55, "b", ha="center", va="center", fontsize=10, fontweight="bold",
            color=WARN, zorder=4)
    ax.annotate("", xy=(4.55, 4.32), xytext=(4.55, 5.23),
                arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.5), zorder=2)
    ax.text(5.35, 5.55, "bias", ha="left", va="center", fontsize=8.5, color=WARN, fontweight="bold")

    node_box(ax, (6.85, 3.5), "f(z)", facecolor=PAPER, edgecolor=BLUE,
             width=1.15, height=0.85, fontsize=12, bold=True)
    ax.text(6.85, 2.55, "activation", ha="center", fontsize=8, color=NAVY, fontweight="bold")
    ax.annotate("", xy=(6.25, 3.5), xytext=(5.36, 3.5),
                arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.6), zorder=2)

    ax.add_patch(Circle((8.95, 3.5), 0.45, facecolor=BLUE, edgecolor=NAVY, linewidth=1.5, zorder=3))
    ax.text(8.95, 3.5, "ŷ", ha="center", va="center", fontsize=13, fontweight="bold",
            color=PAPER, zorder=4)
    ax.annotate("", xy=(8.45, 3.5), xytext=(7.46, 3.5),
                arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.6), zorder=2)
    ax.text(8.95, 4.35, "output", ha="center", fontsize=8.5, color=INK, fontweight="bold")

    ax.text(5.0, 0.75, "Fires according to the activation applied to that same combined signal.",
            ha="center", fontsize=8.2, color=GOOD, fontweight="bold")

    fig.suptitle("dendrites → inputs  ·  synapse strength → weights  ·  soma → weighted sum  ·  threshold → activation",
                 fontsize=10, color=INK, y=0.035)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


@figure("neuron_anatomy")
def neuron_anatomy() -> plt.Figure:
    """The two steps of a neuron, with the house-price numbers flowing through."""
    fig, ax = plt.subplots(figsize=(12.5, 4.9))
    blank_axes(ax, xlim=(0, 13), ylim=(1.1, 6.1))
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Inputs with real values.
    rows = [(4.45, "x1 = 0.8", "size  (1600 sq ft)", "w1 = 0.4"),
            (2.35, "x2 = 0.3", "rooms (3 rooms)", "w2 = 0.6")]
    for y, val, meaning, w in rows:
        node_box(ax, (1.55, y), val, facecolor=MIST, edgecolor=NAVY,
                 width=1.95, height=0.72, fontsize=10.5, bold=True)
        ax.text(1.55, y - 0.62, meaning, ha="center", fontsize=7.8, color=GREY, style="italic")
        ax.annotate("", xy=(4.55, 3.4), xytext=(2.6, y),
                    arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.6), zorder=2)
        ax.text((2.6 + 4.55) / 2, (y + 3.4) / 2 + 0.26, w, ha="center", va="center",
                fontsize=9.5, color=BLUE, fontweight="bold",
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.4), zorder=5)

    # Bias.
    node_box(ax, (3.6, 5.5), "b = 0.1", facecolor=PAPER, edgecolor=WARN,
             width=1.25, height=0.55, fontsize=9.5, bold=True)
    ax.annotate("", xy=(5.15, 4.32), xytext=(3.9, 5.2),
                arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.5), zorder=5)

    # Step 1 box.
    ax.add_patch(FancyBboxPatch((4.6, 2.5), 3.15, 1.85,
                                boxstyle="round,pad=0.06,rounding_size=0.12",
                                facecolor=SKY, edgecolor=NAVY, linewidth=1.8, zorder=3))
    ax.text(6.17, 4.02, "STEP 1 — weighted sum", ha="center", fontsize=8.8,
            fontweight="bold", color=NAVY, zorder=4)
    ax.text(6.17, 3.45, "z = w1·x1 + w2·x2 + b", ha="center", fontsize=10,
            color=PAPER, fontweight="bold", zorder=4)
    ax.text(6.17, 3.02, "= 0.32 + 0.18 + 0.1", ha="center", fontsize=9, color=PAPER, zorder=4)
    ax.text(6.17, 2.68, "z = 0.60", ha="center", fontsize=11, color=INK,
            fontweight="bold", zorder=4)

    # Step 2 box.
    ax.add_patch(FancyBboxPatch((8.35, 2.5), 2.5, 1.85,
                                boxstyle="round,pad=0.06,rounding_size=0.12",
                                facecolor=PAPER, edgecolor=BLUE, linewidth=1.8, zorder=3))
    ax.text(9.6, 4.02, "STEP 2 — activation", ha="center", fontsize=8.8,
            fontweight="bold", color=NAVY, zorder=4)
    ax.text(9.6, 3.42, "ŷ = f(z)", ha="center", fontsize=11, color=INK,
            fontweight="bold", zorder=4)
    ax.text(9.6, 3.0, "Linear:  f(z) = z", ha="center", fontsize=8.5, color=BLUE, zorder=4)
    ax.text(9.6, 2.68, "ŷ = 0.60", ha="center", fontsize=11, color=INK,
            fontweight="bold", zorder=4)
    ax.annotate("", xy=(8.28, 3.4), xytext=(7.82, 3.4),
                arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.8), zorder=2)

    # Output.
    node_box(ax, (12.0, 3.4), "$60,000", facecolor=BLUE, edgecolor=NAVY,
             textcolor=PAPER, width=1.7, height=0.78, fontsize=10.5, bold=True)
    ax.annotate("", xy=(11.12, 3.4), xytext=(10.9, 3.4),
                arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.8), zorder=2)
    ax.text(12.0, 2.72, "true price:\n$250,000", ha="center", fontsize=8,
            color=WARN, fontweight="bold")

    ax.text(6.5, 1.72, "Every neuron in every network does exactly these two steps.",
            ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.text(6.5, 1.34,
            "The prediction is badly wrong because the weights were chosen arbitrarily — "
            "nothing has been learned yet.",
            ha="center", fontsize=8.5, color=GREY, style="italic")

    ax.set_title("Anatomy of a single neuron — predicting a house price",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
    fig.tight_layout()
    return fig


@figure("weights_bias_effect")
def weights_bias_effect() -> plt.Figure:
    """Weights rotate the decision boundary; bias shifts it."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    xs = np.linspace(-1.5, 3.5, 200)

    def draw(ax, cases, title, subtitle):
        for (w1, w2, b, colour, style, label) in cases:
            # Boundary is where z = 0  ->  w1*x + w2*y + b = 0
            ys = -(w1 * xs + b) / w2
            ax.plot(xs, ys, color=colour, linewidth=2.2, linestyle=style, label=label, zorder=3)
        ax.axhline(0, color=LINE, linewidth=1.0, zorder=1)
        ax.axvline(0, color=LINE, linewidth=1.0, zorder=1)
        ax.set_xlim(-1.5, 3.5)
        ax.set_ylim(-1.5, 3.5)
        ax.set_xlabel("x1", fontsize=9, color=INK)
        ax.set_ylabel("x2", fontsize=9, color=INK)
        ax.grid(alpha=0.18, color=LINE)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
        panel_title(ax, title, subtitle)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    draw(axes[0],
         [(1.0, 1.0, 0.0, BLUE, "-", "w = [1.0, 1.0]"),
          (2.0, 1.0, 0.0, SKY, "--", "w = [2.0, 1.0]"),
          (1.0, 2.0, 0.0, GREY, ":", "w = [1.0, 2.0]")],
         "Weights ROTATE the boundary", "bias fixed at b = 0")

    draw(axes[1],
         [(1.0, 1.0, -2.0, GREY, ":", "b = -2.0"),
          (1.0, 1.0, 0.0, BLUE, "-", "b =  0.0"),
          (1.0, 1.0, 2.0, GOOD, "--", "b = +2.0")],
         "Bias SHIFTS the boundary", "weights fixed at w = [1, 1]")

    # Third panel: why bias is not optional.
    ax = axes[2]
    ys_no_bias = -(1.0 * xs) / 1.0
    ax.plot(xs, ys_no_bias, color=WARN, linewidth=2.4, label="no bias — pinned to origin", zorder=3)
    for b, alpha in ((-2.0, 0.45), (-1.0, 0.6), (1.0, 0.6), (2.0, 0.45)):
        ax.plot(xs, -(1.0 * xs + b) / 1.0, color=GOOD, linewidth=1.5,
                alpha=alpha, zorder=2)
    ax.plot([], [], color=GOOD, linewidth=1.5, label="with bias — free to move")
    ax.scatter([0], [0], s=90, color=WARN, zorder=5, edgecolor=PAPER, linewidth=1.4)
    ax.annotate("every boundary\nforced through here",
                xy=(0, 0), xytext=(1.75, 0.95), fontsize=7.8, color=WARN,
                fontweight="bold", ha="center",
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5),
                arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.3))
    ax.axhline(0, color=LINE, linewidth=1.0, zorder=1)
    ax.axvline(0, color=LINE, linewidth=1.0, zorder=1)
    ax.set_xlim(-1.5, 3.5)
    ax.set_ylim(-1.5, 3.5)
    ax.set_xlabel("x1", fontsize=9, color=INK)
    ax.set_ylabel("x2", fontsize=9, color=INK)
    ax.grid(alpha=0.18, color=LINE)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    panel_title(ax, "Why bias is not optional", "without b, z = 0 whenever x = 0")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)

    fig.tight_layout()
    return fig
