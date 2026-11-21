"""Figures for Note 04 -- Backpropagation & Gradient Descent.

Six diagrams:

* ``training_loop``              -- the four-step cycle
* ``chain_rule_flow``            -- forward values, backward gradients
* ``gradient_descent_valley``    -- actual steps descending a loss curve
* ``regression_training_curve``  -- 100 epochs, $60K -> $250K
* ``learning_rate_effect``       -- too small / just right / exploding
* ``bce_vs_mse_training``        -- 94.23% vs 85.42% on the same task
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

# The regression problem used throughout the note.
X1, X2, Y_TRUE = 0.8, 0.3, 2.5
W1_0, W2_0, B_0 = 0.4, 0.6, 0.1


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _train_regression(lr: float, epochs: int = 100):
    """Return (loss_history, pred_history) for the house-price neuron."""
    w1, w2, b = W1_0, W2_0, B_0
    losses, preds = [], []
    for _ in range(epochs):
        z = w1 * X1 + w2 * X2 + b
        losses.append((Y_TRUE - z) ** 2)
        preds.append(z)
        g = -2 * (Y_TRUE - z)
        w1 -= lr * g * X1
        w2 -= lr * g * X2
        b -= lr * g
    return np.array(losses), np.array(preds)


@figure("training_loop")
def training_loop() -> plt.Figure:
    """The four steps of training, drawn as a cycle."""
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    blank_axes(ax, xlim=(0, 12), ylim=(0, 7.4))
    for spine in ax.spines.values():
        spine.set_visible(False)

    steps = [
        (3.0, 5.4, "1 · FORWARD PASS", "Input flows through all\nlayers to produce ŷ", SKY, PAPER),
        (9.0, 5.4, "2 · COMPUTE LOSS", "Compare ŷ to the\ntrue label y", MIST, INK),
        (9.0, 1.9, "3 · BACKWARD PASS", "Chain rule finds ∂L/∂w\nfor every weight", MIST, INK),
        (3.0, 1.9, "4 · UPDATE WEIGHTS", "w = w − η·∂L/∂w\nin the direction that helps", SKY, PAPER),
    ]
    for x, y, title, body, face, textcol in steps:
        ax.add_patch(FancyBboxPatch((x - 1.85, y - 0.85), 3.7, 1.7,
                                    boxstyle="round,pad=0.06,rounding_size=0.14",
                                    facecolor=face, edgecolor=NAVY, linewidth=1.8, zorder=3))
        ax.text(x, y + 0.42, title, ha="center", va="center", fontsize=10,
                fontweight="bold", color=textcol if face is SKY else NAVY, zorder=4)
        ax.text(x, y - 0.26, body, ha="center", va="center", fontsize=8.3,
                color=textcol, zorder=4)

    arrows = [
        ((4.95, 5.4), (7.05, 5.4), "data forward", GOOD, 0.42),
        ((9.0, 4.45), (9.0, 2.85), "error", WARN, 0.0),
        ((7.05, 1.9), (4.95, 1.9), "gradients backward", WARN, -0.42),
        ((3.0, 2.85), (3.0, 4.45), "repeat", GOOD, 0.0),
    ]
    for (sx, sy), (ex, ey), label, colour, off in arrows:
        ax.add_patch(FancyArrowPatch((sx, sy), (ex, ey), arrowstyle="-|>",
                                     mutation_scale=18, color=colour, linewidth=2.0, zorder=2))
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        ax.text(mx + (0 if off else 0.62), my + off, label, ha="center", va="center",
                fontsize=8.3, color=colour, fontweight="bold",
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.4), zorder=5)

    ax.text(6.0, 6.85, "Forward propagation produces predictions.",
            ha="center", fontsize=9.6, color=GOOD, fontweight="bold")
    ax.text(6.0, 0.42, "Backpropagation improves predictions by updating weights.",
            ha="center", fontsize=9.6, color=WARN, fontweight="bold")
    ax.set_title("The training loop — four steps, repeated",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
    fig.tight_layout()
    return fig


@figure("chain_rule_flow")
def chain_rule_flow() -> plt.Figure:
    """Values flowing forward, gradients flowing back, with real numbers."""
    fig, ax = plt.subplots(figsize=(12.8, 5.4))
    blank_axes(ax, xlim=(0, 13), ylim=(0.3, 5.6))
    for spine in ax.spines.values():
        spine.set_visible(False)

    xs = [1.5, 4.4, 7.3, 10.2, 12.4]
    labels = ["w1 = 0.4", "z = 0.60", "ŷ = 0.60", "L = 3.6100", ""]
    subs = ["the weight", "wx + b", "f(z), linear", "(y − ŷ)²", ""]

    for x, lab, sub in zip(xs[:4], labels[:4], subs[:4]):
        node_box(ax, (x, 3.7), lab, facecolor=MIST, edgecolor=NAVY,
                 width=2.25, height=0.82, fontsize=10.5, bold=True)
        ax.text(x, 2.98, sub, ha="center", fontsize=8, color=GREY, style="italic")

    # Forward arrows.
    for a, b in zip(xs[:3], xs[1:4]):
        ax.add_patch(FancyArrowPatch((a + 1.2, 3.9), (b - 1.2, 3.9), arrowstyle="-|>",
                                     mutation_scale=16, color=GOOD, linewidth=1.9, zorder=2))
    ax.text(2.95, 4.32, "FORWARD — values", ha="center", fontsize=9,
            color=GOOD, fontweight="bold")

    # Backward arrows with the partial derivatives.
    grads = ["∂z/∂w = x1 = 0.8", "∂ŷ/∂z = 1", "∂L/∂ŷ = −3.80"]
    for (a, b), g in zip(zip(xs[:3], xs[1:4]), grads):
        ax.add_patch(FancyArrowPatch((b - 1.2, 1.95), (a + 1.2, 1.95), arrowstyle="-|>",
                                     mutation_scale=16, color=WARN, linewidth=1.9, zorder=2))
        ax.text((a + b) / 2, 1.52, g, ha="center", fontsize=8.4, color=WARN,
                fontweight="bold")
    ax.text(8.8, 2.38, "BACKWARD — gradients", ha="center", fontsize=9,
            color=WARN, fontweight="bold")

    ax.add_patch(FancyBboxPatch((0.55, 0.5), 11.9, 0.72,
                                boxstyle="round,pad=0.05,rounding_size=0.1",
                                facecolor=PAPER, edgecolor=NAVY, linewidth=1.5, zorder=3))
    ax.text(6.5, 0.86,
            "∂L/∂w1  =  ∂L/∂ŷ × ∂ŷ/∂z × ∂z/∂w1  =  (−3.80) × (1) × (0.8)  =  −3.0400",
            ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold", zorder=4)

    ax.set_title("The chain rule — multiply the links to get one weight's gradient",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=10)
    fig.tight_layout()
    return fig


@figure("gradient_descent_valley")
def gradient_descent_valley() -> plt.Figure:
    """Steps descending the loss curve, for two learning rates."""
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0))

    # Loss as a function of w1 alone (w2, b held at their initial values).
    w_grid = np.linspace(-0.6, 3.0, 300)
    loss_of = lambda w: (Y_TRUE - (w * X1 + W2_0 * X2 + B_0)) ** 2

    for ax, lr, title in ((axes[0], 0.25, "η = 0.25 — steady descent"),
                          (axes[1], 1.45, "η = 1.45 — overshoots and oscillates")):
        ax.plot(w_grid, loss_of(w_grid), color=SKY, linewidth=2.4, zorder=2)
        w = W1_0
        pts = [w]
        for _ in range(7):
            z = w * X1 + W2_0 * X2 + B_0
            w = w - lr * (-2 * (Y_TRUE - z)) * X1
            pts.append(w)
        pts = np.array(pts)
        ax.plot(pts, loss_of(pts), "o-", color=WARN if lr > 1 else GOOD,
                linewidth=1.7, markersize=7, zorder=4,
                markeredgecolor=PAPER, markeredgewidth=1.2)
        for i, p in enumerate(pts[:4]):
            ax.annotate(f"{i}", xy=(p, loss_of(p)), xytext=(p, loss_of(p) + 0.42),
                        ha="center", fontsize=8.5, color=NAVY, fontweight="bold")
        best = (Y_TRUE - W2_0 * X2 - B_0) / X1
        ax.axvline(best, color=GREY, linestyle=":", linewidth=1.4, zorder=1)
        ax.text(best, 4.75, "minimum", ha="center", fontsize=8.3,
                color=GREY, fontweight="bold",
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.2), zorder=6)
        ax.set_xlabel("w1", fontsize=9.5, color=INK)
        ax.set_ylabel("loss", fontsize=9.5, color=INK)
        ax.grid(alpha=0.18, color=LINE)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)
        panel_title(ax, title, "7 steps from the same start")

    axes[0].set_ylim(-0.3, 6.0)
    axes[1].set_ylim(-0.3, 6.0)
    fig.suptitle("Gradient descent walks downhill — the learning rate is the stride length",
                 fontsize=11, color=INK, y=0.035)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


@figure("regression_training_curve")
def regression_training_curve() -> plt.Figure:
    """100 epochs: loss falling, prediction climbing to the true price."""
    losses, preds = _train_regression(0.1)
    epochs = np.arange(1, 101)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axes[0]
    ax.plot(epochs, losses, color=WARN, linewidth=2.5, zorder=3)
    ax.scatter([1, 2, 3, 10], losses[[0, 1, 2, 9]], s=55, color=NAVY, zorder=5,
               edgecolor=PAPER, linewidth=1.3)
    ax.annotate(f"epoch 1: {losses[0]:.4f}", xy=(1, losses[0]), xytext=(17, 3.1),
                fontsize=8.6, color=NAVY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=NAVY, linewidth=1.1))
    ax.annotate(f"epoch 2: {losses[1]:.4f}\n−57% in ONE step", xy=(2, losses[1]),
                xytext=(21, 1.62), fontsize=8.6, color=GOOD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GOOD, linewidth=1.1))
    ax.set_xlabel("epoch", fontsize=9.5, color=INK)
    ax.set_ylabel("loss  (MSE)", fontsize=9.5, color=INK)
    ax.grid(alpha=0.18, color=LINE)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Loss falls steeply, then flattens", "the gradient shrinks as the error shrinks")

    ax = axes[1]
    ax.plot(epochs, preds * 100, color=BLUE, linewidth=2.5, zorder=3, label="prediction")
    ax.axhline(Y_TRUE * 100, color=GOOD, linewidth=1.8, linestyle="--", zorder=2,
               label="true price  $250K")
    ax.scatter([1], [preds[0] * 100], s=70, color=WARN, zorder=5,
               edgecolor=PAPER, linewidth=1.4)
    ax.annotate("$60K", xy=(1, preds[0] * 100), xytext=(9, 78),
                fontsize=9, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.2))
    ax.axvline(25, color=GREY, linestyle=":", linewidth=1.4, zorder=1)
    ax.text(27, 130, "converged by\nepoch 25", fontsize=8.6, color=GREY, fontweight="bold")
    ax.set_xlabel("epoch", fontsize=9.5, color=INK)
    ax.set_ylabel("predicted price  ($ thousands)", fontsize=9.5, color=INK)
    ax.grid(alpha=0.18, color=LINE)
    ax.legend(fontsize=8.6, loc="lower right", framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "The prediction climbs to the truth", "$60K → $250K")

    fig.tight_layout()
    return fig


@figure("learning_rate_effect")
def learning_rate_effect() -> plt.Figure:
    """Too small, just right, and exploding."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    epochs = np.arange(1, 101)

    settings = [
        (0.01, "η = 0.01 — too small", "still not there after 100 epochs", GREY),
        (0.1, "η = 0.1 — just right", "converged", GOOD),
        (1.0, "η = 1.0 — too large", "diverged to 1e+78", WARN),
    ]
    for ax, (lr, title, subtitle, colour) in zip(axes, settings):
        losses, _ = _train_regression(lr)
        losses = np.clip(losses, 1e-35, 1e80)
        ax.semilogy(epochs, losses, color=colour, linewidth=2.5, zorder=3)
        ax.set_xlabel("epoch", fontsize=9, color=INK)
        ax.set_ylabel("loss  (log scale)", fontsize=9, color=INK)
        ax.grid(alpha=0.2, color=LINE, which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)
        panel_title(ax, title, subtitle)
        ax.text(0.5, 0.06, f"final loss  {losses[-1]:.2e}", transform=ax.transAxes,
                ha="center", fontsize=9, color=colour, fontweight="bold",
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=2))

    fig.suptitle("Same problem, same 100 epochs — only the learning rate changed",
                 fontsize=11, color=INK, y=0.035)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


@figure("bce_vs_mse_training")
def bce_vs_mse_training() -> plt.Figure:
    """The same classification task trained with each loss."""
    x1, x2, y = 0.7, 0.5, 1
    lr, epochs = 0.1, 100

    def run(kind):
        w1, w2, b = 0.3, 0.5, -0.2
        probs = []
        for _ in range(epochs):
            z = w1 * x1 + w2 * x2 + b
            p = _sigmoid(z)
            d = (p - y) if kind == "BCE" else 2 * (p - y) * p * (1 - p)
            w1 -= lr * d * x1
            w2 -= lr * d * x2
            b -= lr * d
            # Record AFTER the update so the final point is the trained value.
            probs.append(_sigmoid(w1 * x1 + w2 * x2 + b))
        return np.array(probs)

    bce, mse = run("BCE"), run("MSE")
    ep = np.arange(1, epochs + 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(ep, bce * 100, color=GOOD, linewidth=2.8, label="Binary Cross-Entropy", zorder=4)
    ax.plot(ep, mse * 100, color=WARN, linewidth=2.8, label="MSE", zorder=3)
    ax.axhline(100, color=GREY, linestyle="--", linewidth=1.3, zorder=1)
    ax.fill_between(ep, mse * 100, bce * 100, color=GOOD, alpha=0.10, zorder=2)

    for series, colour, dy in ((bce, GOOD, +3.2), (mse, WARN, -4.4)):
        ax.scatter([epochs], [series[-1] * 100], s=80, color=colour, zorder=6,
                   edgecolor=PAPER, linewidth=1.5)
        ax.annotate(f"{series[-1] * 100:.2f}%", xy=(epochs, series[-1] * 100),
                    xytext=(epochs - 17, series[-1] * 100 + dy), fontsize=10,
                    color=colour, fontweight="bold")

    ax.text(52, 71, f"{(bce[-1] - mse[-1]) * 100:.2f} percentage points,\nfrom the loss function alone",
            ha="center", fontsize=9.2, color=NAVY, fontweight="bold",
            bbox=dict(facecolor=PAPER, edgecolor=NAVY, boxstyle="round,pad=0.4", linewidth=1.2))

    ax.set_xlabel("epoch", fontsize=9.5, color=INK)
    ax.set_ylabel("predicted probability of the TRUE class  (%)", fontsize=9.5, color=INK)
    ax.set_ylim(50, 104)
    ax.grid(alpha=0.18, color=LINE)
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("Identical task, identical steps — only the loss differs",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig
