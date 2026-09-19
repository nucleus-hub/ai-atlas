"""Figures for Topic 08 -- Optimization Basics & Loss Functions."""

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
    figure,
    node_box,
    panel_title,
)


def _descent(theta: float, lr: float, steps: int) -> list[float]:
    """Gradient descent on J(theta) = (theta - 3)^2, matching the practical notebook."""
    history = [theta]
    for _ in range(steps):
        theta = theta - lr * 2 * (theta - 3)
        history.append(theta)
    return history


@figure("gradient_descent_path")
def gradient_descent_path() -> plt.Figure:
    """The notebook's exact run: theta = -4, lr = 0.2, 20 steps."""
    history = _descent(-4.0, 0.2, 20)
    grid = np.linspace(-5, 11, 300)

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    ax.plot(grid, (grid - 3) ** 2, color=NAVY, linewidth=2.4, label="Cost function J(theta) = (theta - 3)$^2$")
    costs = [(t - 3) ** 2 for t in history]
    ax.plot(history, costs, color=WARN, linestyle="--", alpha=0.6, linewidth=1.4)
    ax.scatter(history, costs, color=WARN, s=42, zorder=5, label="Gradient descent steps")
    ax.axvline(3, color=GOOD, linestyle=":", linewidth=1.8, label="True minimum")

    ax.annotate("start: theta = -4", xy=(history[0], costs[0]), xytext=(-4.6, 33),
                fontsize=9, color=INK, arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1))
    ax.annotate(f"after 20 steps: theta = {history[-1]:.3f}", xy=(history[-1], costs[-1]),
                xytext=(4.6, 16), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1))
    ax.text(7.4, 40, "steps shrink as\nthe curve flattens", fontsize=8.5, color=GREY, style="italic")

    ax.set_xlabel("theta", fontsize=10, color=INK)
    ax.set_ylabel("Cost", fontsize=10, color=INK)
    ax.set_title("Gradient Descent Converging to the Minimum", fontsize=12, color=NAVY, fontweight="bold")
    ax.legend(fontsize=8.5, frameon=False, loc="upper center")
    ax.grid(color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


@figure("learning_rate_choice")
def learning_rate_choice() -> plt.Figure:
    """Too small, just right, too large."""
    grid = np.linspace(-5, 11, 300)
    runs = [
        (0.02, "Too small\ncrawls -- many steps wasted", WARN),
        (0.20, "Just right\nsteady convergence", GOOD),
        (1.02, "Too large\novershoots and diverges", WARN),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7))
    for ax, (lr, title, colour) in zip(axes, runs):
        history = _descent(-4.0, lr, 18)
        costs = [(t - 3) ** 2 for t in history]
        ax.plot(grid, (grid - 3) ** 2, color=NAVY, linewidth=2.0)
        ax.plot(history, costs, color=colour, linestyle="--", linewidth=1.3, alpha=0.7)
        ax.scatter(history, costs, color=colour, s=28, zorder=5)
        ax.set_title(f"{title}\nlearning rate = {lr}", fontsize=9.5, color=NAVY, fontweight="bold")
        ax.set_ylim(-4, 90)
        ax.set_xlim(-6, 12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle("The Learning Rate Controls the Step Size", fontsize=11.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("optimization_loop")
def optimization_loop() -> plt.Figure:
    """The universal training loop shared by every model in the course."""
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    blank_axes(ax, (0, 10.4), (0, 3.6))
    panel_title(ax, "The Universal Training Loop", "Identical from linear regression to a large language model")

    steps = [
        (1.5, "Predict", "model makes\na guess"),
        (3.7, "Loss", "measure how\nwrong it was"),
        (5.9, "Gradient", "which direction\nreduces loss"),
        (8.1, "Update", "theta := theta\n- alpha * grad"),
    ]
    for i, (x, name, sub) in enumerate(steps):
        node_box(ax, (x, 2.4), name, facecolor=MIST if i else NAVY,
                 textcolor=INK if i else PAPER, edgecolor=BLUE if i else NAVY,
                 width=1.7, height=0.6, fontsize=9.5, bold=True)
        ax.text(x, 1.62, sub, ha="center", fontsize=8, color=GREY)
        if i < 3:
            ax.annotate("", xy=(steps[i + 1][0] - 0.88, 2.4), xytext=(x + 0.88, 2.4),
                        arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.6))

    ax.annotate("", xy=(1.5, 3.0), xytext=(8.1, 3.0),
                arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.4,
                                connectionstyle="arc3,rad=-0.18"))
    ax.text(4.8, 0.75, "repeat until the loss stops improving", ha="center",
            fontsize=9, color=NAVY, style="italic")
    fig.tight_layout()
    return fig


@figure("mse_vs_crossentropy")
def mse_vs_crossentropy() -> plt.Figure:
    """The two loss functions, each on the task it belongs to."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))

    error = np.linspace(-3, 3, 300)
    ax1.plot(error, error**2, color=NAVY, linewidth=2.6)
    ax1.fill_between(error, error**2, color=MIST, alpha=0.35)
    ax1.set_title("MSE -- regression\n(y - y_hat)$^2$", fontsize=10.5, color=NAVY, fontweight="bold")
    ax1.set_xlabel("prediction error", fontsize=9.5, color=INK)
    ax1.set_ylabel("loss", fontsize=9.5, color=INK)
    ax1.text(0, 7.2, "squaring punishes\nbig misses hardest", ha="center", fontsize=8.5, color=GREY)
    ax1.set_ylim(0, 9.5)

    p = np.linspace(0.001, 0.999, 300)
    ax2.plot(p, -np.log(p), color=NAVY, linewidth=2.6, label="true label = 1")
    ax2.plot(p, -np.log(1 - p), color="#E07A3F", linewidth=2.6, linestyle="--", label="true label = 0")
    ax2.set_title("Cross-Entropy -- classification\n-[y log(p) + (1-y) log(1-p)]",
                  fontsize=10.5, color=NAVY, fontweight="bold")
    ax2.set_xlabel("predicted probability", fontsize=9.5, color=INK)
    ax2.set_ylim(0, 5.2)
    ax2.legend(fontsize=8.5, frameon=False)
    ax2.text(0.5, 4.3, "confident AND wrong\n-> enormous penalty", ha="center", fontsize=8.5, color=WARN)

    for ax in (ax1, ax2):
        ax.grid(color=LINE, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Pick the Loss That Matches the Task", fontsize=11.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("loss_landscape_local_minima")
def loss_landscape_local_minima() -> plt.Figure:
    """A bowl versus a bumpy landscape -- where gradient descent can get stuck."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    x = np.linspace(-5, 11, 300)
    ax1.plot(x, (x - 3) ** 2, color=NAVY, linewidth=2.4)
    ax1.scatter([3], [0], s=90, color=GOOD, zorder=5)
    ax1.text(3, 9, "one global minimum\nany start converges", ha="center", fontsize=8.5, color=GOOD)
    ax1.set_title("Convex: the friendly case", fontsize=10.5, color=NAVY, fontweight="bold")

    x2 = np.linspace(-4, 6, 400)
    y2 = 0.28 * (x2 - 1) ** 2 + 2.6 * np.sin(1.7 * x2) + 4
    ax2.plot(x2, y2, color=NAVY, linewidth=2.4)
    gmin = int(np.argmin(y2))
    ax2.scatter([x2[gmin]], [y2[gmin]], s=90, color=GOOD, zorder=5)
    ax2.text(x2[gmin], y2[gmin] - 1.4, "global", ha="center", fontsize=8.5, color=GOOD)
    for guess in (-2.6, 3.6):
        idx = int(np.argmin(np.abs(x2 - guess)))
        ax2.scatter([x2[idx]], [y2[idx]], s=70, color=WARN, zorder=5)
    ax2.text(3.9, 9.5, "local minima -- descent\ncan stop early here", fontsize=8.5, color=WARN, ha="center")
    ax2.set_title("Non-convex: the realistic case", fontsize=10.5, color=NAVY, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("parameter", fontsize=9.5, color=INK)
        ax.set_ylabel("loss", fontsize=9.5, color=INK)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)
    fig.tight_layout()
    return fig


@figure("loss_across_models")
def loss_across_models() -> plt.Figure:
    """One mechanism, every scale of model."""
    fig, ax = plt.subplots(figsize=(9.6, 3.3))
    blank_axes(ax, (0, 10.4), (0, 3.6))
    panel_title(ax, "The Same Mechanism at Every Scale", "Gradient descent plus a loss function")

    models = [
        (1.6, "Linear\nRegression", "MSE"),
        (4.0, "Logistic\nRegression", "Cross-entropy"),
        (6.4, "Neural\nNetwork", "MSE / cross-entropy"),
        (8.8, "LLM /\nGenAI", "Cross-entropy\nover next tokens"),
    ]
    for x, name, loss in models:
        node_box(ax, (x, 2.4), name, facecolor=MIST, width=2.0, height=0.8, fontsize=9, bold=True)
        ax.text(x, 1.5, loss, ha="center", fontsize=8, color=GREY)

    ax.add_patch(plt.Rectangle((0.5, 0.55), 9.4, 0.55, facecolor="#F2F6FD", edgecolor=LINE, linewidth=1.2))
    ax.text(5.2, 0.82, "all trained by: measure the loss  ->  follow the gradient  ->  update the parameters",
            ha="center", va="center", fontsize=9, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig
