"""Figures for note 05 -- Logistic Regression."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, LINE, NAVY, PAPER, SKY, WARN, blank_axes, figure, panel_title

# The exam pass/fail worked example used throughout note 05.
HOURS = np.array([1, 2, 3, 4, 5, 6, 7.])
PASSED = np.array([0, 0, 0, 0, 1, 1, 1.])
# A fitted-looking sigmoid that crosses 0.5 at ~4.5 hours.
LOGIT_B0, LOGIT_B1 = -6.4, 1.42


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


@figure("sigmoid_curve")
def _sigmoid_curve() -> plt.Figure:
    """The S-curve, its threshold, and the flattening at both ends."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6), facecolor=PAPER)
    z = np.linspace(-8, 8, 500)
    p = sigmoid(z)

    ax.axhspan(0.5, 1.04, facecolor="#EAF7EF", zorder=0)
    ax.axhspan(-0.04, 0.5, facecolor="#FDEDEC", zorder=0)
    ax.plot(z, p, color=NAVY, linewidth=2.8, zorder=3)

    ax.axhline(0.5, color=WARN, linewidth=1.8, linestyle="--", zorder=2)
    ax.axvline(0, color=GREY, linewidth=1.2, linestyle=":", zorder=2)
    ax.scatter([0], [0.5], s=130, color=WARN, zorder=4)

    ax.text(-7.6, 0.80, "predict  YES", fontsize=12, fontweight="bold", color=GOOD)
    ax.text(-7.6, 0.16, "predict  NO", fontsize=12, fontweight="bold", color=WARN)
    ax.text(5.05, 0.545, "threshold = 0.5", fontsize=9.5, color=WARN, fontweight="bold")
    ax.annotate("z = 0  ->  p = 0.5\nmaximum uncertainty", xy=(0, 0.5), xytext=(1.5, 0.245),
                fontsize=9, color=WARN, arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.4))

    for zx, label in ((-6.6, "flattens near 0\nnever reaches it"), (4.4, "flattens near 1\nnever reaches it")):
        ax.annotate(label, xy=(zx, sigmoid(zx)), xytext=(zx - 0.2, sigmoid(zx) + 0.22),
                    fontsize=8.5, color=NAVY, ha="left",
                    arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.2))

    ax.set_xlim(-8, 8)
    ax.set_ylim(-0.04, 1.10)
    ax.set_xlabel("z  =  B0 + B1*x   (the linear equation)", fontsize=10, color=INK)
    ax.set_ylabel("sigmoid(z)  =  probability", fontsize=10, color=INK)
    ax.tick_params(labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("The Sigmoid Function:  1 / (1 + e^-z)",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig


@figure("linear_vs_logistic")
def _linear_vs_logistic() -> plt.Figure:
    """A straight line predicting impossible probabilities, versus the sigmoid."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), facecolor=PAPER)
    grid = np.linspace(-0.6, 9.2, 300)

    # -- Left: linear regression forced onto a 0/1 target --------------------
    ax = axes[0]
    slope, intercept = np.polyfit(HOURS, PASSED, 1)
    line = slope * grid + intercept
    ax.axhspan(1.0, 1.75, facecolor="#FDEDEC", zorder=0)
    ax.axhspan(-0.75, 0.0, facecolor="#FDEDEC", zorder=0)
    ax.plot(grid, line, color=WARN, linewidth=2.6, zorder=3)
    ax.scatter(HOURS, PASSED, s=110, color=BLUE, zorder=4)
    ax.axhline(1, color=GREY, linewidth=1.2, linestyle=":")
    ax.axhline(0, color=GREY, linewidth=1.2, linestyle=":")

    for hx in (9.0, -0.4):
        val = slope * hx + intercept
        if val > 1 or val < 0:
            ax.scatter([hx], [val], s=120, marker="X", color=WARN, zorder=5)
            ax.annotate(f"p = {val:.2f}", xy=(hx, val), xytext=(hx - 2.4, val + (0.20 if val > 1 else -0.24)),
                        fontsize=9.5, color=WARN, fontweight="bold",
                        arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.3))

    ax.text(4.2, 1.52, "IMPOSSIBLE: probability above 1", fontsize=9, color=WARN,
            ha="center", fontweight="bold")
    ax.text(4.2, -0.60, "IMPOSSIBLE: probability below 0", fontsize=9, color=WARN,
            ha="center", fontweight="bold")
    panel_title(ax, "Linear Regression on a 0/1 target", "unbounded -- predicts nonsense")
    ax.set_ylim(-0.75, 1.75)

    # -- Right: the sigmoid, safely bounded ----------------------------------
    ax = axes[1]
    ax.plot(grid, sigmoid(LOGIT_B0 + LOGIT_B1 * grid), color=GOOD, linewidth=2.6, zorder=3)
    ax.scatter(HOURS, PASSED, s=110, color=BLUE, zorder=4)
    ax.axhline(1, color=GREY, linewidth=1.2, linestyle=":")
    ax.axhline(0, color=GREY, linewidth=1.2, linestyle=":")
    ax.axhline(0.5, color=WARN, linewidth=1.5, linestyle="--")
    ax.text(8.6, 0.54, "0.5", fontsize=9, color=WARN, fontweight="bold", ha="right")
    ax.text(4.2, 1.52, "always stays between 0 and 1", fontsize=9.5, color=GOOD,
            ha="center", fontweight="bold")
    panel_title(ax, "Logistic Regression (sigmoid)", "bounded -- always a valid probability")
    ax.set_ylim(-0.75, 1.75)

    for ax in axes:
        ax.set_xlim(-0.9, 9.4)
        ax.set_xlabel("Study Hours", fontsize=9.5, color=INK)
        ax.set_ylabel("Predicted probability of passing", fontsize=9.5, color=INK)
        ax.tick_params(labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle("Why Not Just Use Linear Regression?", fontsize=15,
                 fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig


@figure("exam_pass_fail")
def _exam_pass_fail() -> plt.Figure:
    """The seven study-hour points with the fitted probability curve."""
    fig, ax = plt.subplots(figsize=(9.6, 5.4), facecolor=PAPER)
    grid = np.linspace(0.2, 7.8, 400)
    curve = sigmoid(LOGIT_B0 + LOGIT_B1 * grid)
    crossover = -LOGIT_B0 / LOGIT_B1

    ax.plot(grid, curve, color=NAVY, linewidth=2.6, zorder=3, label="Fitted probability curve")
    ax.axhline(0.5, color=WARN, linewidth=1.6, linestyle="--", zorder=2)
    ax.axvline(crossover, color=WARN, linewidth=1.6, linestyle=":", zorder=2)
    ax.scatter([crossover], [0.5], s=140, color=WARN, zorder=5)

    fails = PASSED == 0
    ax.scatter(HOURS[fails], PASSED[fails], s=125, color=WARN, marker="s", zorder=4, label="Fail (0)")
    ax.scatter(HOURS[~fails], PASSED[~fails], s=125, color=GOOD, zorder=4, label="Pass (1)")

    ax.annotate(f"crosses 0.5 at ~{crossover:.1f} hours\nthe decision threshold",
                xy=(crossover, 0.5), xytext=(5.15, 0.30), fontsize=9.5, color=WARN,
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.4))
    ax.text(1.15, 0.80, "4 hours and below:\nlow probability of passing", fontsize=9, color=INK)
    ax.text(5.35, 0.70, "5 hours and above:\nhigh probability", fontsize=9, color=INK)
    ax.annotate("steepest here -- a small change\nin study time matters most",
                xy=(crossover + 0.30, sigmoid(LOGIT_B0 + LOGIT_B1 * (crossover + 0.30))),
                xytext=(2.30, 0.30), fontsize=8.5, color=GREY,
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.1))

    ax.set_xlim(0.2, 7.8)
    ax.set_ylim(-0.09, 1.12)
    ax.set_xticks(range(1, 8))
    ax.set_xlabel("Study Hours", fontsize=10, color=INK)
    ax.set_ylabel("P(pass)", fontsize=10, color=INK)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="center left")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("Worked Example: Predicting Exam Pass / Fail",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig


@figure("decision_boundary")
def _decision_boundary() -> plt.Figure:
    """A straight boundary it can fit, and a curved one it cannot."""
    rng = np.random.default_rng(5)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=PAPER)

    # -- Left: linearly separable -- logistic regression handles it ----------
    ax = axes[0]
    yes = rng.normal([1.55, 3.35], 0.52, size=(18, 2))
    no = rng.normal([3.55, 1.55], 0.52, size=(18, 2))
    ax.scatter(yes[:, 0], yes[:, 1], s=75, color=BLUE, label="Class: Yes")
    ax.scatter(no[:, 0], no[:, 1], s=75, color=NAVY, marker="s", label="Class: No")
    ax.plot([0.35, 4.65], [0.55, 4.45], color=WARN, linewidth=2.4, linestyle="--")
    ax.text(4.15, 4.55, "decision\nboundary", fontsize=9, color=WARN, ha="center", fontweight="bold")
    ax.text(2.5, 0.28, "B0 + B1*x1 + B2*x2 = 0   (where p = 0.5)",
            fontsize=9, color=INK, ha="center", style="italic")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    panel_title(ax, "Straight boundary", " logistic regression fits this")

    # -- Right: a ring -- no straight line can separate it -------------------
    ax = axes[1]
    angle = rng.uniform(0, 2 * np.pi, 26)
    radius = rng.uniform(1.55, 2.05, 26)
    ring = np.column_stack([2.5 + radius * np.cos(angle), 2.5 + radius * np.sin(angle)])
    core = rng.normal([2.5, 2.5], 0.42, size=(18, 2))
    ax.scatter(ring[:, 0], ring[:, 1], s=75, color=NAVY, marker="s", label="Class: No")
    ax.scatter(core[:, 0], core[:, 1], s=75, color=BLUE, label="Class: Yes")
    for offset in (-0.9, 0.0, 0.9):
        ax.plot([0.35, 4.65], [1.6 + offset, 3.4 + offset], color=WARN,
                linewidth=1.6, linestyle="--", alpha=0.75)
    ax.text(2.5, 0.28, "no straight line can separate these",
            fontsize=9.5, color=WARN, ha="center", fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    panel_title(ax, "Curved separation", " a real limitation")

    for ax in axes:
        blank_axes(ax)
        ax.set_xlabel("Feature 1", fontsize=9.5, color=INK)
        ax.set_ylabel("Feature 2", fontsize=9.5, color=INK)

    fig.suptitle("The Decision Boundary Is Always Straight",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig


@figure("threshold_dial")
def _threshold_dial() -> plt.Figure:
    """One set of predicted probabilities, read at three thresholds."""
    # Ten held-out cases: predicted probability, and the true label.
    probs = np.array([0.05, 0.18, 0.27, 0.34, 0.46, 0.55, 0.62, 0.74, 0.88, 0.95])
    truth = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 1])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), facecolor=PAPER)

    for ax, threshold, label in zip(axes, (0.3, 0.5, 0.7), ("lenient", "default", "strict")):
        predicted = (probs >= threshold).astype(int)
        tp = int(((predicted == 1) & (truth == 1)).sum())
        fp = int(((predicted == 1) & (truth == 0)).sum())
        fn = int(((predicted == 0) & (truth == 1)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        colours = [GOOD if pr == 1 else GREY for pr in predicted]
        ax.barh(range(len(probs)), probs, 0.62, color=colours)
        ax.axvline(threshold, color=WARN, linewidth=2.2, linestyle="--")

        for i, (p, t) in enumerate(zip(probs, truth)):
            ax.text(0.022, i, "actual: Yes" if t == 1 else "actual: No", fontsize=7.4,
                    color=PAPER if p > 0.30 else INK, va="center")

        panel_title(ax, f"Threshold = {threshold}", f"{label}")
        ax.text(0.5, -1.85, f"Precision {precision:.0%}   ·   Recall {recall:.0%}",
                fontsize=11, fontweight="bold", color=NAVY, ha="center")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-2.4, len(probs) - 0.3)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Predicted probability", fontsize=9, color=INK)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle("The Threshold Dial: Same Model, Same Probabilities, Different Trade-off",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.03)
    fig.tight_layout()
    return fig
