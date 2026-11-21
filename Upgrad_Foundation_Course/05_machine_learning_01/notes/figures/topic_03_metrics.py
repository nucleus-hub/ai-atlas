"""Figures for note 03 -- Model Evaluation Metrics."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, LINE, NAVY, PAPER, SKY, WARN, blank_axes, figure, panel_title

# The 10-email worked example used throughout note 03.
TP, FP, FN, TN = 3, 1, 2, 4


def _matrix_panel(ax: plt.Axes, tp: int, fp: int, fn: int, tn: int,
                  pos: str = "Spam", neg: str = "Not Spam",
                  highlight: str = "") -> None:
    """Draw a 2x2 confusion matrix.

    ``highlight`` accepts "precision" (the predicted-yes column) or
    "recall" (the actually-yes row) to dim everything else.
    """
    cells = [
        # (col, row, value, short, long, key)
        (0, 1, tp, "TP", "correctly caught", "both"),
        (1, 1, fn, "FN", "missed it", "recall"),
        (0, 0, fp, "FP", "false alarm", "precision"),
        (1, 0, tn, "TN", "correctly ignored", "neither"),
    ]

    for col, row, value, short, long, key in cells:
        lit = (
            not highlight
            or key == "both"
            or (highlight == "precision" and key == "precision")
            or (highlight == "recall" and key == "recall")
        )
        face = "#CFE0FF" if lit else "#F4F6FA"
        edge = NAVY if lit else LINE
        ax.add_patch(plt.Rectangle((col * 2.0, row * 2.0), 1.92, 1.92,
                                   facecolor=face, edgecolor=edge, linewidth=2.0 if lit else 1.0))
        ax.text(col * 2.0 + 0.96, row * 2.0 + 1.36, short, fontsize=11, fontweight="bold",
                color=NAVY if lit else GREY, ha="center")
        ax.text(col * 2.0 + 0.96, row * 2.0 + 0.86, str(value), fontsize=22, fontweight="bold",
                color=NAVY if lit else GREY, ha="center")
        ax.text(col * 2.0 + 0.96, row * 2.0 + 0.38, long, fontsize=8, style="italic",
                color=INK if lit else GREY, ha="center")

    ax.text(0.96, 4.20, f"Predicted:\n{pos}", fontsize=9.5, fontweight="bold", color=NAVY, ha="center")
    ax.text(2.96, 4.20, f"Predicted:\n{neg}", fontsize=9.5, fontweight="bold", color=NAVY, ha="center")
    ax.text(-0.30, 2.96, f"Actual:\n{pos}", fontsize=9.5, fontweight="bold", color=NAVY,
            ha="right", va="center")
    ax.text(-0.30, 0.96, f"Actual:\n{neg}", fontsize=9.5, fontweight="bold", color=NAVY,
            ha="right", va="center")

    ax.set_xlim(-1.65, 4.05)
    ax.set_ylim(-0.95, 4.85)
    ax.axis("off")


@figure("confusion_matrix")
def _confusion_matrix() -> plt.Figure:
    """The 2x2 grid, filled in with the 10-email counts."""
    fig, ax = plt.subplots(figsize=(7.6, 6.0), facecolor=PAPER)
    _matrix_panel(ax, TP, FP, FN, TN)
    ax.set_title("The Confusion Matrix\n10 emails: 5 spam, 5 real",
                 fontsize=14, fontweight="bold", color=NAVY, pad=16)
    ax.text(1.96, -0.62,
            "Rows = the truth   ·   Columns = what the model did",
            fontsize=9, color=INK, ha="center", style="italic")
    fig.tight_layout()
    return fig


@figure("precision_recall")
def _precision_recall() -> plt.Figure:
    """Precision reads a column; recall reads a row."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), facecolor=PAPER)

    _matrix_panel(axes[0], TP, FP, FN, TN, highlight="precision")
    axes[0].set_title("PRECISION reads a COLUMN\n"
                      "of everything flagged, how much was right?",
                      fontsize=12.5, fontweight="bold", color=NAVY, pad=14)
    axes[0].text(1.96, -0.66, "Precision = TP / (TP + FP) = 3 / 4 = 75%",
                 fontsize=11, color=BLUE, ha="center", fontweight="bold")

    _matrix_panel(axes[1], TP, FP, FN, TN, highlight="recall")
    axes[1].set_title("RECALL reads a ROW\n"
                      "of all real spam, how much was caught?",
                      fontsize=12.5, fontweight="bold", color=NAVY, pad=14)
    axes[1].text(1.96, -0.66, "Recall = TP / (TP + FN) = 3 / 5 = 60%",
                 fontsize=11, color=BLUE, ha="center", fontweight="bold")

    fig.suptitle("Precision vs Recall: Different Slices of the Same Table",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.03)
    fig.tight_layout()
    return fig


@figure("accuracy_trap")
def _accuracy_trap() -> plt.Figure:
    """97% accuracy sitting next to 0% recall on the cancer example."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), facecolor=PAPER,
                             gridspec_kw={"width_ratios": [1.15, 1]})

    _matrix_panel(axes[0], 0, 0, 30, 970, pos="Cancer", neg="Healthy")
    axes[0].set_title("1,000 patients — the model says\n\"healthy\" to everyone",
                      fontsize=12.5, fontweight="bold", color=NAVY, pad=14)
    axes[0].text(1.96, -0.66, "It never once predicts cancer.",
                 fontsize=10, color=WARN, ha="center", fontweight="bold")

    ax = axes[1]
    bars = [("Accuracy", 0.97, GOOD), ("Recall", 0.00, WARN)]
    for i, (label, value, colour) in enumerate(bars):
        ax.bar(i, max(value, 0.004), 0.5, color=colour)
        ax.text(i, value + 0.055, f"{value:.0%}", fontsize=20, fontweight="bold",
                color=colour, ha="center")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0%", "50%", "100%"], fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("Same model, two verdicts",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=14)
    ax.text(0.5, 0.62, "accuracy says\n\"excellent\"", fontsize=9.5, color=GOOD,
            ha="center", style="italic")
    ax.text(0.5, 0.30, "recall says\n\"catastrophic\"", fontsize=9.5, color=WARN,
            ha="center", style="italic")

    fig.suptitle("The Accuracy Trap on Imbalanced Data", fontsize=15,
                 fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig


@figure("regression_metrics")
def _regression_metrics() -> plt.Figure:
    """RMSE measures error against the fit; R-squared compares it to the mean."""
    houses = np.array([1, 2, 3, 4])
    actual = np.array([50, 60, 55, 70.])
    predicted = np.array([48, 63, 54, 66.])
    mean = actual.mean()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), facecolor=PAPER)

    # -- Left: residuals against the model's own predictions (RMSE) ---------
    ax = axes[0]
    ax.plot(houses, predicted, color=NAVY, linewidth=2.0, marker="s", markersize=8,
            label="Model prediction", zorder=2)
    ax.scatter(houses, actual, s=110, color=BLUE, zorder=3, label="Actual price")
    for h, a, p in zip(houses, actual, predicted):
        ax.plot([h, h], [a, p], color=WARN, linewidth=2.0, linestyle="--", zorder=1)
        ax.text(h + 0.10, (a + p) / 2, f"{a - p:+.0f}", fontsize=9.5, color=WARN,
                fontweight="bold", va="center")
    panel_title(ax, "RMSE — error against the model", "square, average, then root: 2.74 lakhs")
    ax.text(2.5, 45.6, "RSS = 4 + 9 + 1 + 16 = 30", fontsize=10, color=WARN,
            ha="center", fontweight="bold")

    # -- Right: the same points against the always-the-mean baseline --------
    ax = axes[1]
    ax.axhline(mean, color=GREY, linewidth=2.0, linestyle="-",
               label=f"Baseline: always {mean}")
    ax.scatter(houses, actual, s=110, color=BLUE, zorder=3, label="Actual price")
    for h, a in zip(houses, actual):
        ax.plot([h, h], [a, mean], color=GOOD, linewidth=2.0, linestyle="--", zorder=1)
        ax.text(h + 0.10, (a + mean) / 2, f"{a - mean:+.2f}", fontsize=9, color=GOOD,
                fontweight="bold", va="center")
    panel_title(ax, "R² — error against the baseline", "how much better than guessing the mean?")
    ax.text(2.5, 45.6, "TSS = 218.75    →    R² = 1 − 30/218.75 = 0.863",
            fontsize=10, color=GOOD, ha="center", fontweight="bold")

    for ax in axes:
        ax.set_xlim(0.5, 4.6)
        ax.set_ylim(44, 74)
        ax.set_xticks(houses)
        ax.set_xticklabels([f"House {h}" for h in houses], fontsize=9)
        ax.set_ylabel("Price (lakhs)", fontsize=9, color=INK)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle("Regression Metrics: RMSE and R² on the Same Four Houses",
                 fontsize=15, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig
