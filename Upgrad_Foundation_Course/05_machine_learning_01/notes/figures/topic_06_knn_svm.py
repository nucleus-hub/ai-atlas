"""Figures for note 06 -- KNN & SVM.

Numbers here are either the toy-dataset values straight from the practical
notebook, or reproduced by re-running the notebook's documented preprocessing
steps (same RANDOM_STATE=42) against the same Titanic CSV -- see the note's
"honest flag" callout for why the elbow-scan number is treated as the
reliable one.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from .core import BLUE, GOOD, GREY, INK, LINE, NAVY, PAPER, WARN, panel_title, figure

# -- KNN toy example: two clusters, one query point, K=3 -----------------
KNN_CLASS0 = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
KNN_CLASS1 = np.array([[8, 8], [8, 9], [9, 8], [9, 9]])
KNN_QUERY = np.array([2.5, 2.5])
KNN_NEIGHBORS = np.array([[2, 2], [2, 1], [1, 2]])  # the 3 nearest, real distances below
KNN_DISTANCES = np.array([0.71, 1.58, 1.58])

# -- Elbow scan: reproduced by re-running the notebook's exact pipeline --
# (same RANDOM_STATE=42, same preprocessing) against seaborn's titanic.csv.
ELBOW_K = np.arange(1, 21)
ELBOW_ERROR = np.array([
    0.2691, 0.2646, 0.2197, 0.2152, 0.2242, 0.2287, 0.2332, 0.2332, 0.2197, 0.2197,
    0.2152, 0.2152, 0.2197, 0.2018, 0.2108, 0.2197, 0.2332, 0.2332, 0.2377, 0.2242,
])
BEST_K = 14

# -- SVM toy example: same 8-point dataset as the notebook ---------------
SVM_TOY_X = np.array([
    [1, 2], [2, 1], [2, 3], [3, 2],   # Class 0
    [6, 7], [7, 6], [7, 8], [8, 7],   # Class 1
])
SVM_TOY_Y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# -- Final 3-way comparison: exact numbers from the notebook's last cell -
COMPARISON_METRICS = ["Accuracy", "Precision", "Recall", "F1 Score"]
COMPARISON = {
    "Logistic Regression": [0.7758, 0.7195, 0.6860, 0.7024],
    "KNN": [0.7623, 0.7200, 0.6279, 0.6708],
    "SVM": [0.7668, 0.7125, 0.6628, 0.6867],
}


@figure("knn_vote_toy")
def _knn_vote_toy() -> plt.Figure:
    """The 8-point toy dataset, the query point, and its 3 nearest neighbors."""
    fig, ax = plt.subplots(figsize=(8.4, 6.2), facecolor=PAPER)

    ax.scatter(*KNN_CLASS0.T, s=140, color=BLUE, zorder=4, label="Class 0")
    ax.scatter(*KNN_CLASS1.T, s=140, color=NAVY, marker="s", zorder=4, label="Class 1")
    ax.scatter(*KNN_QUERY, s=220, color=WARN, marker="*", zorder=6, label="New point (2.5, 2.5)")

    # Hand-placed label positions -- a generic midpoint offset collides with
    # the star and with each other on this particular toy layout.
    label_positions = [(2.90, 2.85), (2.75, 1.30), (1.15, 2.85)]
    for neighbor, dist, label_xy in zip(KNN_NEIGHBORS, KNN_DISTANCES, label_positions):
        ax.plot([KNN_QUERY[0], neighbor[0]], [KNN_QUERY[1], neighbor[1]],
                color=WARN, linestyle="--", linewidth=1.4, zorder=3)
        ax.annotate(f"{dist}", xy=label_xy, fontsize=8.5, color=WARN, fontweight="bold")
        ax.scatter(*neighbor, s=260, facecolors="none", edgecolors=GOOD,
                   linewidth=2, zorder=5)

    ax.text(2.5, 4.3, "K=3 nearest neighbors\nall Class 0 -> unanimous vote",
            fontsize=10, color=GOOD, ha="center", fontweight="bold")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Feature 1", fontsize=10, color=INK)
    ax.set_ylabel("Feature 2", fontsize=10, color=INK)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("KNN Toy Example: The K=3 Vote", fontsize=14, fontweight="bold",
                 color=NAVY, pad=12)
    fig.tight_layout()
    return fig


@figure("knn_elbow_curve")
def _knn_elbow_curve() -> plt.Figure:
    """Real error-rate-vs-K scan, with the K=14 minimum marked."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6), facecolor=PAPER)

    ax.plot(ELBOW_K, ELBOW_ERROR, color=NAVY, linewidth=2.2, marker="o",
            markersize=5, zorder=3)
    best_idx = int(np.where(ELBOW_K == BEST_K)[0][0])
    ax.scatter([BEST_K], [ELBOW_ERROR[best_idx]], s=170, color=WARN, zorder=5)
    ax.axvline(BEST_K, color=WARN, linewidth=1.4, linestyle=":", zorder=2)
    ax.annotate(f"Best K = {BEST_K}\nerror = {ELBOW_ERROR[best_idx]}",
                xy=(BEST_K, ELBOW_ERROR[best_idx]), xytext=(BEST_K + 1.6, ELBOW_ERROR[best_idx] + 0.022),
                fontsize=9.5, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=1.4))

    ax.text(2.0, 0.267, "K too small:\njagged, overfits noise", fontsize=8.5,
            color=GREY, ha="center")
    ax.text(18.5, 0.240, "K too large:\noversmoothed,\nunderfits", fontsize=8.5,
            color=GREY, ha="center")

    ax.set_xticks(ELBOW_K)
    ax.set_xlabel("Number of Neighbors (K)", fontsize=10, color=INK)
    ax.set_ylabel("Test-Set Error Rate", fontsize=10, color=INK)
    ax.tick_params(labelsize=8.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("Choosing K: The Elbow Method (Titanic Data)", fontsize=14,
                 fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig


@figure("svm_margin_toy")
def _svm_margin_toy() -> plt.Figure:
    """The toy dataset's margin, boundary, and its 2 support vectors."""
    svm = SVC(kernel="linear", C=1.0).fit(SVM_TOY_X, SVM_TOY_Y)

    fig, ax = plt.subplots(figsize=(8.4, 6.2), facecolor=PAPER)

    xx = np.linspace(0, 9, 200)
    w = svm.coef_[0]
    b = svm.intercept_[0]
    yy_boundary = -(w[0] * xx + b) / w[1]
    margin = 1 / np.sqrt(np.sum(w ** 2))
    stretch = np.sqrt(1 + (w[0] / w[1]) ** 2)
    yy_up = yy_boundary + margin * stretch
    yy_down = yy_boundary - margin * stretch

    ax.plot(xx, yy_boundary, color=NAVY, linewidth=2.4, zorder=3, label="Decision boundary")
    ax.plot(xx, yy_up, color=NAVY, linestyle="--", linewidth=1.3, zorder=2, label="Margin edge")
    ax.plot(xx, yy_down, color=NAVY, linestyle="--", linewidth=1.3, zorder=2)
    ax.fill_between(xx, yy_down, yy_up, color=NAVY, alpha=0.06, zorder=1)

    class0 = SVM_TOY_X[SVM_TOY_Y == 0]
    class1 = SVM_TOY_X[SVM_TOY_Y == 1]
    ax.scatter(*class0.T, s=140, color=BLUE, zorder=4, label="Class 0")
    ax.scatter(*class1.T, s=140, color=WARN, marker="s", zorder=4, label="Class 1")
    ax.scatter(*svm.support_vectors_.T, s=280, facecolors="none", edgecolors=GOOD,
               linewidth=2.2, zorder=5, label="Support vectors")

    for sv in svm.support_vectors_:
        ax.annotate("support\nvector", xy=sv, xytext=(sv[0] + 0.6, sv[1] - 1.1),
                    fontsize=8.5, color=GOOD, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.2))

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_xlabel("Feature 1", fontsize=10, color=INK)
    ax.set_ylabel("Feature 2", fontsize=10, color=INK)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("SVM Toy Example: The Widest Road and Its 2 Support Vectors",
                 fontsize=13.5, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig


@figure("model_comparison_bar")
def _model_comparison_bar() -> plt.Figure:
    """Logistic Regression vs KNN vs SVM, same Titanic test set."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6), facecolor=PAPER)

    x = np.arange(len(COMPARISON_METRICS))
    width = 0.26
    colors = {"Logistic Regression": NAVY, "KNN": BLUE, "SVM": WARN}

    for i, (model, values) in enumerate(COMPARISON.items()):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=model, color=colors[model], zorder=3)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
                    fontsize=7.6, ha="center", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(COMPARISON_METRICS, fontsize=10)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Score", fontsize=10, color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper right", title="Model")
    ax.grid(axis="y", color=LINE, linewidth=0.8, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("Logistic Regression vs KNN vs SVM: Same Titanic Test Set",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig
