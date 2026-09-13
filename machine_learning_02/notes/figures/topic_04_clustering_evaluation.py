"""Figures for Topic 04 -- Evaluating Clustering (Silhouette Score & ARI)."""

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
    WARN,
    figure,
)

CLUSTER_COLOURS = (NAVY, "#E07A3F", GOOD)


def _blobs(spread: float, seed: int = 21) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = np.array([[1.6, 4.2], [4.6, 4.6], [3.2, 1.5]])
    pts, labels = [], []
    for i, c in enumerate(centres):
        pts.append(rng.normal(c, spread, size=(40, 2)))
        labels.append(np.full(40, i))
    return np.vstack(pts), np.concatenate(labels)


def _silhouette_values(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Silhouette score per point, computed directly from the definition."""
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    scores = np.zeros(len(points))
    for i in range(len(points)):
        own = labels == labels[i]
        own[i] = False
        a = dist[i, own].mean() if own.any() else 0.0
        b = min(dist[i, labels == other].mean() for other in set(labels) - {labels[i]})
        scores[i] = (b - a) / max(a, b)
    return scores


@figure("silhouette_a_and_b")
def silhouette_a_and_b() -> plt.Figure:
    """What a and b actually measure for one highlighted point."""
    points, labels = _blobs(0.55)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))

    for k in range(3):
        sel = labels == k
        ax.scatter(points[sel, 0], points[sel, 1], s=26, color=CLUSTER_COLOURS[k], alpha=0.45)

    idx = int(np.argmin(np.linalg.norm(points - np.array([2.3, 4.3]), axis=1)))
    p = points[idx]
    own_centre = points[labels == labels[idx]].mean(axis=0)
    other = 1 if labels[idx] == 0 else 0
    other_centre = points[labels == other].mean(axis=0)

    ax.scatter(*p, s=140, color=WARN, edgecolor=PAPER, linewidth=1.6, zorder=8)
    ax.annotate("", xy=own_centre, xytext=p, arrowprops=dict(arrowstyle="->", color=NAVY, linewidth=2.2))
    ax.annotate("", xy=other_centre, xytext=p, arrowprops=dict(arrowstyle="->", color="#E07A3F", linewidth=2.2))

    ax.text(*(p + [-0.55, -1.05]), "a = average distance\nto its OWN cluster",
            fontsize=9, color=NAVY, ha="center",
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5))
    ax.text(*(other_centre + [0.15, -1.25]), "b = average distance to the\nNEAREST OTHER cluster",
            fontsize=9, color="#E07A3F", ha="center",
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5))
    ax.text(3.1, 0.25, "s = (b - a) / max(a, b)", ha="center", fontsize=11, color=INK, fontweight="bold",
            bbox=dict(facecolor="#F2F6FD", edgecolor=LINE, boxstyle="round,pad=0.45"))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("The Silhouette Score for a Single Point", fontsize=12, color=NAVY, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    fig.tight_layout()
    return fig


@figure("silhouette_score_bands")
def silhouette_score_bands() -> plt.Figure:
    """The -1 to +1 interpretation scale."""
    fig, ax = plt.subplots(figsize=(9.0, 2.5))
    bands = [
        (-1.0, 0.0, "#F5B7B1", "-1.0 to 0\nlikely in the WRONG cluster"),
        (0.0, 0.5, "#FDEBD0", "0 to +0.5\nweak or overlapping clusters"),
        (0.5, 0.7, "#D6EAF8", "+0.5 to +0.7\nreasonable structure"),
        (0.7, 1.0, "#ABEBC6", "+0.7 to +1.0\nstrong, well-separated"),
    ]
    for lo, hi, colour, label in bands:
        ax.axvspan(lo, hi, color=colour)
        ax.text((lo + hi) / 2, 0.5, label, ha="center", va="center", fontsize=8.5, color=INK)

    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([-1, -0.5, 0, 0.5, 0.7, 1])
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Silhouette Score", fontsize=10, color=INK)
    ax.set_title("Reading a Silhouette Score", fontsize=12, color=NAVY, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    fig.tight_layout()
    return fig


@figure("silhouette_good_vs_bad")
def silhouette_good_vs_bad() -> plt.Figure:
    """Well-separated versus overlapping clusters, with their silhouette plots."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))

    for col, (spread, caption) in enumerate(((0.45, "Well-separated"), (1.25, "Overlapping"))):
        points, labels = _blobs(spread)
        scores = _silhouette_values(points, labels)

        ax = axes[0][col]
        for k in range(3):
            sel = labels == k
            ax.scatter(points[sel, 0], points[sel, 1], s=22, color=CLUSTER_COLOURS[k], alpha=0.75)
        ax.set_title(f"{caption}\nmean silhouette = {scores.mean():.2f}",
                     fontsize=10.5, color=NAVY, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[1][col]
        y = 0
        for k in range(3):
            vals = np.sort(scores[labels == k])
            ax.barh(np.arange(y, y + len(vals)), vals, height=1.0, color=CLUSTER_COLOURS[k])
            y += len(vals) + 6
        ax.axvline(scores.mean(), color=WARN, linestyle="--", linewidth=1.6)
        ax.text(scores.mean() + 0.02, y * 0.5, "mean", fontsize=8.5, color=WARN, rotation=90, va="center")
        ax.set_xlim(-0.35, 1)
        ax.set_yticks([])
        ax.set_xlabel("Silhouette score per point", fontsize=9, color=INK)
        ax.grid(axis="x", color=LINE, linewidth=0.7)
        ax.set_axisbelow(True)

    for row in axes:
        for ax in row:
            for spine in ax.spines.values():
                spine.set_edgecolor(LINE)
    fig.suptitle("Good Clustering Versus Bad, Judged Without Any Labels",
                 fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("silhouette_picks_k")
def silhouette_picks_k() -> plt.Figure:
    """Mean silhouette across candidate K values -- an independent check on the elbow."""
    points, _ = _blobs(0.55)
    ks = np.arange(2, 9)
    means = []
    for k in ks:
        rng = np.random.default_rng(0)
        cent = points[rng.choice(len(points), k, replace=False)]
        for _ in range(40):
            assign = np.linalg.norm(points[:, None] - cent[None], axis=2).argmin(axis=1)
            cent = np.array([points[assign == j].mean(axis=0) if np.any(assign == j) else cent[j] for j in range(k)])
        means.append(_silhouette_values(points, assign).mean())

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    best = int(ks[int(np.argmax(means))])
    colours = [NAVY if k == best else MIST for k in ks]
    ax.bar(ks, means, color=colours, edgecolor=BLUE, width=0.62)
    for k, m in zip(ks, means):
        ax.text(k, m + 0.012, f"{m:.2f}", ha="center", fontsize=8.5, color=INK,
                fontweight="bold" if k == best else "normal")
    ax.set_xlabel("Number of Clusters (K)", fontsize=10, color=INK)
    ax.set_ylabel("Mean Silhouette Score", fontsize=10, color=INK)
    ax.set_title(f"Silhouette Agrees With the Elbow: K = {best}", fontsize=12, color=NAVY, fontweight="bold")
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


@figure("ari_vs_silhouette")
def ari_vs_silhouette() -> plt.Figure:
    """When each metric is usable, and what ARI compares."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    points, truth = _blobs(0.55)
    found = truth.copy()
    rng = np.random.default_rng(2)
    flip = rng.choice(len(found), 14, replace=False)
    found[flip] = (found[flip] + 1) % 3

    for ax, labels, title in (
        (ax1, truth, "True labels (ground truth)"),
        (ax2, found, "Clusters the algorithm found"),
    ):
        for k in range(3):
            sel = labels == k
            ax.scatter(points[sel, 0], points[sel, 1], s=24, color=CLUSTER_COLOURS[k], alpha=0.8)
        ax.set_title(title, fontsize=10.5, color=NAVY, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    ax2.scatter(points[flip, 0], points[flip, 1], s=110, facecolor="none",
                edgecolor=WARN, linewidth=1.5, zorder=6)
    ax2.text(3.1, 0.05, "circled = disagreements ARI penalises", ha="center", fontsize=8.5, color=WARN)

    fig.suptitle("ARI Compares Two Labellings -- so it needs the truth",
                 fontsize=12, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("clustering_metric_choice")
def clustering_metric_choice() -> plt.Figure:
    """Which metric applies in which situation."""
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("Do You Have True Labels?", fontsize=12, color=NAVY, fontweight="bold")

    from matplotlib.patches import FancyBboxPatch

    def box(x, y, w, h, text, face, edge, tcolour=INK, size=9, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor=face, edgecolor=edge, linewidth=1.6, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=tcolour,
                fontweight="bold" if bold else "normal", zorder=4)

    box(5, 3.3, 3.0, 0.65, "Clustering result", NAVY, NAVY, PAPER, 9.5, True)
    box(2.6, 1.7, 3.6, 1.0,
        "NO labels\n(the real world)\n-> Silhouette Score, Elbow", "#D6EAF8", BLUE)
    box(7.4, 1.7, 3.6, 1.0,
        "Labels exist\n(teaching / validation)\n-> ARI, plus Silhouette", "#ABEBC6", GOOD)
    ax.annotate("", xy=(2.6, 2.25), xytext=(4.4, 3.0), arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.5))
    ax.annotate("", xy=(7.4, 2.25), xytext=(5.6, 3.0), arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.5))
    ax.text(5, 0.5, "ARI is never available during genuine unsupervised training",
            ha="center", fontsize=8.5, color=GREY, style="italic")
    fig.tight_layout()
    return fig
