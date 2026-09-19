"""Figures for Topic 03 -- Clustering & K-Means."""

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

CLUSTER_COLOURS = (NAVY, "#E07A3F", GOOD)


def _three_blobs(seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centres = np.array([[1.6, 4.2], [4.6, 4.6], [3.2, 1.5]])
    points, labels = [], []
    for i, c in enumerate(centres):
        pts = rng.normal(c, 0.55, size=(38, 2))
        points.append(pts)
        labels.append(np.full(38, i))
    return np.vstack(points), np.concatenate(labels)


@figure("supervised_vs_unsupervised")
def supervised_vs_unsupervised() -> plt.Figure:
    """Labelled training versus discovering structure with no target."""
    points, labels = _three_blobs()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.2))

    for i in range(3):
        sel = labels == i
        ax1.scatter(points[sel, 0], points[sel, 1], s=26, color=CLUSTER_COLOURS[i],
                    marker=["o", "s", "^"][i], label=f"Known class {i + 1}")
    ax1.legend(fontsize=8, frameon=False, loc="lower right")
    ax1.set_title("Supervised: labels are given", fontsize=10.5, color=NAVY, fontweight="bold")

    ax2.scatter(points[:, 0], points[:, 1], s=26, color=GREY)
    ax2.set_title("Unsupervised: no target variable at all", fontsize=10.5, color=NAVY, fontweight="bold")
    ax2.text(3.1, 0.15, "the model must find the groups itself",
             ha="center", fontsize=8.5, color=GREY, style="italic")

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Feature 1", fontsize=9, color=INK)
        ax.set_ylabel("Feature 2", fontsize=9, color=INK)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)
    fig.tight_layout()
    return fig


@figure("intra_vs_inter_cluster")
def intra_vs_inter_cluster() -> plt.Figure:
    """High similarity inside a cluster, low similarity between clusters."""
    points, labels = _three_blobs()
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    names = ["Cluster A", "Cluster B", "Cluster C"]
    for i in range(3):
        sel = labels == i
        ax.scatter(points[sel, 0], points[sel, 1], s=30, color=CLUSTER_COLOURS[i], alpha=0.75)
        cx, cy = points[sel].mean(axis=0)
        ax.text(cx, cy + 1.35, names[i], ha="center", fontsize=9.5, color=CLUSTER_COLOURS[i], fontweight="bold")
        circle = plt.Circle((cx, cy), 1.15, fill=False, edgecolor=CLUSTER_COLOURS[i], linestyle="--", linewidth=1.3)
        ax.add_patch(circle)

    a = points[labels == 0].mean(axis=0)
    b = points[labels == 1].mean(axis=0)
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.6))
    ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.22, "inter-cluster distance: LARGE",
            ha="center", fontsize=8.5, color=INK,
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5))
    ax.annotate("intra-cluster spread: SMALL", xy=(a[0] + 0.5, a[1] - 0.5), xytext=(a[0] - 1.1, a[1] - 2.1),
                fontsize=8.5, color=NAVY, arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1))

    ax.set_xlabel("Feature 1", fontsize=10, color=INK)
    ax.set_ylabel("Feature 2", fontsize=10, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Clustering Goal: Tight Inside, Far Apart Outside", fontsize=12, color=NAVY, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    fig.tight_layout()
    return fig


@figure("distance_metrics")
def distance_metrics() -> plt.Figure:
    """Euclidean, Manhattan and cosine compared on the same pair of points."""
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7))
    p, q = np.array([1.0, 1.0]), np.array([4.0, 3.0])

    ax = axes[0]
    ax.plot([p[0], q[0]], [p[1], q[1]], color=NAVY, linewidth=2.4)
    ax.set_title("Euclidean\nstraight-line distance", fontsize=10, color=NAVY, fontweight="bold")
    ax.text(2.4, 2.35, "sqrt(SUM (xi-yi)^2)\n= 3.61", fontsize=8.5, color=INK)

    ax = axes[1]
    ax.plot([p[0], q[0]], [p[1], p[1]], color="#E07A3F", linewidth=2.4)
    ax.plot([q[0], q[0]], [p[1], q[1]], color="#E07A3F", linewidth=2.4)
    ax.set_title("Manhattan\nsum of absolute differences", fontsize=10, color=NAVY, fontweight="bold")
    ax.text(1.4, 2.4, "SUM |xi-yi|\n= 3 + 2 = 5", fontsize=8.5, color=INK)

    ax = axes[2]
    for vec, colour in ((q, NAVY), (np.array([3.6, 1.2]), GOOD)):
        ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=colour, linewidth=2.2))
    ax.set_title("Cosine similarity\nmeasures orientation", fontsize=10, color=NAVY, fontweight="bold")
    ax.text(1.1, 3.5, "x . y / (||x|| ||y||)\nangle matters, length does not", fontsize=8.5, color=INK)

    for ax in axes:
        ax.scatter(*p, s=45, color=INK, zorder=5)
        ax.scatter(*q, s=45, color=INK, zorder=5)
        ax.set_xlim(0, 5.2)
        ax.set_ylim(0, 4.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(color=LINE, linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)
    fig.tight_layout()
    return fig


@figure("kmeans_iterations")
def kmeans_iterations() -> plt.Figure:
    """Assign, update, repeat -- three snapshots of K-Means converging."""
    points, _ = _three_blobs()
    rng = np.random.default_rng(5)
    centroids = np.array([[1.0, 1.2], [2.0, 2.0], [3.0, 2.4]]) + rng.normal(0, 0.1, (3, 2))

    snapshots = []
    for step in range(6):
        dists = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        assign = dists.argmin(axis=1)
        if step in (0, 1, 5):
            snapshots.append((centroids.copy(), assign.copy(), step))
        centroids = np.array([
            points[assign == k].mean(axis=0) if np.any(assign == k) else centroids[k] for k in range(3)
        ])

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    titles = ["Iteration 1: random centroids", "Iteration 2: centroids move", "Converged: centroids stop changing"]
    for ax, (cent, assign, _), title in zip(axes, snapshots, titles):
        for k in range(3):
            sel = assign == k
            ax.scatter(points[sel, 0], points[sel, 1], s=20, color=CLUSTER_COLOURS[k], alpha=0.6)
            ax.scatter(*cent[k], s=210, color=CLUSTER_COLOURS[k], marker="X",
                       edgecolor=PAPER, linewidth=1.8, zorder=6)
        ax.set_title(title, fontsize=9.5, color=NAVY, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle("K-Means: assign points, move centroids, repeat  (X = centroid)",
                 fontsize=11.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("elbow_method")
def elbow_method() -> plt.Figure:
    """WCSS against K, with the elbow marked at K = 3."""
    points, _ = _three_blobs()
    ks = np.arange(1, 11)
    wcss = []
    for k in ks:
        rng = np.random.default_rng(0)
        cent = points[rng.choice(len(points), k, replace=False)]
        for _ in range(30):
            assign = np.linalg.norm(points[:, None] - cent[None], axis=2).argmin(axis=1)
            cent = np.array([points[assign == j].mean(axis=0) if np.any(assign == j) else cent[j] for j in range(k)])
        wcss.append(float(((points - cent[assign]) ** 2).sum()))

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(ks, wcss, color=NAVY, linewidth=2.4, marker="o", markersize=6)
    ax.scatter([3], [wcss[2]], s=190, facecolor="none", edgecolor=WARN, linewidth=2.4, zorder=6)
    ax.annotate(
        "Elbow -> K = 3\nafter here, extra clusters\nbuy almost nothing",
        xy=(3, wcss[2]),
        xytext=(5.1, wcss[1] * 0.82),
        fontsize=9,
        color=INK,
        arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.5),
    )
    ax.set_xlabel("Number of Clusters (K)", fontsize=10, color=INK)
    ax.set_ylabel("WCSS", fontsize=10, color=INK)
    ax.set_xticks(ks)
    ax.set_title("The Elbow Method for Choosing K", fontsize=12, color=NAVY, fontweight="bold")
    ax.grid(color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


@figure("kmeans_limitations")
def kmeans_limitations() -> plt.Figure:
    """Where K-Means struggles: non-spherical shapes, unequal sizes, outliers."""
    rng = np.random.default_rng(4)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7))

    # Concentric rings.
    ax = axes[0]
    for radius, colour in ((1.0, NAVY), (2.6, "#E07A3F")):
        theta = rng.uniform(0, 2 * np.pi, 110)
        r = radius + rng.normal(0, 0.13, 110)
        ax.scatter(r * np.cos(theta), r * np.sin(theta), s=14, color=colour, alpha=0.75)
    ax.axline((0, -3), (0, 3), color=WARN, linestyle="--", linewidth=1.8)
    ax.set_title("Non-spherical clusters", fontsize=10, color=NAVY, fontweight="bold")
    ax.text(0, -3.6, "K-Means cuts straight through", ha="center", fontsize=8, color=WARN)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-4.1, 3.4)

    # Unequal sizes.
    ax = axes[1]
    big = rng.normal([0, 0], 1.25, size=(180, 2))
    small = rng.normal([3.6, 2.2], 0.30, size=(22, 2))
    ax.scatter(big[:, 0], big[:, 1], s=14, color=NAVY, alpha=0.6)
    ax.scatter(small[:, 0], small[:, 1], s=14, color="#E07A3F", alpha=0.85)
    ax.set_title("Clusters of very different size", fontsize=10, color=NAVY, fontweight="bold")
    ax.text(1.6, -3.4, "the big blob tends to swallow the small one",
            ha="center", fontsize=8, color=WARN)
    ax.set_xlim(-4, 5.2)
    ax.set_ylim(-3.8, 3.8)

    # Outliers.
    ax = axes[2]
    core = rng.normal([0, 0], 0.75, size=(90, 2))
    ax.scatter(core[:, 0], core[:, 1], s=16, color=NAVY, alpha=0.7)
    ax.scatter([4.4], [4.0], s=90, color=WARN, marker="X")
    ax.annotate("one outlier drags\nthe centroid across",
                xy=(4.4, 4.0), xytext=(0.2, 3.4), fontsize=8, color=WARN,
                arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.3))
    ax.set_title("Outliers and noise", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_xlim(-3, 5.6)
    ax.set_ylim(-3, 5.0)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)
    fig.tight_layout()
    return fig


@figure("beyond_kmeans")
def beyond_kmeans() -> plt.Figure:
    """Hierarchical dendrogram and DBSCAN density groups."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 4.0))

    # Dendrogram, drawn by hand so no scipy dependency creeps in.
    leaves = {"A": 0.5, "B": 1.5, "C": 3.0, "D": 4.0, "E": 5.0}
    for name, x in leaves.items():
        ax1.text(x, -0.18, name, ha="center", fontsize=9, color=INK)

    def merge(x1: float, x2: float, height: float) -> float:
        ax1.plot([x1, x1, x2, x2], [0, height, height, 0], color=NAVY, linewidth=1.8)
        return (x1 + x2) / 2

    ab = merge(0.5, 1.5, 0.9)
    cd = merge(3.0, 4.0, 0.7)
    cde = merge(cd, 5.0, 1.5)
    merge(ab, cde, 2.4)
    ax1.axhline(1.9, color=WARN, linestyle="--", linewidth=1.5)
    ax1.text(5.2, 1.98, "cut here -> 2 clusters", fontsize=8.5, color=WARN, ha="right")
    ax1.set_ylim(-0.4, 2.9)
    ax1.set_xlim(0, 5.8)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Hierarchical Clustering\ncut the dendrogram at any level", fontsize=10.5,
                  color=NAVY, fontweight="bold")

    # DBSCAN.
    rng = np.random.default_rng(9)
    theta = rng.uniform(0, np.pi, 70)
    moon1 = np.c_[np.cos(theta) * 2, np.sin(theta) * 1.3] + rng.normal(0, 0.11, (70, 2))
    moon2 = np.c_[np.cos(theta) * 2 + 2, -np.sin(theta) * 1.3 + 0.4] + rng.normal(0, 0.11, (70, 2))
    noise = rng.uniform([-2.6, -1.6], [4.6, 2.0], size=(11, 2))
    ax2.scatter(moon1[:, 0], moon1[:, 1], s=18, color=NAVY)
    ax2.scatter(moon2[:, 0], moon2[:, 1], s=18, color="#E07A3F")
    ax2.scatter(noise[:, 0], noise[:, 1], s=40, color=GREY, marker="x")
    ax2.text(1.0, -1.9, "grey x = automatically labelled noise", ha="center", fontsize=8.5, color=GREY)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("DBSCAN (density-based)\nirregular shapes, no K needed", fontsize=10.5,
                  color=NAVY, fontweight="bold")

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)
    fig.tight_layout()
    return fig
