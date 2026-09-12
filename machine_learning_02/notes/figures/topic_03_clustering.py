"""Figures for note 03 -- Clustering & K-Means.

Toy-example numbers are computed live from the notebook's 8-point dataset;
the Iris elbow, silhouette and cluster numbers are reproduced by re-running
the practical notebook's exact pipeline (StandardScaler, n_init=10,
random_state=42).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

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
    node_box,
    panel_title,
)

# -- The notebook's 8-point toy dataset ----------------------------------
TOY = np.array([
    [1, 1], [1.5, 2], [2, 1], [1, 2],        # Group A (bottom-left)
    [8, 8], [8.5, 7], [9, 8], [8, 9],        # Group B (top-right)
], dtype=float)

CLUSTER_COLOURS = [BLUE, WARN, GOOD]


def run_kmeans(points: np.ndarray, centroids: np.ndarray, steps: int = 6):
    """Plain-numpy K-Means, yielding (centroids, labels, wcss) per iteration."""
    centroids = centroids.copy()
    history = []
    for _ in range(steps):
        distances = np.linalg.norm(points[:, None] - centroids[None], axis=2)
        labels = distances.argmin(axis=1)
        wcss = sum(((points[labels == k] - centroids[k]) ** 2).sum() for k in range(len(centroids)))
        history.append((centroids.copy(), labels.copy(), wcss))
        updated = np.array([
            points[labels == k].mean(axis=0) if (labels == k).any() else centroids[k]
            for k in range(len(centroids))
        ])
        if np.allclose(updated, centroids):
            break
        centroids = updated
    return history


@figure("clustering_concept")
def clustering_concept() -> plt.Figure:
    """Unlabelled points in, three discovered groups out."""
    rng = np.random.default_rng(7)
    centres = [(2.0, 2.2), (6.4, 2.8), (4.2, 6.4)]
    clouds = [rng.normal(c, 0.62, size=(26, 2)) for c in centres]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    ax = axes[0]
    allpts = np.vstack(clouds)
    ax.scatter(allpts[:, 0], allpts[:, 1], color=GREY, s=34, edgecolor=PAPER, linewidth=0.6)
    ax.set_title("What the algorithm receives\nunlabelled points -- no target column", fontsize=10.5,
                 fontweight="bold", color=NAVY)

    ax = axes[1]
    for cloud, colour, name in zip(clouds, CLUSTER_COLOURS, ["Cluster A", "Cluster B", "Cluster C"]):
        ax.scatter(cloud[:, 0], cloud[:, 1], color=colour, s=34, edgecolor=PAPER, linewidth=0.6, label=name)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("What it discovers\ngroups of mutually similar points", fontsize=10.5,
                 fontweight="bold", color=NAVY)

    for ax in axes:
        ax.set_xlim(0, 8.6)
        ax.set_ylim(0, 8.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("Feature 1", fontsize=9, color=INK)
        ax.set_ylabel("Feature 2", fontsize=9, color=INK)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle(
        "Clustering: high similarity WITHIN a group, low similarity BETWEEN groups",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig


@figure("kmeans_iterations")
def kmeans_iterations() -> plt.Figure:
    """The assign/update loop converging, from deliberately poor starting centroids."""
    start = np.array([[1.0, 1.0], [2.0, 1.0]])  # both centroids in the bottom-left group
    history = run_kmeans(TOY, start)

    fig, axes = plt.subplots(1, len(history), figsize=(4.0 * len(history), 4.3))
    if len(history) == 1:
        axes = [axes]

    for step, (ax, (centroids, labels, wcss)) in enumerate(zip(axes, history)):
        for k, colour in enumerate(CLUSTER_COLOURS[: len(centroids)]):
            members = TOY[labels == k]
            if len(members):
                ax.scatter(members[:, 0], members[:, 1], color=colour, s=95,
                           edgecolor=PAPER, linewidth=1.2, zorder=3)
            ax.scatter(*centroids[k], color=colour, marker="X", s=290,
                       edgecolor=INK, linewidth=1.4, zorder=5)

        ax.set_xlim(-0.6, 10.4)
        ax.set_ylim(-0.6, 10.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2, color=LINE)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

        converged = step == len(history) - 1
        label = "converged -- centroids stopped moving" if converged else "assign, then recompute means"
        ax.set_title(
            f"Iteration {step}    WCSS = {wcss:.2f}\n{label}",
            fontsize=10,
            fontweight="bold",
            color=GOOD if converged else NAVY,
        )

    fig.suptitle(
        'K-Means converging from a deliberately bad start (both centroids in one group)',
        fontsize=12.5,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig


@figure("toy_kmeans_result")
def toy_kmeans_result() -> plt.Figure:
    """The notebook's toy result: two clusters, centroids marked."""
    history = run_kmeans(TOY, np.array([[1.0, 1.0], [8.0, 8.0]]))
    centroids, labels, wcss = history[-1]

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    for k, colour in enumerate(CLUSTER_COLOURS[:2]):
        members = TOY[labels == k]
        ax.scatter(members[:, 0], members[:, 1], color=colour, s=120,
                   edgecolor=PAPER, linewidth=1.3, zorder=3, label=f"Cluster {k}")
        cx, cy = centroids[k]
        ax.scatter(cx, cy, color=colour, marker="X", s=320, edgecolor=INK, linewidth=1.5, zorder=5)
        ax.annotate(
            f"centroid\n({cx:.3f}, {cy:.1f})",
            xy=(cx, cy),
            xytext=(cx + 1.5, cy - 1.6),
            fontsize=8.5,
            color=INK,
            ha="center",
            arrowprops=dict(arrowstyle="->", color=GREY, linewidth=1.1),
        )
        for point in members:
            ax.plot([point[0], cx], [point[1], cy], color=colour, linewidth=0.8, alpha=0.45, zorder=2)

    ax.set_xlim(-1, 11.5)
    ax.set_ylim(-1, 11.5)
    ax.set_xlabel("Feature 1", fontsize=10, color=INK)
    ax.set_ylabel("Feature 2", fontsize=10, color=INK)
    ax.grid(alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(
        ax,
        "Toy K-Means: 2 Clusters Found",
        f"Thin lines are the distances WCSS adds up  (WCSS = {wcss:.3f})",
    )
    fig.tight_layout()
    return fig


# -- Iris elbow, reproduced from the notebook pipeline (scaled, n_init=10, rs=42) --
ELBOW_K = list(range(1, 9))
ELBOW_WCSS = [900.0, 312.1, 200.1, 164.0, 139.5, 120.5, 104.9, 93.0]


@figure("elbow_method")
def elbow_method() -> plt.Figure:
    """WCSS against K, with the elbow and the size of each drop made explicit."""
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    ax.plot(ELBOW_K, ELBOW_WCSS, marker="o", color=BLUE, linewidth=2.2, markersize=8, zorder=3)
    ax.axvline(3, color=WARN, linestyle="--", linewidth=1.5, zorder=2)
    ax.annotate(
        "Elbow -> K = 3\nthe bend where extra clusters\nstop paying for themselves",
        xy=(3, 200.1),
        xytext=(4.15, 505),
        fontsize=9,
        color=WARN,
        arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.3),
    )

    # Annotate how much each additional cluster actually buys.
    for i in range(1, len(ELBOW_K)):
        drop = ELBOW_WCSS[i - 1] - ELBOW_WCSS[i]
        mid_x = (ELBOW_K[i - 1] + ELBOW_K[i]) / 2
        mid_y = (ELBOW_WCSS[i - 1] + ELBOW_WCSS[i]) / 2
        ax.annotate(
            f"-{drop:.0f}",
            xy=(mid_x, mid_y),
            xytext=(6, 10),
            textcoords="offset points",
            fontsize=8,
            color=GREY,
        )

    ax.set_xlabel("Number of Clusters (K)", fontsize=10, color=INK)
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=10, color=INK)
    ax.set_xticks(ELBOW_K)
    ax.set_ylim(0, 1000)
    ax.grid(alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(
        ax,
        "Elbow Method: WCSS vs. Number of Clusters",
        "Iris (scaled) -- the drops shrink sharply after K = 3",
    )
    fig.tight_layout()
    return fig


@figure("kmeans_limitations")
def kmeans_limitations() -> plt.Figure:
    """Three shapes K-Means handles badly, and the one it handles well."""
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))

    # 1. Spherical and equally sized -- K-Means is happy.
    ax = axes[0]
    for centre, colour in zip([(2, 2), (6, 6)], CLUSTER_COLOURS):
        cloud = rng.normal(centre, 0.72, size=(45, 2))
        ax.scatter(cloud[:, 0], cloud[:, 1], color=colour, s=26, edgecolor=PAPER, linewidth=0.5)
    ax.set_title("Compact, spherical, similar size\nK-Means works well", fontsize=10,
                 fontweight="bold", color=GOOD)

    # 2. Two crescents -- non-spherical, K-Means slices them wrongly.
    ax = axes[1]
    t = np.linspace(0, np.pi, 70)
    moon_a = np.c_[np.cos(t) * 3 + 4, np.sin(t) * 3 + 2.4] + rng.normal(0, 0.22, (70, 2))
    moon_b = np.c_[np.cos(t) * 3 + 6, -np.sin(t) * 3 + 5.2] + rng.normal(0, 0.22, (70, 2))
    both = np.vstack([moon_a, moon_b])
    split = both[:, 0] < both[:, 0].mean()   # the vertical-ish cut K-Means would make
    ax.scatter(both[split][:, 0], both[split][:, 1], color=BLUE, s=26, edgecolor=PAPER, linewidth=0.5)
    ax.scatter(both[~split][:, 0], both[~split][:, 1], color=WARN, s=26, edgecolor=PAPER, linewidth=0.5)
    ax.axvline(both[:, 0].mean(), color=INK, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_title("Crescent shapes\nK-Means cuts straight through", fontsize=10,
                 fontweight="bold", color=WARN)

    # 3. An outlier dragging a centroid away.
    ax = axes[2]
    cloud = rng.normal((3, 3), 0.62, size=(20, 2))
    outliers = np.array([[9.4, 9.2], [9.8, 8.7], [9.1, 9.7], [9.9, 9.9]])
    ax.scatter(cloud[:, 0], cloud[:, 1], color=BLUE, s=30, edgecolor=PAPER, linewidth=0.5)
    ax.scatter(outliers[:, 0], outliers[:, 1], color=WARN, s=85, edgecolor=INK, linewidth=0.9)
    honest = cloud.mean(axis=0)
    pulled = np.vstack([cloud, outliers]).mean(axis=0)
    ax.scatter(*honest, color=GOOD, marker="X", s=260, edgecolor=INK, linewidth=1.3, zorder=6)
    ax.scatter(*pulled, color=WARN, marker="X", s=260, edgecolor=INK, linewidth=1.3, zorder=6)
    ax.annotate(
        "",
        xy=pulled,
        xytext=honest,
        arrowprops=dict(arrowstyle="->", color=WARN, linewidth=2.2),
    )
    ax.text(1.1, 7.9, "where the centroid\nSHOULD sit", fontsize=8.5, color=GOOD)
    ax.text(4.6, 4.9, "where it actually\nlands -- dragged by\n4 outliers", fontsize=8.5, color=WARN)
    ax.set_title("Outliers\nthe mean is not robust", fontsize=10, fontweight="bold", color=WARN)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2, color=LINE)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle(
        "K-Means assumes compact, spherical, similarly sized clusters -- and dislikes outliers",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig


@figure("distance_metrics")
def distance_metrics() -> plt.Figure:
    """Euclidean vs Manhattan vs Cosine, on the same pair of points."""
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    p, q = np.array([1.0, 1.0]), np.array([4.0, 3.0])

    # Euclidean -- straight line.
    ax = axes[0]
    ax.plot([p[0], q[0]], [p[1], q[1]], color=BLUE, linewidth=2.6, zorder=2)
    euclid = np.linalg.norm(p - q)
    ax.set_title(f"Euclidean\nstraight-line distance = {euclid:.2f}", fontsize=10,
                 fontweight="bold", color=NAVY)

    # Manhattan -- along the grid.
    ax = axes[1]
    ax.plot([p[0], q[0]], [p[1], p[1]], color=WARN, linewidth=2.6, zorder=2)
    ax.plot([q[0], q[0]], [p[1], q[1]], color=WARN, linewidth=2.6, zorder=2)
    manhattan = np.abs(p - q).sum()
    ax.set_title(f"Manhattan\nalong the grid = {manhattan:.2f}", fontsize=10,
                 fontweight="bold", color=NAVY)

    # Cosine -- the angle between the vectors.
    ax = axes[2]
    ax.annotate("", xy=p, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GOOD, linewidth=2.2))
    ax.annotate("", xy=q, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=GOOD, linewidth=2.2))
    cosine = p @ q / (np.linalg.norm(p) * np.linalg.norm(q))
    angle = np.degrees(np.arccos(cosine))
    ax.set_title(f"Cosine similarity\nangle only = {cosine:.3f}  ({angle:.1f} deg)", fontsize=10,
                 fontweight="bold", color=NAVY)

    for ax in axes:
        ax.scatter(*p, color=INK, s=80, zorder=4)
        ax.scatter(*q, color=INK, s=80, zorder=4)
        ax.text(p[0] - 0.1, p[1] - 0.45, "x (1, 1)", fontsize=9, color=INK, ha="center")
        ax.text(q[0] + 0.15, q[1] + 0.28, "y (4, 3)", fontsize=9, color=INK, ha="center")
        ax.set_xlim(-0.4, 5.4)
        ax.set_ylim(-0.4, 4.4)
        ax.set_xticks(range(6))
        ax.set_yticks(range(5))
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, color=LINE)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle("Three ways to measure 'how similar are these two points?'",
                 fontsize=12, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return fig
