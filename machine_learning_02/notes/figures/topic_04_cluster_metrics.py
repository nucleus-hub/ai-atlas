"""Figures for note 04 -- Clustering Evaluation Metrics.

The silhouette worked example is computed live on a 4-point dataset small
enough to verify by hand. The Iris silhouette plot and the metric-vs-K
comparison are reproduced from the practical notebook's pipeline
(StandardScaler, n_init=10, random_state=42).
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
    figure,
    panel_title,
)

CLUSTER_COLOURS = [BLUE, WARN, GOOD]


def silhouette_by_hand(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """s = (b - a) / max(a, b) for every point -- singletons score 0."""
    scores = []
    for i, (point, label) in enumerate(zip(points, labels)):
        same = [np.linalg.norm(point - p) for j, p in enumerate(points) if j != i and labels[j] == label]
        if not same:                       # a lone point in its cluster
            scores.append(0.0)
            continue
        a = np.mean(same)
        b = min(
            np.mean([np.linalg.norm(point - p) for j, p in enumerate(points) if labels[j] == other])
            for other in set(labels) - {label}
        )
        scores.append((b - a) / max(a, b))
    return np.array(scores)


# -- A 4-point example, deliberately tiny enough to check with a calculator --
DEMO_POINTS = np.array([[1.0, 1.0], [2.0, 1.0], [6.0, 1.0], [7.0, 1.0]])
DEMO_GOOD = np.array([0, 0, 1, 1])
DEMO_BAD = np.array([0, 0, 0, 1])


@figure("silhouette_anatomy")
def silhouette_anatomy() -> plt.Figure:
    """What a and b actually measure, on one highlighted point."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))

    own = np.array([[1.4, 3.1], [2.1, 3.6], [1.8, 2.4], [2.6, 2.9]])
    other = np.array([[6.4, 3.2], [7.1, 3.7], [6.8, 2.5], [7.4, 3.0]])
    focus = own[0]

    for point in own[1:]:
        ax.plot([focus[0], point[0]], [focus[1], point[1]], color=BLUE, linewidth=1.6, alpha=0.8, zorder=2)
    for point in other:
        ax.plot([focus[0], point[0]], [focus[1], point[1]], color=WARN,
                linewidth=1.1, alpha=0.55, linestyle="--", zorder=2)

    ax.scatter(own[:, 0], own[:, 1], color=BLUE, s=115, edgecolor=PAPER, linewidth=1.3, zorder=4)
    ax.scatter(other[:, 0], other[:, 1], color=WARN, s=115, edgecolor=PAPER, linewidth=1.3, zorder=4)
    ax.scatter(*focus, color=NAVY, s=280, edgecolor=PAPER, linewidth=2, zorder=6, marker="o")

    ax.text(focus[0] - 0.15, focus[1] + 0.45, "the point\nbeing scored", fontsize=9,
            color=NAVY, ha="center", fontweight="bold")
    ax.text(2.25, 1.75, "a = average distance to its\nOWN cluster  (want SMALL)",
            fontsize=9.5, color=BLUE, ha="center")
    ax.text(5.1, 4.5, "b = average distance to the NEAREST\nother cluster  (want LARGE)",
            fontsize=9.5, color=WARN, ha="center")

    ax.text(4.4, 0.72, "s = (b - a) / max(a, b)", fontsize=13, color=INK,
            ha="center", fontweight="bold",
            bbox=dict(facecolor=MIST, edgecolor=NAVY, boxstyle="round,pad=0.45"))

    ax.set_xlim(0.2, 8.6)
    ax.set_ylim(0.2, 5.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Anatomy of the Silhouette Score",
                "Compare where a point sits against its own cluster versus its nearest neighbour cluster")
    fig.tight_layout()
    return fig


@figure("silhouette_worked_example")
def silhouette_worked_example() -> plt.Figure:
    """The same 4 points, clustered well and clustered badly."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for ax, labels, title in (
        (axes[0], DEMO_GOOD, "Sensible clustering"),
        (axes[1], DEMO_BAD, "Point at x=6 forced into the wrong cluster"),
    ):
        scores = silhouette_by_hand(DEMO_POINTS, labels)
        for i, ((x, y), label, score) in enumerate(zip(DEMO_POINTS, labels, scores)):
            colour = CLUSTER_COLOURS[label]
            ax.scatter(x, y, color=colour, s=230, edgecolor=PAPER, linewidth=1.5, zorder=4)
            ax.text(x, y, str(label), color=PAPER, fontsize=10, fontweight="bold",
                    ha="center", va="center", zorder=5)
            good = score > 0
            ax.annotate(
                f"s = {score:+.3f}",
                xy=(x, y),
                xytext=(0, -30 if i % 2 == 0 else -52),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
                fontweight="bold",
                color=GOOD if good else WARN,
            )

        mean = scores.mean()
        ax.text(4.0, 1.62, f"Overall Silhouette Score = {mean:.3f}", fontsize=11,
                ha="center", color=INK, fontweight="bold",
                bbox=dict(facecolor=MIST if mean > 0.5 else "#FBE9E7",
                          edgecolor=NAVY if mean > 0.5 else WARN,
                          boxstyle="round,pad=0.35"))

        ax.set_xlim(0, 8)
        ax.set_ylim(0.15, 1.85)
        ax.set_yticks([])
        ax.set_xticks(range(9))
        ax.tick_params(labelsize=8.5)
        ax.grid(axis="x", alpha=0.25, color=LINE)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10.5, fontweight="bold", color=NAVY)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle(
        "A misplaced point earns a NEGATIVE score -- the metric points straight at it",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig


@figure("silhouette_scale")
def silhouette_scale() -> plt.Figure:
    """The -1 to +1 interpretation bands from the slides."""
    fig, ax = plt.subplots(figsize=(9.2, 2.9))

    bands = [
        (-1.0, 0.0, "#FBE9E7", WARN, "-1.0 to 0\nPoints likely in\nthe WRONG cluster"),
        (0.0, 0.5, "#FEF5E7", "#B9770E", "0 to +0.5\nWeak or\noverlapping clusters"),
        (0.5, 0.7, "#F4F9F5", GREY, "0.5 - 0.7\nreasonable"),
        (0.7, 1.0, "#E4F3E9", GOOD, "+0.7 to +1.0\nStrong, well-\nseparated clusters"),
    ]
    for lo, hi, fill, edge, label in bands:
        ax.axvspan(lo, hi, color=fill, zorder=1)
        ax.axvline(hi, color=LINE, linewidth=1, zorder=2)
        ax.text((lo + hi) / 2, 0.62, label, ha="center", va="center",
                fontsize=8.6, color=edge, fontweight="bold", zorder=4)

    # The practical's actual Iris result.
    ax.annotate(
        "Iris result\n0.469",
        xy=(0.469, 0.12),
        xytext=(0.469, -0.42),
        ha="center",
        fontsize=9.5,
        color=NAVY,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=NAVY, linewidth=1.8),
    )

    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.55, 1)
    ax.set_yticks([])
    ax.set_xticks([-1, -0.5, 0, 0.5, 0.7, 1])
    ax.tick_params(labelsize=9)
    ax.set_xlabel("Silhouette Score", fontsize=10, color=INK)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Reading the Silhouette Score", "One scale, four verdicts")
    fig.tight_layout()
    return fig


# -- Per-point Iris silhouettes, reproduced from the notebook pipeline --
IRIS_CLUSTER_STATS = {
    0: dict(n=39, mean=0.3556, lo=0.0098, hi=0.5445),
    1: dict(n=50, mean=0.6842, lo=0.1226, hi=0.7709),
    2: dict(n=61, mean=0.3641, lo=0.0636, hi=0.5681),
}
IRIS_OVERALL_SILHOUETTE = 0.4686


@figure("iris_silhouette_plot")
def iris_silhouette_plot() -> plt.Figure:
    """Per-cluster silhouette summary for the practical's Iris clustering."""
    fig, ax = plt.subplots(figsize=(8.2, 4.4))

    names = {0: "Cluster 0\n(mostly virginica)", 1: "Cluster 1\n(setosa)", 2: "Cluster 2\n(versicolor +)"}
    ypos = np.arange(len(IRIS_CLUSTER_STATS))

    for i, (cluster, stats) in enumerate(sorted(IRIS_CLUSTER_STATS.items())):
        colour = CLUSTER_COLOURS[cluster]
        ax.barh(i, stats["mean"], color=colour, alpha=0.85, height=0.56,
                edgecolor=colour, linewidth=1.2, zorder=3)
        ax.plot([stats["lo"], stats["hi"]], [i, i], color=INK, linewidth=1.6, zorder=4)
        for edge in (stats["lo"], stats["hi"]):
            ax.plot([edge, edge], [i - 0.11, i + 0.11], color=INK, linewidth=1.6, zorder=4)
        ax.text(stats["mean"] + 0.018, i + 0.19, f"mean {stats['mean']:.3f}",
                va="center", fontsize=9, color=INK, fontweight="bold")
        ax.text(0.015, i - 0.2, f"n = {stats['n']}", va="center", fontsize=8.5, color=PAPER)

    ax.axvline(IRIS_OVERALL_SILHOUETTE, color=NAVY, linestyle="--", linewidth=1.8, zorder=5)
    ax.text(IRIS_OVERALL_SILHOUETTE + 0.012, 2.42, f"overall = {IRIS_OVERALL_SILHOUETTE:.3f}",
            fontsize=9.5, color=NAVY, fontweight="bold")

    ax.set_yticks(ypos)
    ax.set_yticklabels([names[c] for c in sorted(IRIS_CLUSTER_STATS)], fontsize=9)
    ax.set_xlim(0, 0.85)
    ax.set_xlabel("Silhouette Score  (bar = cluster mean, line = min to max)", fontsize=9.5, color=INK)
    ax.grid(axis="x", alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Per-Cluster Silhouette Scores (Iris)",
                "Setosa's cluster scores twice as well as the two overlapping ones")
    fig.tight_layout()
    return fig


# -- Both metrics across K, reproduced from the notebook pipeline --
K_VALUES = [2, 3, 4, 5, 6]
K_SILHOUETTE = [0.5932, 0.4686, 0.4276, 0.3292, 0.3525]
K_ARI = [0.5681, 0.7163, 0.6072, 0.4681, 0.4616]


@figure("silhouette_vs_ari")
def silhouette_vs_ari() -> plt.Figure:
    """The two metrics disagree about K -- and ARI is the one that knows the truth."""
    fig, ax = plt.subplots(figsize=(7.8, 4.6))

    ax.plot(K_VALUES, K_SILHOUETTE, marker="o", color=BLUE, linewidth=2.2, markersize=8,
            label="Silhouette Score (no labels needed)")
    ax.plot(K_VALUES, K_ARI, marker="s", color=WARN, linewidth=2.2, markersize=8,
            label="Adjusted Rand Index (uses true species)")

    best_sil = K_VALUES[int(np.argmax(K_SILHOUETTE))]
    best_ari = K_VALUES[int(np.argmax(K_ARI))]
    ax.scatter([best_sil], [max(K_SILHOUETTE)], s=260, facecolor="none",
               edgecolor=BLUE, linewidth=2.2, zorder=6)
    ax.scatter([best_ari], [max(K_ARI)], s=260, facecolor="none",
               edgecolor=WARN, linewidth=2.2, zorder=6)

    ax.annotate(
        f"Silhouette prefers K = {best_sil}\n(geometry: 2 blobs)",
        xy=(best_sil, max(K_SILHOUETTE)),
        xytext=(2.35, 0.30),
        fontsize=9,
        color=BLUE,
        arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.3),
    )
    ax.annotate(
        f"ARI prefers K = {best_ari}\n(truth: 3 species)",
        xy=(best_ari, max(K_ARI)),
        xytext=(3.75, 0.80),
        fontsize=9,
        color=WARN,
        arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.3),
    )

    ax.set_xlabel("Number of Clusters (K)", fontsize=10, color=INK)
    ax.set_ylabel("Score", fontsize=10, color=INK)
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0.2, 0.9)
    ax.grid(alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Silhouette vs. ARI across K (Iris)",
                "Two reasonable metrics, two different answers")
    fig.tight_layout()
    return fig


@figure("ari_chance_correction")
def ari_chance_correction() -> plt.Figure:
    """Why 'Adjusted' matters: raw Rand flatters random labellings."""
    fig, ax = plt.subplots(figsize=(8.4, 4.0))

    categories = ["Random labelling\n(150 flowers, 3 clusters)", "The practical's\nK-Means result"]
    rand_scores = [0.5565, 0.8737]
    ari_scores = [-0.0010, 0.7163]

    x = np.arange(len(categories))
    width = 0.34

    bars1 = ax.bar(x - width / 2, rand_scores, width, label="Rand Index (unadjusted)",
                   color=MIST, edgecolor=GREY, linewidth=1.2)
    bars2 = ax.bar(x + width / 2, ari_scores, width, label="Adjusted Rand Index",
                   color=BLUE, edgecolor=NAVY, linewidth=1.2)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    height + (0.03 if height >= 0 else -0.07),
                    f"{height:.3f}", ha="center", fontsize=9.5,
                    fontweight="bold", color=INK)

    ax.axhline(0, color=INK, linewidth=1.1)
    ax.annotate(
        "Random guessing scores 0.557 --\nlooks respectable, means nothing",
        xy=(-width / 2, 0.5565),
        xytext=(0.16, 0.60),
        fontsize=9,
        color=WARN,
        arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.3),
    )
    ax.annotate(
        "Adjusted: random\ncorrectly scores ~0",
        xy=(width / 2, 0),
        xytext=(0.60, -0.30),
        fontsize=9,
        color=GOOD,
        arrowprops=dict(arrowstyle="->", color=GOOD, linewidth=1.3),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_ylabel("Score", fontsize=10, color=INK)
    ax.set_ylim(-0.45, 1.35)
    ax.grid(axis="y", alpha=0.25, color=LINE)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, 'Why the "Adjusted" in Adjusted Rand Index',
                "Chance correction is what makes 0 mean 'no better than random'")
    fig.tight_layout()
    return fig
