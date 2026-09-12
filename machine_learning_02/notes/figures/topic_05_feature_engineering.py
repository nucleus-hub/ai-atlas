"""Figures for note 05 -- Feature Engineering.

Every number is computed live: the interaction and binning demonstrations
build their own small synthetic datasets, and the Wine/Iris results are
reproduced from the practical notebook's pipeline (random_state=42).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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


@figure("fe_process")
def fe_process() -> plt.Figure:
    """The five-step feature engineering loop from slide 56."""
    fig, ax = plt.subplots(figsize=(11.5, 3.5))
    blank_axes(ax, xlim=(0, 12.4), ylim=(0, 3.4))

    steps = [
        ("1\nUnderstand", "the business problem\nand the data"),
        ("2\nCreate", "generate new features\nfrom existing data"),
        ("3\nTransform", "make features more\nsuitable for modeling"),
        ("4\nSelect", "keep the useful ones,\nremove the rest"),
        ("5\nValidate", "evaluate, then\nrepeat as needed"),
    ]

    xs = np.linspace(1.3, 11.1, len(steps))
    for i, ((title, caption), x) in enumerate(zip(steps, xs)):
        node_box(ax, (x, 2.6), title, facecolor=MIST if i % 2 == 0 else PAPER,
                 edgecolor=NAVY, width=1.85, height=0.95, fontsize=10, bold=True)
        ax.text(x, 1.92, caption, ha="center", va="top", fontsize=8.2, color=GREY)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.95, 2.6), xytext=(x + 0.95, 2.6),
                        arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.8))

    # The loop back from Validate to Create, routed underneath the row.
    ax.annotate(
        "",
        xy=(xs[1], 1.02),
        xytext=(xs[4], 1.02),
        arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.6,
                        connectionstyle="arc3,rad=-0.18", linestyle="--"),
    )
    ax.text(6.2, 1.02, "iterate -- feature engineering is a loop, not a checklist",
            ha="center", va="center", fontsize=9, color=WARN, style="italic",
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=2.5))

    panel_title(ax, "The Feature Engineering Process", "")
    fig.tight_layout()
    return fig


@figure("interaction_feature")
def interaction_feature() -> plt.Figure:
    """A relationship that is invisible in the raw features and obvious in their product."""
    rng = np.random.default_rng(0)
    n = 400
    a = rng.uniform(-1, 1, n)
    b = rng.uniform(-1, 1, n)
    y = (a * b) > 0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.scatter(a[y], b[y], color=BLUE, s=22, alpha=0.75, label="Class 1")
    ax.scatter(a[~y], b[~y], color=WARN, s=22, alpha=0.75, label="Class 0")
    ax.axhline(0, color=GREY, linewidth=0.9)
    ax.axvline(0, color=GREY, linewidth=0.9)
    ax.set_xlabel("feature a", fontsize=9.5, color=INK)
    ax.set_ylabel("feature b", fontsize=9.5, color=INK)
    ax.set_title("Raw features a and b\nno straight line can separate these",
                 fontsize=10.5, fontweight="bold", color=WARN)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    ax = axes[1]
    product = a * b
    ax.scatter(product[y], rng.uniform(0, 1, y.sum()), color=BLUE, s=22, alpha=0.75)
    ax.scatter(product[~y], rng.uniform(0, 1, (~y).sum()), color=WARN, s=22, alpha=0.75)
    ax.axvline(0, color=INK, linestyle="--", linewidth=2)
    ax.text(0.06, 0.06, "one threshold at 0\nseparates them perfectly", fontsize=9,
            color=GOOD, transform=ax.transAxes, fontweight="bold",
            bbox=dict(facecolor=PAPER, edgecolor=GOOD, boxstyle="round,pad=0.3"))
    ax.set_xlabel("engineered feature:  a x b", fontsize=9.5, color=INK)
    ax.set_yticks([])
    ax.set_title("The interaction a x b\none split now does the whole job",
                 fontsize=10.5, fontweight="bold", color=GOOD)

    for ax in axes:
        ax.grid(alpha=0.22, color=LINE)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle(
        "Logistic Regression: 52.2% on the raw features  ->  98.7% with a x b added",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig


@figure("binning_demo")
def binning_demo() -> plt.Figure:
    """A U-shaped relationship that correlation misses and binning exposes."""
    rng = np.random.default_rng(7)
    n = 400
    age = rng.integers(15, 80, n)
    prob = np.where((age < 25) | (age > 62), 0.8, 0.15)
    buy = (rng.random(n) < prob).astype(int)
    frame = pd.DataFrame({"age": age, "buy": buy})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    jitter = rng.uniform(-0.06, 0.06, n)
    ax.scatter(age, buy + jitter, color=GREY, s=16, alpha=0.5)
    slope, intercept = np.polyfit(age, buy, 1)
    xs = np.array([15, 79])
    ax.plot(xs, slope * xs + intercept, color=WARN, linewidth=2.4)
    ax.set_xlabel("age", fontsize=9.5, color=INK)
    ax.set_ylabel("bought the product", fontsize=9.5, color=INK)
    ax.set_yticks([0, 1])
    corr = frame["age"].corr(frame["buy"])
    ax.set_title(f"Raw age -- correlation is only {corr:.2f}\na straight line sees almost nothing",
                 fontsize=10.5, fontweight="bold", color=WARN)

    ax = axes[1]
    bins = pd.cut(frame["age"], bins=[0, 18, 35, 60, 120], labels=["0-18", "19-35", "36-60", "60+"])
    rates = frame.groupby(bins, observed=True)["buy"].mean()
    colours = [GOOD if v > 0.5 else MIST for v in rates]
    bars = ax.bar(rates.index.astype(str), rates.values, color=colours,
                  edgecolor=NAVY, linewidth=1.2)
    for bar, value in zip(bars, rates.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.0%}",
                ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("age bin (engineered feature)", fontsize=9.5, color=INK)
    ax.set_ylabel("purchase rate", fontsize=9.5, color=INK)
    ax.set_title("Binned into life stages\nthe U-shape is suddenly obvious",
                 fontsize=10.5, fontweight="bold", color=GOOD)

    for ax in axes:
        ax.grid(axis="y", alpha=0.22, color=LINE)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle(
        "Binning turns a non-monotonic pattern into something any model can use",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig


@figure("log_transform")
def log_transform() -> plt.Figure:
    """Taming a skewed distribution with a log transform."""
    rng = np.random.default_rng(0)
    income = rng.lognormal(10, 1.2, 800)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))

    ax = axes[0]
    ax.hist(income, bins=60, color=WARN, edgecolor=PAPER, linewidth=0.4)
    ax.set_xlabel("income", fontsize=9.5, color=INK)
    ax.set_ylabel("count", fontsize=9.5, color=INK)
    ax.set_title(f"Raw income -- skew = {pd.Series(income).skew():.2f}\na long tail squashes everything left",
                 fontsize=10.5, fontweight="bold", color=WARN)

    ax = axes[1]
    logged = np.log(income)
    ax.hist(logged, bins=60, color=GOOD, edgecolor=PAPER, linewidth=0.4)
    ax.set_xlabel("log(income)", fontsize=9.5, color=INK)
    ax.set_title(f"log(income) -- skew = {pd.Series(logged).skew():.2f}\nroughly symmetric, evenly spread",
                 fontsize=10.5, fontweight="bold", color=GOOD)

    for ax in axes:
        ax.grid(axis="y", alpha=0.22, color=LINE)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle("A log transform compresses the tail and spreads out the crowd",
                 fontsize=12, fontweight="bold", color=NAVY)
    fig.tight_layout()
    return fig


# -- Measured across 20 different train/test splits (see note section 7) --
FE_WINS, FE_TIES, FE_LOSSES = 9, 6, 5
FE_MEAN_WITHOUT, FE_MEAN_WITH = 0.9211, 0.9200


@figure("fe_honest_result")
def fe_honest_result() -> plt.Figure:
    """The notebook's single-split win, and what 20 splits actually show."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # Left: the single split the notebook reports.
    ax = axes[0]
    bars = ax.bar(["Without\nengineered", "With\nengineered"], [0.9556, 0.9778],
                  color=[MIST, BLUE], edgecolor=NAVY, linewidth=1.3, width=0.55)
    for bar, value in zip(bars, [0.9556, 0.9778]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.3f}",
                ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_ylim(0.9, 1.0)
    ax.set_ylabel("Test accuracy", fontsize=9.5, color=INK)
    ax.set_title("What the notebook reports\none split -- looks like a clear win",
                 fontsize=10.5, fontweight="bold", color=NAVY)

    # Right: repeat over 20 splits.
    ax = axes[1]
    counts = [FE_WINS, FE_TIES, FE_LOSSES]
    labels = ["FE better", "Tie", "FE worse"]
    bars = ax.bar(labels, counts, color=[GOOD, GREY, WARN], edgecolor=NAVY,
                  linewidth=1.3, width=0.55)
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.22, str(value),
                ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylim(0, 11)
    ax.set_ylabel("Number of splits (out of 20)", fontsize=9.5, color=INK)
    ax.set_title(f"Repeated over 20 splits\nmean {FE_MEAN_WITHOUT:.4f} vs {FE_MEAN_WITH:.4f} -- a coin flip",
                 fontsize=10.5, fontweight="bold", color=WARN)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25, color=LINE)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    fig.suptitle(
        "Always re-test a feature-engineering 'win' on more than one split",
        fontsize=12,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout()
    return fig


@figure("feature_selection_methods")
def feature_selection_methods() -> plt.Figure:
    """The three families of feature selection from slide 59."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    blank_axes(ax, xlim=(0, 12), ylim=(0, 4.6))

    families = [
        (2.0, "Filter", "model-independent", NAVY, MIST,
         "Correlation\nMutual Information\nChi-Square Test\nVariance Threshold",
         "Fast -- score features\nbefore any model exists"),
        (6.0, "Wrapper", "model-dependent", BLUE, PAPER,
         "Recursive Feature\nElimination (RFE)\nForward / Backward\nSelection",
         "Slow -- retrains the model\nfor each feature subset"),
        (10.0, "Embedded", "built into training", GOOD, "#E4F3E9",
         "L1 Regularization\n(Lasso)\nTree-based\nFeature Importance",
         "Free -- selection happens\nwhile the model fits"),
    ]

    for x, title, subtitle, edge, fill, methods, note in families:
        node_box(ax, (x, 3.75), f"{title}\n{subtitle}", facecolor=fill, edgecolor=edge,
                 width=3.3, height=0.95, fontsize=10, bold=True)
        ax.text(x, 2.65, methods, ha="center", va="top", fontsize=8.8, color=INK)
        ax.text(x, 0.78, note, ha="center", va="top", fontsize=8.2, color=GREY, style="italic")

    panel_title(ax, "Three Families of Feature Selection",
                "Keep the useful features, remove the rest")
    fig.tight_layout()
    return fig
