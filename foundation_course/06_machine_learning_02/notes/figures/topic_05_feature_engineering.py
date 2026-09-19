"""Figures for Topic 05 -- Feature Engineering."""

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
    blank_axes,
    figure,
    node_box,
    panel_title,
)


@figure("feature_engineering_process")
def feature_engineering_process() -> plt.Figure:
    """The five-stage loop, including the iterate-back arrow."""
    fig, ax = plt.subplots(figsize=(11.2, 3.2))
    blank_axes(ax, (0, 11.4), (0, 3.4))
    panel_title(ax, "The Feature Engineering Process", "")

    steps = [
        ("1\nUnderstand", "business problem\n+ explore data"),
        ("2\nCreate Features", "generate new ones\nfrom existing data"),
        ("3\nTransform", "make features\nmodel-friendly"),
        ("4\nSelect Features", "keep the useful,\ndrop the rest"),
        ("5\nValidate & Iterate", "evaluate, then\nrepeat as needed"),
    ]
    xs = np.linspace(1.2, 10.2, 5)
    for i, (x, (title, sub)) in enumerate(zip(xs, steps)):
        first_or_last = i in (0, 4)
        node_box(
            ax, (x, 2.3), title,
            facecolor=NAVY if first_or_last else MIST,
            textcolor=PAPER if first_or_last else INK,
            edgecolor=NAVY if first_or_last else BLUE,
            width=1.9, height=0.9, fontsize=8.5, bold=True,
        )
        ax.text(x, 1.42, sub, ha="center", va="center", fontsize=7.5, color=GREY)
        if i < 4:
            ax.annotate("", xy=(xs[i + 1] - 0.98, 2.3), xytext=(x + 0.98, 2.3),
                        arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.6))

    ax.annotate("", xy=(xs[1], 0.85), xytext=(xs[4], 0.85),
                arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.5, linestyle="--"))
    ax.text((xs[1] + xs[4]) / 2, 0.52, "iterate -- feature engineering is a loop, not a checklist",
            ha="center", fontsize=8.5, color=WARN, style="italic")
    fig.tight_layout()
    return fig


@figure("log_transform_effect")
def log_transform_effect() -> plt.Figure:
    """A skewed income distribution before and after a log transform."""
    rng = np.random.default_rng(12)
    income = rng.lognormal(mean=10.2, sigma=0.85, size=3000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax1.hist(income, bins=60, color=MIST, edgecolor=BLUE, linewidth=0.5)
    ax1.set_title("Raw income -- heavily right-skewed", fontsize=10.5, color=NAVY, fontweight="bold")
    ax1.set_xlabel("income", fontsize=9, color=INK)

    ax2.hist(np.log(income), bins=60, color="#CDEBD6", edgecolor=GOOD, linewidth=0.5)
    ax2.set_title("log(income) -- roughly symmetric", fontsize=10.5, color=NAVY, fontweight="bold")
    ax2.set_xlabel("log(income)", fontsize=9, color=INK)

    for ax in (ax1, ax2):
        ax.set_yticks([])
        ax.grid(axis="x", color=LINE, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_edgecolor(LINE)

    fig.suptitle("Mathematical Transforms: same information, easier shape to learn",
                 fontsize=11.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("binning_ages")
def binning_ages() -> plt.Figure:
    """Continuous age turned into four ordered bins."""
    rng = np.random.default_rng(8)
    ages = np.clip(rng.normal(38, 16, 1200), 1, 92)
    edges = [0, 18, 35, 60, 100]
    names = ["0-18", "19-35", "36-60", "60+"]
    counts = [int(((ages > lo) & (ages <= hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8))
    ax1.hist(ages, bins=45, color=MIST, edgecolor=BLUE, linewidth=0.5)
    for e in edges[1:-1]:
        ax1.axvline(e, color=WARN, linestyle="--", linewidth=1.6)
    ax1.set_title("Continuous age with bin boundaries", fontsize=10.5, color=NAVY, fontweight="bold")
    ax1.set_xlabel("age", fontsize=9, color=INK)
    ax1.set_yticks([])

    ax2.bar(names, counts, color=NAVY, edgecolor=NAVY, width=0.6)
    for i, c in enumerate(counts):
        ax2.text(i, c + 12, str(c), ha="center", fontsize=9, color=INK)
    ax2.set_title("After binning: four ordered categories", fontsize=10.5, color=NAVY, fontweight="bold")
    ax2.set_xlabel("age_group", fontsize=9, color=INK)
    ax2.set_yticks([])

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=LINE, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_edgecolor(LINE)

    fig.suptitle("Binning (Discretization)", fontsize=11.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("interaction_feature")
def interaction_feature() -> plt.Figure:
    """Two weak features multiplied into one strong one."""
    rng = np.random.default_rng(15)
    quantity = rng.integers(1, 12, 160)
    price = rng.uniform(2, 40, 160)
    total = quantity * price
    churn = (total < 80).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for label, colour, marker in ((0, NAVY, "o"), (1, "#E07A3F", "s")):
        sel = churn == label
        ax1.scatter(quantity[sel], price[sel], s=24, color=colour, marker=marker, alpha=0.75,
                    label="high value" if label == 0 else "low value")
    ax1.set_xlabel("quantity", fontsize=9.5, color=INK)
    ax1.set_ylabel("unit_price", fontsize=9.5, color=INK)
    ax1.set_title("Separately: no clean boundary", fontsize=10.5, color=NAVY, fontweight="bold")
    ax1.legend(fontsize=8, frameon=False)

    for label, colour in ((0, NAVY), (1, "#E07A3F")):
        sel = churn == label
        ax2.scatter(total[sel], rng.normal(0, 1, sel.sum()), s=24, color=colour, alpha=0.75)
    ax2.axvline(80, color=GOOD, linestyle="--", linewidth=2)
    ax2.text(88, 2.4, "one clean split", fontsize=9, color=GOOD)
    ax2.set_xlabel("total_spend = quantity x unit_price", fontsize=9.5, color=INK)
    ax2.set_yticks([])
    ax2.set_title("Combined: the pattern appears", fontsize=10.5, color=NAVY, fontweight="bold")

    for ax in (ax1, ax2):
        ax.grid(color=LINE, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Interaction Features: A x B can beat A and B", fontsize=11.5, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("encoding_categorical")
def encoding_categorical() -> plt.Figure:
    """One-hot versus label encoding, and the ordering trap."""
    fig, ax = plt.subplots(figsize=(9.4, 3.8))
    blank_axes(ax, (0, 10), (0, 4.6))
    panel_title(ax, "Encoding Categorical Features", "Turning categories into numbers without inventing an order")

    cats = ["Sunny", "Overcast", "Rainy"]
    for i, c in enumerate(cats):
        node_box(ax, (1.1, 3.3 - i * 0.75), c, facecolor=MIST, width=1.6, height=0.55, fontsize=8.5)

    ax.text(3.9, 4.05, "One-Hot Encoding", ha="center", fontsize=9.5, color=GOOD, fontweight="bold")
    for i in range(3):
        row = ["1" if j == i else "0" for j in range(3)]
        for j, v in enumerate(row):
            node_box(ax, (3.1 + j * 0.72, 3.3 - i * 0.75), v,
                     facecolor="#CDEBD6" if v == "1" else PAPER,
                     edgecolor=GOOD if v == "1" else LINE, width=0.62, height=0.52, fontsize=8.5)
    ax.text(3.9, 0.85, "no fake ordering\nsafe default", ha="center", fontsize=8, color=GOOD)

    ax.text(7.4, 4.05, "Label Encoding", ha="center", fontsize=9.5, color=WARN, fontweight="bold")
    for i, v in enumerate(["0", "1", "2"]):
        node_box(ax, (7.4, 3.3 - i * 0.75), v, facecolor="#FBE3E0", edgecolor=WARN,
                 width=0.62, height=0.52, fontsize=8.5)
    ax.text(7.4, 0.85, "implies Rainy > Sunny\nfine for trees, risky elsewhere",
            ha="center", fontsize=8, color=WARN)
    fig.tight_layout()
    return fig


@figure("feature_selection_methods")
def feature_selection_methods() -> plt.Figure:
    """Filter, wrapper and embedded selection families."""
    fig, ax = plt.subplots(figsize=(9.8, 3.6))
    blank_axes(ax, (0, 10), (0, 4))
    panel_title(ax, "Three Families of Feature Selection", "")

    families = [
        (1.9, "Filter", "model-independent", "Correlation\nMutual Information\nChi-Square\nVariance Threshold", MIST),
        (5.0, "Wrapper", "model-dependent", "RFE\nForward selection\nBackward selection", "#FDE9D9"),
        (8.1, "Embedded", "built into training", "L1 (Lasso)\nTree feature\nimportance", "#CDEBD6"),
    ]
    for x, name, sub, items, colour in families:
        node_box(ax, (x, 3.0), f"{name}\n{sub}", facecolor=colour, width=2.7, height=0.85, fontsize=9, bold=True)
        ax.text(x, 1.55, items, ha="center", va="center", fontsize=8.2, color=INK)

    ax.text(5.0, 0.3, "cheap and fast  ->  slow and thorough  ->  free with the model",
            ha="center", fontsize=8.5, color=GREY, style="italic")
    fig.tight_layout()
    return fig


@figure("data_leakage")
def data_leakage() -> plt.Figure:
    """The leakage trap: scaling before splitting versus after."""
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    blank_axes(ax, (0, 10), (0, 4.2))
    panel_title(ax, "Avoiding Data Leakage", "Only use information available at prediction time")

    node_box(ax, (2.4, 3.4), "WRONG", facecolor="#FBE3E0", edgecolor=WARN, textcolor=WARN,
             width=1.6, height=0.5, fontsize=9, bold=True)
    for i, step in enumerate(["Scale the FULL dataset", "Split into train / test", "Test stats already leaked"]):
        node_box(ax, (2.4, 2.5 - i * 0.72), step, facecolor=PAPER, edgecolor=WARN if i == 2 else LINE,
                 textcolor=WARN if i == 2 else INK, width=3.4, height=0.55, fontsize=8)

    node_box(ax, (7.0, 3.4), "RIGHT", facecolor="#CDEBD6", edgecolor=GOOD, textcolor=GOOD,
             width=1.6, height=0.5, fontsize=9, bold=True)
    for i, step in enumerate(["Split into train / test", "Fit the scaler on TRAIN only", "Apply it to test"]):
        node_box(ax, (7.0, 2.5 - i * 0.72), step, facecolor=PAPER, edgecolor=GOOD if i == 2 else LINE,
                 textcolor=GOOD if i == 2 else INK, width=3.4, height=0.55, fontsize=8)

    ax.text(5.0, 0.2, "Leakage produces gorgeous test scores and a model that fails in production",
            ha="center", fontsize=8.5, color=GREY, style="italic")
    fig.tight_layout()
    return fig
