"""Figures for note 02 -- Data Preparation & Train-Test Split."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, LINE, NAVY, PAPER, SKY, WARN, blank_axes, figure, panel_title


@figure("preprocessing_pipeline")
def _preprocessing_pipeline() -> plt.Figure:
    """Raw -> cleaned/encoded -> scaled, shown as three snapshots of one table."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), facecolor=PAPER)

    stages = [
        (
            "1. RAW DATA",
            "blanks and text -- the model crashes",
            ["Age", "Salary", "City", "Purch."],
            [
                ["25", "50000", "Delhi", "No"],
                ["?", "60000", "Mumbai", "Yes"],
                ["30", "", "Delhi", "No"],
                ["25", "50000", "Delhi", "No"],
                ["35", "120000", "?", "Yes"],
            ],
            {(1, 0), (2, 1), (4, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3)},
            WARN,
        ),
        (
            "2. CLEANED & ENCODED",
            "gaps filled, text turned into numbers",
            ["Age", "Salary", "City", "Purch."],
            [
                ["25", "50000", "Delhi", "0"],
                ["28", "60000", "Mumbai", "1"],
                ["30", "85000", "Delhi", "0"],
                ["25", "50000", "Delhi", "0"],
                ["35", "120000", "B'lore", "1"],
            ],
            {(1, 0), (2, 1), (4, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3)},
            GOOD,
        ),
        (
            "3. MODEL-READY",
            "scaled -- no feature dominates",
            ["Age", "Salary", "City", "Purch."],
            [
                ["-1.2", "0.3", "1", "0"],
                ["-0.1", "0.8", "2", "1"],
                ["0.5", "0.0", "1", "0"],
                ["-1.2", "-0.5", "1", "0"],
                ["1.1", "1.7", "3", "1"],
            ],
            set(),
            GOOD,
        ),
    ]

    for ax, (title, subtitle, headers, rows, marks, mark_colour) in zip(axes, stages):
        ax.axis("off")
        ax.set_title(f"{title}\n{subtitle}", fontsize=11.5, fontweight="bold", color=NAVY, pad=12)

        table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.7)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor(LINE)
            if row == 0:
                cell.set_facecolor("#CFE0FF")
                cell.set_text_props(fontweight="bold", color=NAVY)
            elif (row - 1, col) in marks:
                cell.set_facecolor("#FDEDEC" if mark_colour == WARN else "#EAF7EF")
                cell.set_text_props(color=mark_colour, fontweight="bold")
            else:
                cell.set_facecolor(PAPER)

    fig.suptitle("The Preprocessing Journey: Raw → Clean → Model-Ready", fontsize=15,
                 fontweight="bold", color=NAVY, y=1.04)
    fig.tight_layout()
    # Widen the column gaps first, then drop the arrows into the clear space.
    fig.subplots_adjust(wspace=0.30)
    for x in (0.343, 0.671):
        fig.text(x, 0.46, "→", fontsize=30, color=BLUE, ha="center", va="center", fontweight="bold")
    return fig


@figure("encoding_comparison")
def _encoding_comparison() -> plt.Figure:
    """Label encoding's false ladder versus one-hot's independent switches."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), facecolor=PAPER)
    cities = ["Delhi", "Mumbai", "B'lore"]

    # -- Left: label encoding invents an ordering ---------------------------
    ax = axes[0]
    for i, (city, code) in enumerate(zip(cities, [1, 2, 3])):
        y = 3.9 - i * 1.15
        ax.text(0.35, y, city, fontsize=11, color=INK, va="center")
        ax.annotate("", xy=(2.05, y), xytext=(1.45, y),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.5))
        ax.add_patch(plt.Rectangle((2.15, y - 0.28), 0.62, 0.56, facecolor="#CFE0FF",
                                   edgecolor=NAVY, linewidth=1.5))
        ax.text(2.46, y, str(code), fontsize=13, fontweight="bold", color=NAVY,
                ha="center", va="center")

    # Arrow points UP the invented ladder: Delhi (baseline) -> B'lore (3x).
    ax.annotate("", xy=(3.95, 1.45), xytext=(3.95, 3.95),
                arrowprops=dict(arrowstyle="-|>", color=WARN, lw=2))
    ladder = [(3.90, "baseline"), (2.75, "2x Delhi?"), (1.60, "3x Delhi?")]
    for y, label in ladder:
        ax.text(4.12, y, label, fontsize=8.5, color=WARN, va="center")
    ax.text(3.95, 0.95, "the model reads a LADDER\nthat does not exist", fontsize=9.5,
            color=WARN, ha="center", va="center", fontweight="bold")
    panel_title(ax, "Label Encoding on nominal data", "invents a false ordering")

    # -- Right: one-hot gives each category its own switch ------------------
    ax = axes[1]
    headers = ["is_Delhi", "is_Mumbai", "is_B'lore"]
    for j, head in enumerate(headers):
        ax.text(1.65 + j * 1.15, 4.42, head, fontsize=9, color=NAVY, ha="center", fontweight="bold")

    rows = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for i, (city, row) in enumerate(zip(cities, rows)):
        y = 3.75 - i * 1.05
        ax.text(0.30, y, city, fontsize=11, color=INK, va="center")
        for j, value in enumerate(row):
            hot = value == 1
            ax.add_patch(plt.Rectangle((1.35 + j * 1.15, y - 0.30), 0.60, 0.60,
                                       facecolor=SKY if hot else PAPER,
                                       edgecolor=NAVY if hot else LINE,
                                       linewidth=1.8 if hot else 1.2))
            ax.text(1.65 + j * 1.15, y, str(value), fontsize=12,
                    fontweight="bold" if hot else "normal",
                    color=PAPER if hot else GREY, ha="center", va="center")

    ax.text(2.80, 0.72, "exactly ONE switch is hot per row\nno ordering, no fake arithmetic",
            fontsize=9.5, color=GOOD, ha="center", va="center", fontweight="bold")
    panel_title(ax, "One-Hot Encoding", "one independent switch per category")

    for ax in axes:
        blank_axes(ax, xlim=(0, 5.6), ylim=(0.3, 4.9))

    fig.suptitle("Encoding Categorical Data: Label vs One-Hot", fontsize=15,
                 fontweight="bold", color=NAVY, y=1.00)
    fig.tight_layout()
    return fig


@figure("scaling_comparison")
def _scaling_comparison() -> plt.Figure:
    """Age vs Salary before scaling, after min-max, and after z-score."""
    ages = np.array([25, 28, 30, 25, 35], dtype=float)
    salaries = np.array([50000, 60000, 85000, 50000, 120000], dtype=float)

    def minmax(v: np.ndarray) -> np.ndarray:
        return (v - v.min()) / (v.max() - v.min())

    def zscore(v: np.ndarray) -> np.ndarray:
        return (v - v.mean()) / v.std()

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), facecolor=PAPER)
    idx = np.arange(len(ages))
    width = 0.36

    panels = [
        ("Before scaling", "Salary swamps Age entirely", ages, salaries, False),
        ("Normalization (Min-Max)", "everything squashed into 0 to 1", minmax(ages), minmax(salaries), True),
        ("Standardization (Z-Score)", "centred on 0, in std units", zscore(ages), zscore(salaries), True),
    ]

    for ax, (title, subtitle, a_vals, s_vals, comparable) in zip(axes, panels):
        ax.bar(idx - width / 2, a_vals, width, color=BLUE, label="Age")
        ax.bar(idx + width / 2, s_vals, width, color=NAVY, label="Salary")
        panel_title(ax, title, subtitle)
        ax.set_xticks(idx)
        ax.set_xticklabels([f"P{i + 1}" for i in idx], fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.axhline(0, color=INK, linewidth=1.0)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)
        if not comparable:
            ax.text(2.0, salaries.max() * 0.55,
                    "Age bars are invisible --\nthey are ~3000x smaller",
                    fontsize=9, color=WARN, ha="center", fontweight="bold")

    fig.suptitle("Why Scaling Matters: the Same Two Features, Three Ways", fontsize=15,
                 fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig


@figure("train_test_split")
def _train_test_split() -> plt.Figure:
    """The 80/20 bar with the learning and checking parts labelled."""
    fig, ax = plt.subplots(figsize=(11.5, 3.9), facecolor=PAPER)

    ax.add_patch(plt.Rectangle((0, 1.55), 8.0, 1.15, facecolor="#CFE0FF", edgecolor=NAVY, linewidth=2))
    ax.add_patch(plt.Rectangle((8.0, 1.55), 2.0, 1.15, facecolor="#FDE9C9", edgecolor="#B9770E", linewidth=2))

    ax.text(4.0, 2.28, "TRAINING SET — 80%", fontsize=13, fontweight="bold", color=NAVY, ha="center")
    ax.text(4.0, 1.88, "Learning Part", fontsize=10.5, color=NAVY, ha="center", style="italic")
    ax.text(9.0, 2.28, "TEST — 20%", fontsize=11.5, fontweight="bold", color="#7E5109", ha="center")
    ax.text(9.0, 1.88, "Checking Part", fontsize=9.5, color="#7E5109", ha="center", style="italic")

    ax.text(4.0, 1.05, "The model studies this data to learn\npatterns and relationships.",
            fontsize=9.5, color=INK, ha="center", va="top")
    ax.text(9.0, 1.05, "Held back and used only to\nevaluate generalization.",
            fontsize=9.5, color=INK, ha="center", va="top")

    ax.annotate("", xy=(10.0, 3.05), xytext=(0, 3.05),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.4))
    ax.text(5.0, 3.22, "Full dataset", fontsize=10, color=GREY, ha="center", style="italic")
    ax.text(5.0, 0.18, "The test set is touched ONCE, at the end — never for training, tuning or feature selection.",
            fontsize=9.5, color=WARN, ha="center", fontweight="bold")

    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    ax.set_title("Train-Test Split", fontsize=15, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig


@figure("overfitting")
def _overfitting() -> plt.Figure:
    """Underfit, good fit and overfit against the same noisy sample."""
    rng = np.random.default_rng(7)
    x = np.linspace(0.4, 4.6, 14)
    y = 1.1 + 0.55 * x + 0.42 * np.sin(1.7 * x) + rng.normal(0, 0.16, x.size)

    fits = [
        ("Underfit", "too simple — misses the pattern", 1, WARN, "Train 64%  ·  Test 63%"),
        ("Good fit", "learned the real trend", 3, GOOD, "Train 88%  ·  Test 86%"),
        ("Overfit", "memorized the noise", 11, WARN, "Train 100%  ·  Test 61%"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), facecolor=PAPER)

    for ax, (title, subtitle, degree, colour, score) in zip(axes, fits):
        ax.scatter(x, y, s=70, color=BLUE, zorder=3)
        curve = np.poly1d(np.polyfit(x, y, degree))
        # Stay inside the sampled range and clip the tail -- a high-degree fit
        # blows up at the edges, which distracts from the point being made.
        span = np.linspace(x.min(), x.max(), 400)
        ax.plot(span, np.clip(curve(span), 0.35, 4.55), color=colour, linewidth=2.2, zorder=2)
        panel_title(ax, title, subtitle)
        ax.text(2.5, 0.55, score, fontsize=10, color=colour, ha="center", fontweight="bold")
        blank_axes(ax, xlim=(0.2, 4.8), ylim=(0.3, 4.6))

    fig.suptitle("Underfitting vs Good Fit vs Overfitting", fontsize=15,
                 fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig
