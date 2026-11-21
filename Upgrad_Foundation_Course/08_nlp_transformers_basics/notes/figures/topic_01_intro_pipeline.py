"""Figures for Note 01 -- introduction to NLP and the NLP pipeline."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import BLUE, GOOD, INK, MIST, NAVY, PAPER, SKY, WARN, arrow, blank_axes, figure, stage_box

STAGES = [
    ("1\nRaw Text", "Original\ninput string"),
    ("2\nTokenize", '"The", "movie",\n"was" ...'),
    ("3\nClean", "Remove noise,\nlowercase"),
    ("4\nPOS Tag", "movie = NOUN\nloved = VERB"),
    ("5\nParse", "Build\ngrammar tree"),
    ("6\nUnderstand", "Extract sentiment\n/ entities"),
]


@figure("nlp_pipeline_stages")
def nlp_pipeline_stages() -> plt.Figure:
    """The six-stage assembly line that raw text travels through."""
    fig, ax = plt.subplots(figsize=(13.5, 4.6))
    blank_axes(ax, xlim=(0, 15), ylim=(0, 5))
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        7.5,
        4.5,
        'Input:  "The movie was absolutely fantastic! I loved every scene."',
        ha="center",
        fontsize=10.5,
        color=NAVY,
        fontweight="bold",
    )

    width, gap = 2.1, 0.3
    left = 0.4
    for i, (title, detail) in enumerate(STAGES):
        x = left + width / 2 + i * (width + gap)
        shade = SKY if i % 2 else MIST
        stage_box(ax, (x, 2.9), title, facecolor=shade, width=width, height=1.0, fontsize=10)
        ax.text(x, 1.85, detail, ha="center", va="center", fontsize=8.2, color=INK)
        if i:
            arrow(ax, (x - width / 2 - gap - 0.02, 2.9), (x - width / 2 + 0.02, 2.9), scale=13)

    ax.text(
        7.5,
        0.7,
        "Each step builds on the previous one -- like a factory assembly line for language.",
        ha="center",
        fontsize=10,
        color=NAVY,
        fontweight="bold",
    )
    ax.set_title("A simplified NLP pipeline", fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("nlp_characters_vs_meaning")
def nlp_characters_vs_meaning() -> plt.Figure:
    """What a machine sees without NLP, versus what NLP adds."""
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    blank_axes(ax, xlim=(0, 13), ylim=(0, 5))
    for spine in ax.spines.values():
        spine.set_visible(False)

    stage_box(ax, (3.1, 3.9), "Without NLP\nthe machine sees characters", facecolor=PAPER, edgecolor=WARN, width=4.8, height=0.95, fontsize=10)
    ax.text(3.1, 2.75, "f  a  n  t  a  s  t  i  c", ha="center", fontsize=12, color=WARN, family="monospace")
    ax.text(3.1, 1.85, "nine arbitrary symbols\nno meaning, no intent, no context", ha="center", fontsize=8.8, color=INK)

    stage_box(ax, (9.9, 3.9), "With NLP\nthe machine sees language", facecolor=BLUE, edgecolor=NAVY, textcolor=PAPER, width=4.8, height=0.95, fontsize=10)
    ax.text(9.9, 2.8, "fantastic  ->  ADJECTIVE,\nstrongly positive", ha="center", fontsize=9.5, color=GOOD, fontweight="bold")
    ax.text(9.9, 1.85, "meaning + intent + context\nusable by a downstream model", ha="center", fontsize=8.8, color=INK)

    arrow(ax, (5.75, 3.9), (7.35, 3.9), scale=17, linewidth=2.0)
    ax.text(6.55, 4.25, "NLP", ha="center", fontsize=10.5, color=NAVY, fontweight="bold")

    ax.text(
        6.5,
        0.55,
        "NLP is the bridge from raw characters to meaning, intent and context.",
        ha="center",
        fontsize=10,
        color=NAVY,
        fontweight="bold",
    )
    ax.set_title("What NLP actually adds", fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig
