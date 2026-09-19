"""Figures for Note 05 -- the Demo 1 preprocessing walkthrough."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, MIST, NAVY, PAPER, SKY, arrow, blank_axes, figure, stage_box

# (label, library) for each hands-on step of the demo.
DEMO_STEPS = [
    ("Review\nDataset", "—"),
    ("Sentence\nSegment", "spaCy"),
    ("Tokenize", "spaCy + NLTK"),
    ("Regex\nClean", "re"),
    ("POS Tag", "spaCy"),
    ("Dependency\nParse", "spaCy"),
    ("Chunk", "spaCy"),
    ("Stopword\nRemoval", "NLTK"),
    ("Lemmatize", "NLTK"),
    ("One\nFunction", "all"),
]


@figure("demo1_pipeline_flow")
def demo1_pipeline_flow() -> plt.Figure:
    """The demo's hands-on route, annotated with the library used at each step."""
    fig, ax = plt.subplots(figsize=(13.2, 5.4))
    blank_axes(ax, xlim=(0, 15.2), ylim=(0, 6.4))
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(7.5, 6.05,
            'Input:  "Battery life is terrible, it barely lasts half a day."',
            ha="center", fontsize=10.5, color=NAVY, fontweight="bold")

    per_row = 5
    width, gap = 2.5, 0.42
    for i, (label, library) in enumerate(DEMO_STEPS):
        row, col = divmod(i, per_row)
        x = 1.6 + col * (width + gap)
        y = 4.55 - row * 2.5
        shade = BLUE if i == len(DEMO_STEPS) - 1 else (SKY if i % 2 else MIST)
        stage_box(ax, (x, y), label, facecolor=shade, width=width, height=0.92, fontsize=9.2,
                  textcolor=PAPER if shade == BLUE else NAVY)
        ax.text(x, y - 0.74, library, ha="center", va="center", fontsize=8.2, color=GREY,
                fontweight="bold")
        if col:
            arrow(ax, (x - width / 2 - gap - 0.02, y), (x - width / 2 + 0.02, y), scale=12)

    # Wrap-around from the end of row 1 to the start of row 2, routed through the
    # empty band between the rows so it never crosses a box.
    band = 3.30
    right_x, left_x = 14.62, 0.62
    row0_y, row1_y = 4.55, 2.05
    for start, end, style in [
        ((14.10, row0_y), (right_x, row0_y), "-"),
        ((right_x, row0_y), (right_x, band), "-"),
        ((right_x, band), (left_x, band), "-"),
        ((left_x, band), (left_x, row1_y), "-"),
        ((left_x, row1_y), (0.34 + 1.25 - 0.32, row1_y), "-|>"),
    ]:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=style, color=GOOD, lw=1.5, mutation_scale=13))
    ax.text(7.5, band + 0.22, "continues", ha="center", fontsize=8.2, color=GOOD,
            fontweight="bold", bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5))

    ax.text(7.5, 0.45,
            "Output:  tokens + POS tags + chunks + lemmas  ->  'battery life terrible barely last half day'",
            ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.set_title("Demo 1: the preprocessing and linguistic-analysis route",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig
