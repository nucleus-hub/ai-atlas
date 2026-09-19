"""Figures for Note 10 -- Demo 2, applications of Transformers."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, MIST, NAVY, PAPER, SKY, WARN, arrow, blank_axes, figure, stage_box

GROUPS = [
    ("Encoder-Only", ["Sentiment classification", "Extractive QA", "NER",
                       "POS tagging", "Paraphrase detection", "Fill-mask"], MIST),
    ("Sentence Transf.", ["Semantic similarity", "Text clustering"], SKY),
    ("Decoder-Only", ["Text generation", "Decoding strategies",
                        "Code generation", "Chat-style generation"], SKY),
    ("Encoder-Decoder", ["Summarization", "Paraphrasing", "Grammar correction",
                          "Zero-shot classification", "Translation"], BLUE),
    ("Vision Transformer", ["Image classification", "Image captioning"], MIST),
]


@figure("demo2_architecture_map")
def demo2_architecture_map() -> plt.Figure:
    """Every Demo 2 application grouped by the architecture family behind it."""
    fig, ax = plt.subplots(figsize=(15.4, 7.4))
    blank_axes(ax, xlim=(0, 17), ylim=(0, 9.4))
    for spine in ax.spines.values():
        spine.set_visible(False)

    x_positions = [1.7, 5.3, 8.9, 12.3, 15.3]
    widths = [2.9, 2.6, 2.6, 2.8, 2.6]
    for (title, tasks, shade), x, w in zip(GROUPS, x_positions, widths):
        stage_box(ax, (x, 8.7), title, facecolor=shade, width=w, height=0.7, fontsize=9.6,
                  textcolor=PAPER if shade == BLUE else NAVY)
        for i, task in enumerate(tasks):
            y = 7.7 - i * 0.85
            stage_box(ax, (x, y), task, facecolor=PAPER, edgecolor=shade, width=w, height=0.62,
                      fontsize=7.9, bold=False)

    ax.set_title("Demo 2: every application, grouped by architecture",
                 fontsize=13, color=NAVY, fontweight="bold", pad=14)
    fig.tight_layout()
    return fig


@figure("fill_mask_predictions")
def fill_mask_predictions() -> plt.Figure:
    """Top-3 BERT fill-mask predictions for a context-dependent sentence."""
    fig, ax = plt.subplots(figsize=(12.2, 4.2))
    blank_axes(ax, xlim=(0, 12), ylim=(0, 5.0))
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(6.0, 4.6, 'Sentence: "She opened the [MASK] and started reading the first chapter."',
            ha="center", fontsize=9.6, color=NAVY, fontweight="bold")

    preds = [("book", 0.8110), ("journal", 0.0465), ("magazine", 0.0209)]
    y0 = 3.5
    for i, (word, score) in enumerate(preds):
        y = y0 - i * 1.05
        ax.text(1.3, y, word, ha="right", fontsize=10, color=INK, fontweight="bold")
        bar_w = score * 8.5
        ax.add_patch(plt.Rectangle((1.5, y - 0.28), bar_w, 0.56,
                                    facecolor=GOOD if i == 0 else SKY, edgecolor="none"))
        ax.text(1.5 + bar_w + 0.2, y, f"{score:.2%}", ha="left", va="center", fontsize=9.2,
                color=INK)

    ax.text(6.0, 0.4,
            '"book" dominates at 81% -- BERT uses both left AND right context to fill the blank.',
            ha="center", fontsize=9.6, color=NAVY, fontweight="bold")
    ax.set_title("Fill-mask: bidirectional context in action", fontsize=12.5, color=NAVY,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("vision_transformer_patches")
def vision_transformer_patches() -> plt.Figure:
    """An image split into patches, treated exactly like word tokens."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 4.8))

    blank_axes(left, xlim=(0, 8), ylim=(0, 8))
    for spine in left.spines.values():
        spine.set_visible(False)
    left.set_title("An image, split into patches", fontsize=11.2, color=NAVY, fontweight="bold", pad=8)
    n = 4
    for i in range(n):
        for j in range(n):
            shade = SKY if (i + j) % 2 == 0 else MIST
            left.add_patch(plt.Rectangle((1 + j * 1.5, 1 + i * 1.5), 1.4, 1.4,
                                          facecolor=shade, edgecolor=NAVY, lw=1.2))
    left.text(4.0, 7.4, "224 x 224 image -> 16 x 16 patches", ha="center", fontsize=9, color=GREY)

    blank_axes(right, xlim=(0, 8), ylim=(0, 8))
    for spine in right.spines.values():
        spine.set_visible(False)
    right.set_title("Each patch becomes a token", fontsize=11.2, color=NAVY, fontweight="bold", pad=8)
    labels = ["patch 1", "patch 2", "patch 3", "..."]
    for i, lab in enumerate(labels):
        y = 6.4 - i * 1.5
        stage_box(right, (2.0, y), lab, facecolor=SKY, width=2.2, height=0.8, fontsize=8.8)
        arrow(right, (3.15, y), (4.35, y), scale=11)
        stage_box(right, (5.5, y), "vector", facecolor=MIST, width=1.8, height=0.8, fontsize=8.8)
    right.text(4.0, 0.5, "Same self-attention mechanism as BERT over word tokens.",
               ha="center", fontsize=9, color=NAVY, fontweight="bold")

    fig.suptitle("Vision Transformer: attention is domain-agnostic",
                 fontsize=13, color=NAVY, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
