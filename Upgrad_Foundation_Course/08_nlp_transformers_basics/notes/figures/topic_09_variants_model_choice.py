"""Figures for Note 09 -- Transformer variants and model choice."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, MIST, NAVY, PAPER, SKY, WARN, arrow, blank_axes, figure, stage_box

VARIANTS = [
    ("Encoder-Only", "KEEP\nBidirectional self-attention", "REMOVE\nDecoder, cross-attention",
     "Needs to fully understand text,\nnot generate it", "BERT, RoBERTa", MIST),
    ("Decoder-Only", "KEEP\nMasked self-attention", "REMOVE\nEncoder, cross-attention",
     "Needs to generate text\none word at a time", "GPT family", SKY),
    ("Encoder-Decoder", "KEEP\nBoth stacks + cross-attention", "REMOVE\nNothing",
     "Needs to transform one\nsequence into another", "T5, BART", BLUE),
]


@figure("transformer_three_variants")
def transformer_three_variants() -> plt.Figure:
    """One blueprint, three variants -- keep only what each job needs."""
    fig, ax = plt.subplots(figsize=(13.2, 5.6))
    blank_axes(ax, xlim=(0, 14.4), ylim=(0, 7.4))
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(7.2, 7.05, "Take the full architecture and keep only the parts each job needs.",
            ha="center", fontsize=10.3, color=NAVY, fontweight="bold")

    for i, (name, keep, remove, best, examples, shade) in enumerate(VARIANTS):
        x = 2.4 + i * 4.8
        stage_box(ax, (x, 6.0), name, facecolor=shade, width=4.2, height=0.8, fontsize=11,
                  textcolor=PAPER if shade == BLUE else NAVY)
        ax.text(x, 5.05, keep, ha="center", va="center", fontsize=8.6, color=GOOD, fontweight="bold")
        ax.text(x, 4.15, remove, ha="center", va="center", fontsize=8.6, color=WARN)
        ax.text(x, 3.05, "BEST FOR", ha="center", fontsize=8, color=GREY, fontweight="bold")
        ax.text(x, 2.5, best, ha="center", va="center", fontsize=8.8, color=INK)
        stage_box(ax, (x, 1.35), examples, facecolor=PAPER, edgecolor=NAVY, width=3.6, height=0.62,
                  fontsize=9, bold=True)

    ax.set_title("One blueprint, three variants", fontsize=13, color=NAVY, fontweight="bold", pad=14)
    fig.tight_layout()
    return fig


@figure("decoder_autoregressive_generation")
def decoder_autoregressive_generation() -> plt.Figure:
    """Each predicted word feeds back in to predict the next, one token at a time."""
    fig, ax = plt.subplots(figsize=(12.6, 4.6))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 5.6))
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(7.0, 5.2, 'Prompt: "Once upon a time"', ha="center", fontsize=10.5, color=NAVY,
            fontweight="bold")

    steps = [
        "Once upon a time",
        "Once upon a time there",
        "Once upon a time there was",
        "Once upon a time there was a",
    ]
    for i, text in enumerate(steps):
        y = 4.15 - i * 1.05
        words = text.split()
        new_word = words[-1] if i else None
        stage_box(ax, (5.0, y), " ".join(words[:-1] if new_word else words), facecolor=MIST,
                  width=6.0, height=0.72, fontsize=9.2, bold=False)
        if new_word:
            stage_box(ax, (8.7, y), new_word, facecolor=BLUE, textcolor=PAPER, width=1.4,
                      height=0.72, fontsize=9.5)
        if i < len(steps) - 1:
            arrow(ax, (5.0, y - 0.42), (5.0, y - 0.68), scale=11, color=GOOD)

    ax.text(7.0, 0.35,
            "Each predicted word is fed back in to predict the next -- one token at a time.",
            ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.set_title("Decoder-only: autoregressive generation", fontsize=12.5, color=NAVY,
                 fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("sentence_transformer_twin_network")
def sentence_transformer_twin_network() -> plt.Figure:
    """Same encoder, shared weights, two sentences pooled and compared by cosine similarity."""
    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    blank_axes(ax, xlim=(0, 12), ylim=(0, 6.4))
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, (sentence, x) in enumerate([('"I love dogs"', 3.0), ('"I adore puppies"', 9.0)]):
        ax.text(x, 6.0, sentence, ha="center", fontsize=10.2, color=NAVY, fontweight="bold")
        arrow(ax, (x, 5.65), (x, 5.0), scale=12)
        stage_box(ax, (x, 4.55), "Same Encoder\n(shared weights)", facecolor=SKY, width=3.4,
                  height=0.85, fontsize=9)
        arrow(ax, (x, 4.1), (x, 3.45), scale=12)
        stage_box(ax, (x, 3.0), "Pooled Vector", facecolor=MIST, width=2.6, height=0.65, fontsize=9)
        arrow(ax, (x, 2.65), (6.0, 1.75), scale=11, color=GREY, linewidth=1.2)

    stage_box(ax, (6.0, 1.25), "Cosine Similarity: 0.87", facecolor=BLUE, textcolor=PAPER,
              width=4.0, height=0.75, fontsize=10.5)
    ax.text(6.0, 0.4, "High similarity -- the two sentences mean nearly the same thing.",
            ha="center", fontsize=9.6, color=NAVY, fontweight="bold")
    ax.set_title("Sentence Transformers: a twin-network design",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("transformer_decision_map")
def transformer_decision_map() -> plt.Figure:
    """Which variant to reach for, given the task."""
    fig, ax = plt.subplots(figsize=(12.6, 4.4))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 5.4))
    for spine in ax.spines.values():
        spine.set_visible(False)

    rows = [
        ("Classify or label text", "Encoder-only", MIST),
        ("Compare meaning / search", "Sentence Transformer", SKY),
        ("Generate open-ended text", "Decoder-only", SKY),
        ("Transform one sequence into another", "Encoder-Decoder", BLUE),
    ]
    y0, dy = 4.5, 1.05
    for i, (task, variant, shade) in enumerate(rows):
        y = y0 - i * dy
        stage_box(ax, (3.6, y), task, facecolor=PAPER, edgecolor=GREY, width=6.6, height=0.78,
                  fontsize=9.4, bold=False)
        arrow(ax, (7.0, y), (8.2, y), scale=13)
        stage_box(ax, (10.8, y), variant, facecolor=shade, width=4.6, height=0.78, fontsize=9.6,
                  textcolor=PAPER if shade == BLUE else NAVY)

    ax.set_title("Decision map: which variant should you reach for?",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig
