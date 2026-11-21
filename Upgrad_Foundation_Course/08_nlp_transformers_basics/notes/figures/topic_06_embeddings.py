"""Figures for Note 06 -- word embeddings."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, LINE, MIST, NAVY, PAPER, SKY, WARN, arrow, blank_axes, figure, stage_box


@figure("cbow_vs_skipgram")
def cbow_vs_skipgram() -> plt.Figure:
    """The two Word2Vec training directions, side by side."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    context = ["quality", "is", "and", "photos"]

    # --- CBOW: context -> target -----------------------------------------
    blank_axes(left, xlim=(0, 10), ylim=(0, 8))
    for spine in left.spines.values():
        spine.set_visible(False)
    left.set_title("CBOW\ncontext -> target", fontsize=11.5, color=NAVY, fontweight="bold", pad=8)

    xs = [1.4, 3.6, 6.0, 8.2]
    for x, word in zip(xs, context):
        stage_box(left, (x, 6.6), word, facecolor=MIST, width=1.9, height=0.72, fontsize=9)
        arrow(left, (x, 6.2), (5.0, 4.75), scale=11, linewidth=1.2)
    stage_box(left, (5.0, 4.35), "average / hidden layer", facecolor=PAPER, edgecolor=GREY,
              width=5.0, height=0.7, fontsize=9, bold=False)
    arrow(left, (5.0, 3.95), (5.0, 2.85), scale=13)
    stage_box(left, (5.0, 2.4), "excellent", facecolor=BLUE, textcolor=PAPER,
              width=2.6, height=0.8, fontsize=10)
    left.text(5.0, 1.45, "predicted target word", ha="center", fontsize=8.8, color=GREY)
    left.text(5.0, 0.6, 'Given "quality is ___ and photos"', ha="center", fontsize=9,
              color=INK, fontweight="bold")

    # --- Skip-gram: target -> context ------------------------------------
    blank_axes(right, xlim=(0, 10), ylim=(0, 8))
    for spine in right.spines.values():
        spine.set_visible(False)
    right.set_title("Skip-Gram\ntarget -> context", fontsize=11.5, color=NAVY, fontweight="bold", pad=8)

    stage_box(right, (5.0, 6.4), "excellent", facecolor=BLUE, textcolor=PAPER,
              width=2.6, height=0.8, fontsize=10)
    right.text(6.6, 6.4, "input target word", ha="left", va="center", fontsize=8.8, color=GREY)
    for x, word in zip(xs, context):
        arrow(right, (5.0, 6.0), (x, 3.3), scale=11, linewidth=1.2)
        stage_box(right, (x, 2.9), word, facecolor=MIST, width=1.9, height=0.72, fontsize=9)
    right.text(5.0, 0.6, 'Given "excellent", predict each neighbour', ha="center", fontsize=9,
               color=INK, fontweight="bold")

    fig.tight_layout()
    return fig


@figure("glove_cooccurrence")
def glove_cooccurrence() -> plt.Figure:
    """The toy co-occurrence matrix GloVe factorizes."""
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    blank_axes(ax, xlim=(0, 11), ylim=(0, 6.0))
    for spine in ax.spines.values():
        spine.set_visible(False)

    words = ["excellent", "great", "quality", "bland"]
    counts = [
        [None, 6, 8, 0],
        [6, None, 5, 1],
        [8, 5, None, 1],
        [0, 1, 1, None],
    ]

    x0, y0, cell = 3.5, 4.0, 1.75
    for j, w in enumerate(words):
        ax.text(x0 + j * cell, y0 + 0.85, w, ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
    for i, w in enumerate(words):
        y = y0 - i * 0.92
        ax.text(x0 - 1.15, y, w, ha="right", va="center", fontsize=9.5, color=NAVY, fontweight="bold")
        for j, v in enumerate(counts[i]):
            if v is None:
                stage_box(ax, (x0 + j * cell, y), "--", facecolor=PAPER, edgecolor=LINE,
                          textcolor=GREY, width=1.35, height=0.72, fontsize=10, bold=False)
            else:
                strong = v >= 5
                stage_box(ax, (x0 + j * cell, y), str(v),
                          facecolor=SKY if strong else PAPER,
                          edgecolor=NAVY if strong else LINE,
                          textcolor=PAPER if strong else GREY,
                          width=1.35, height=0.72, fontsize=10)

    ax.text(5.5, 0.45,
            "High counts = words that keep company.  GloVe factorizes this whole matrix into dense vectors.",
            ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.set_title("GloVe: global word-word co-occurrence counts",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("fasttext_subwords")
def fasttext_subwords() -> plt.Figure:
    """Character n-grams let FastText handle unseen and misspelled words."""
    fig, ax = plt.subplots(figsize=(12.6, 4.4))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 5.2))
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(7.0, 4.75, '"excellent"  ->  character 3-grams', ha="center", fontsize=11,
            color=NAVY, fontweight="bold")

    grams = ["exc", "xce", "cel", "ell", "lle", "len", "ent"]
    for i, g in enumerate(grams):
        stage_box(ax, (1.5 + i * 1.72, 3.55), g, facecolor=SKY, width=1.5, height=0.7,
                  fontsize=9.5, textcolor=PAPER)

    ax.text(7.0, 2.6, 'A misspelling shares almost all of them:', ha="center", fontsize=9.8,
            color=INK)
    mis = ["exc", "xce", "cel", "ele", "len", "ent"]
    shared = {"exc", "xce", "cel", "len", "ent"}
    for i, g in enumerate(mis):
        ok = g in shared
        stage_box(ax, (2.4 + i * 1.72, 1.6), g,
                  facecolor=SKY if ok else PAPER, edgecolor=NAVY if ok else WARN,
                  textcolor=PAPER if ok else WARN, width=1.5, height=0.7, fontsize=9.5)
    ax.text(12.6, 1.6, '"excelent"', ha="left", va="center", fontsize=9.5, color=WARN,
            fontweight="bold")

    ax.text(7.0, 0.45,
            "Five of six n-grams match, so the typo still lands next to the correct word.",
            ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.set_title("FastText: words as bags of character n-grams",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("static_vs_contextual")
def static_vs_contextual() -> plt.Figure:
    """One vector per word, versus a vector shaped by the sentence."""
    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 5.6))
    for spine in ax.spines.values():
        spine.set_visible(False)

    stage_box(ax, (3.4, 5.0), "Static embeddings\none vector per word", facecolor=PAPER,
              edgecolor=WARN, width=5.6, height=0.85, fontsize=10)
    stage_box(ax, (10.4, 5.0), "Contextual embeddings\nthe sentence decides", facecolor=BLUE,
              edgecolor=NAVY, textcolor=PAPER, width=5.6, height=0.85, fontsize=10)

    ax.text(3.4, 3.95, '"the pasta arrived cold"', ha="center", fontsize=9.2, color=INK)
    ax.text(3.4, 3.35, '"staff felt cold and distant"', ha="center", fontsize=9.2, color=INK)
    arrow(ax, (3.4, 3.0), (3.4, 2.35), scale=12, color=WARN)
    stage_box(ax, (3.4, 1.95), "cold\n[0.4, -0.2, 0.7]", facecolor=MIST, width=3.0, height=0.95,
              fontsize=9.2)
    ax.text(3.4, 1.05, "one shared vector -- meaning collapsed", ha="center", fontsize=8.8,
            color=WARN, fontweight="bold")

    ax.text(10.4, 3.95, '"the pasta arrived cold"', ha="center", fontsize=9.2, color=INK)
    arrow(ax, (8.9, 3.7), (8.5, 2.45), scale=11, color=GOOD, linewidth=1.4)
    ax.text(10.4, 3.35, '"staff felt cold and distant"', ha="center", fontsize=9.2, color=INK)
    arrow(ax, (11.9, 3.15), (12.3, 2.45), scale=11, color=GOOD, linewidth=1.4)
    stage_box(ax, (8.5, 1.95), "cold\n[0.9, 0.1, -0.3]", facecolor=SKY, width=2.8, height=0.95,
              fontsize=9.2)
    stage_box(ax, (12.3, 1.95), "cold\n[-0.2, 0.8, 0.5]", facecolor=SKY, width=2.8, height=0.95,
              fontsize=9.2)
    ax.text(10.4, 1.05, "two different vectors -- temperature vs personality", ha="center",
            fontsize=8.8, color=GOOD, fontweight="bold")

    ax.text(7.0, 0.35,
            "Static embeddings are looked up once per word; contextual ones are computed fresh per sentence.",
            ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.set_title("The limitation static embeddings cannot escape",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("embeddings_journey")
def embeddings_journey() -> plt.Figure:
    """The section recap: one-hot through to the contextual bridge."""
    fig, ax = plt.subplots(figsize=(13.2, 3.4))
    blank_axes(ax, xlim=(0, 16), ylim=(0, 3.6))
    for spine in ax.spines.values():
        spine.set_visible(False)

    stages = [
        ("One-Hot\nEncoding", MIST),
        ("Word2Vec\n(CBOW / Skip-gram)", SKY),
        ("GloVe &\nFastText", SKY),
        ("Static\nLimitations", MIST),
        ("Contextual\nBridge", BLUE),
    ]
    width, gap = 2.6, 0.45
    for i, (label, shade) in enumerate(stages):
        x = 1.6 + i * (width + gap)
        stage_box(ax, (x, 2.15), label, facecolor=shade, width=width, height=1.05, fontsize=9.3,
                  textcolor=PAPER if shade == BLUE else NAVY)
        if i:
            arrow(ax, (x - width / 2 - gap - 0.02, 2.15), (x - width / 2 + 0.02, 2.15), scale=12)

    ax.text(8.0, 0.7,
            "Next: the architecture that made one-vector-per-word representations obsolete -- Transformers.",
            ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_title("The embeddings journey", fontsize=12.5, color=NAVY, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig
