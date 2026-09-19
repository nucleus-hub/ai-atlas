"""Figures for Note 07 -- transformers and self-attention."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .core import BLUE, GOOD, GREY, INK, LINE, MIST, NAVY, PAPER, SKY, WARN, arrow, blank_axes, figure, stage_box

WORDS = ["I", "love", "AI"]


def _matrix(ax, top_left, rows, data, col_labels=None, row_labels=None, cell=(1.05, 0.62),
            fmt="{:g}", highlight=None, shade=SKY):
    """Draw a small labelled matrix; ``highlight`` is a set of (i, j) cells to emphasise."""
    x0, y0 = top_left
    cw, ch = cell
    if col_labels:
        for j, lab in enumerate(col_labels):
            ax.text(x0 + j * cw, y0 + ch * 0.95, lab, ha="center", fontsize=8.2,
                    color=NAVY, fontweight="bold")
    for i, row in enumerate(data):
        y = y0 - i * ch
        if row_labels:
            ax.text(x0 - cw * 0.78, y, row_labels[i], ha="right", va="center",
                    fontsize=8.4, color=NAVY, fontweight="bold")
        for j, v in enumerate(row):
            hot = highlight is not None and (i, j) in highlight
            stage_box(ax, (x0 + j * cw, y), fmt.format(v),
                      facecolor=shade if hot else PAPER,
                      edgecolor=NAVY if hot else LINE,
                      textcolor=PAPER if hot else INK,
                      width=cw * 0.9, height=ch * 0.86, fontsize=8.4, bold=hot)
    _ = rows


@figure("attention_library_analogy")
def attention_library_analogy() -> plt.Figure:
    """Query, Key and Value explained as a library search."""
    fig, ax = plt.subplots(figsize=(12.8, 4.2))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 5.0))
    for spine in ax.spines.values():
        spine.set_visible(False)

    roles = [
        ("Q", "Query", "What am I looking for?", '"Who is this action about?"', BLUE),
        ("K", "Key", "What labels does each book have?", "Each word's searchable tag", SKY),
        ("V", "Value", "What's actually inside the book?", "Each word's real content", MIST),
    ]
    for i, (letter, name, question, detail, shade) in enumerate(roles):
        x = 2.5 + i * 4.5
        stage_box(ax, (x, 3.85), f"{letter}   {name}", facecolor=shade, width=3.9, height=0.9,
                  fontsize=11, textcolor=PAPER if shade in (BLUE,) else NAVY)
        ax.text(x, 2.85, question, ha="center", fontsize=9.2, color=INK, fontweight="bold")
        ax.text(x, 2.25, detail, ha="center", fontsize=8.6, color=GREY)

    ax.text(7.0, 1.1,
            "Compare each word's Query against every word's Key to get relevance scores,",
            ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.text(7.0, 0.55, "then blend the Values together, weighted by those scores.",
            ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.set_title("Self-attention: every word plays three roles at once",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("attention_weights_it")
def attention_weights_it() -> plt.Figure:
    """Where the word 'it' looks in a longer sentence."""
    fig, ax = plt.subplots(figsize=(12.8, 4.2))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 5.0))
    for spine in ax.spines.values():
        spine.set_visible(False)

    words = ["The", "cat", "sat", "on", "the", "mat", "because", "was", "tired"]
    pct = [3, 62, 5, 2, 2, 9, 4, 3, 10]

    xs = np.linspace(1.3, 12.7, len(words))
    for x, w, p in zip(xs, words, pct):
        h = 0.25 + (p / 100) * 2.9
        top = p == max(pct)
        stage_box(ax, (x, 1.35 + h / 2), "", facecolor=BLUE if top else MIST,
                  width=1.12, height=h, fontsize=8)
        ax.text(x, 1.35 + h + 0.28, f"{p}%", ha="center", fontsize=9,
                color=BLUE if top else GREY, fontweight="bold")
        ax.text(x, 0.95, w, ha="center", fontsize=9, color=NAVY if top else INK,
                fontweight="bold" if top else "normal")

    ax.text(7.0, 4.62, 'Sentence: "The cat sat on the mat because it was tired"  --  focus word: "it"',
            ha="center", fontsize=10.2, color=NAVY, fontweight="bold")
    ax.text(7.0, 0.32,
            'The model gives "cat" 62% -- correctly resolving what "it" refers to, purely from context.',
            ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    fig.tight_layout()
    return fig


@figure("attention_numeric_walkthrough")
def attention_numeric_walkthrough() -> plt.Figure:
    """The full six-step numeric example for the sentence 'I love AI'."""
    X = np.array([[1, 0, 1, 0], [0, 2, 0, 2], [1, 1, 1, 1]])
    WQ = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
    WK = np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 0]])
    WV = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1]])
    Q, K, V = X @ WQ, X @ WK, X @ WV
    scores = Q @ K.T
    scaled = scores / np.sqrt(3)
    exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    weights = exp / exp.sum(axis=1, keepdims=True)
    out = weights @ V

    fig, ax = plt.subplots(figsize=(13.4, 8.2))
    blank_axes(ax, xlim=(0, 15), ylim=(0, 10.4))
    for spine in ax.spines.values():
        spine.set_visible(False)

    def label(x, y, text):
        ax.text(x, y, text, ha="left", fontsize=9.6, color=NAVY, fontweight="bold")

    label(0.5, 9.9, "Step 0 -- embeddings X  (3 x 4)")
    _matrix(ax, (1.7, 9.15), 3, X, None, WORDS)

    label(6.6, 9.9, "Q = X.WQ")
    _matrix(ax, (7.7, 9.15), 3, Q, None, WORDS)

    label(10.8, 9.9, "K = X.WK")
    _matrix(ax, (11.9, 9.15), 3, K, None, WORDS)

    label(0.5, 7.0, "V = X.WV")
    _matrix(ax, (1.7, 6.25), 3, V, None, WORDS)

    label(5.3, 7.0, "Step 1 -- scores  Q.K'")
    _matrix(ax, (6.9, 6.25), 3, scores, WORDS, WORDS, highlight={(2, 2)})

    label(10.6, 7.0, "Step 2 -- scaled  / sqrt(3)")
    _matrix(ax, (12.0, 6.25), 3, scaled, None, None, fmt="{:.2f}")

    label(0.5, 4.0, "Step 3 -- softmax weights (%)")
    _matrix(ax, (2.3, 3.25), 3, weights * 100, WORDS, WORDS, fmt="{:.0f}",
            highlight={(0, 2), (1, 2), (2, 2)})

    label(6.9, 4.0, "Step 4 -- output = weights . V")
    _matrix(ax, (8.4, 3.25), 3, out, None, WORDS, fmt="{:.2f}", shade=MIST)

    ax.text(0.5, 1.55, 'Worked cell:  "AI" attending to "AI"', fontsize=9.6, color=NAVY,
            fontweight="bold")
    ax.text(0.5, 1.0, "Q_AI = [2, 2, 3]   K_AI = [3, 2, 1]", fontsize=9, color=INK,
            family="monospace")
    ax.text(0.5, 0.52, "score = (2x3) + (2x2) + (3x1) = 6 + 4 + 3 = 13", fontsize=9,
            color=GOOD, family="monospace", fontweight="bold")

    ax.text(8.6, 1.15,
            "Every row of the weight matrix sums to 1.\nEach output row is a context-aware vector\nfor that word -- this is what flows onward.",
            ha="left", fontsize=9.2, color=INK)

    ax.set_title('Self-attention step by step  --  "I love AI"',
                 fontsize=13, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("attention_pipeline_stages")
def attention_pipeline_stages() -> plt.Figure:
    """The six-stage attention computation, start to finish."""
    fig, ax = plt.subplots(figsize=(13.4, 3.4))
    blank_axes(ax, xlim=(0, 15.4), ylim=(0, 3.6))
    for spine in ax.spines.values():
        spine.set_visible(False)

    stages = [
        ("1  Embeddings", "X: 3 x 4", MIST),
        ("2  Project", "Q, K, V: 3 x 3", SKY),
        ("3  Score", "Q.K': 3 x 3", SKY),
        ("4  Scale", "/ sqrt(dk)", MIST),
        ("5  Softmax", "rows sum to 1", SKY),
        ("6  Weighted Sum", "x V -> output", BLUE),
    ]
    width, gap = 2.2, 0.35
    for i, (title, detail, shade) in enumerate(stages):
        x = 1.4 + i * (width + gap)
        stage_box(ax, (x, 2.2), title, facecolor=shade, width=width, height=0.82, fontsize=9.2,
                  textcolor=PAPER if shade == BLUE else NAVY)
        ax.text(x, 1.45, detail, ha="center", fontsize=8.3, color=GREY)
        if i:
            arrow(ax, (x - width / 2 - gap - 0.02, 2.2), (x - width / 2 + 0.02, 2.2), scale=11)

    ax.text(7.7, 0.55,
            "This exact sequence runs inside every attention head, in every layer, for every word, simultaneously.",
            ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.set_title("The attention pipeline", fontsize=12.5, color=NAVY, fontweight="bold", pad=10)
    fig.tight_layout()
    return fig
