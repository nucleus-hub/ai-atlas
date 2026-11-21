"""Figures for Note 04 -- text representation (BoW, N-grams, TF-IDF, embeddings)."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, LINE, MIST, NAVY, PAPER, SKY, WARN, blank_axes, figure, stage_box

VOCAB = ["cat", "sat", "on", "mat", "ate", "the", "rat"]
DOC1 = [1, 1, 1, 1, 0, 0, 0]
DOC2 = [1, 0, 0, 0, 1, 1, 1]


@figure("bow_vectors")
def bow_vectors() -> plt.Figure:
    """Two documents counted into vectors over a shared vocabulary."""
    fig, ax = plt.subplots(figsize=(12.6, 4.8))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 5.4))
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(0.5, 5.0, 'Doc 1:  "cat sat on mat"', ha="left", fontsize=10, color=INK, fontweight="bold")
    ax.text(0.5, 4.55, 'Doc 2:  "cat ate the rat"', ha="left", fontsize=10, color=INK, fontweight="bold")

    x0, cell = 3.4, 1.35
    for j, word in enumerate(VOCAB):
        ax.text(x0 + j * cell, 3.75, word, ha="center", fontsize=9.5, color=NAVY, fontweight="bold")

    for i, (label, vec) in enumerate([("Doc1", DOC1), ("Doc2", DOC2)]):
        y = 2.95 - i * 0.95
        ax.text(2.55, y, label, ha="right", va="center", fontsize=10, color=INK, fontweight="bold")
        for j, v in enumerate(vec):
            stage_box(ax, (x0 + j * cell, y), str(v),
                      facecolor=SKY if v else PAPER, edgecolor=NAVY if v else LINE,
                      textcolor=PAPER if v else GREY, width=1.0, height=0.7, fontsize=10)

    ax.text(7.0, 0.95, "Vocabulary = every unique word.  Each cell counts occurrences in that document.",
            ha="center", fontsize=9.5, color=INK)
    ax.text(7.0, 0.4, "Word order is discarded entirely -- the document is a literal bag of words.",
            ha="center", fontsize=10, color=WARN, fontweight="bold")
    ax.set_title("Bag of Words: counting words into vectors",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("ngrams_sliding")
def ngrams_sliding() -> plt.Figure:
    """Unigrams, bigrams and trigrams over one sentence."""
    fig, ax = plt.subplots(figsize=(12.6, 5.6))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 7.0))
    for spine in ax.spines.values():
        spine.set_visible(False)

    words = ["I", "love", "NLP", "very", "much"]
    xs = [3.0, 5.0, 7.0, 9.0, 11.0]
    ax.text(7.0, 6.6, 'Sentence:  "I love NLP very much"', ha="center", fontsize=10.5,
            color=NAVY, fontweight="bold")
    for x, w in zip(xs, words):
        ax.text(x, 6.05, w, ha="center", fontsize=9.5, color=GREY, fontweight="bold")

    # Each level stacks its windows on staggered sub-rows so overlaps stay legible.
    rows = [
        ("Unigrams\n(1-gram)", 1, MIST, 5.25, 1),
        ("Bigrams\n(2-gram)", 2, SKY, 4.25, 2),
        ("Trigrams\n(3-gram)", 3, BLUE, 2.35, 2),
    ]
    for label, n, shade, y_top, lanes in rows:
        count = len(words) - n + 1
        ax.text(1.2, y_top - 0.32 * (lanes - 1), label, ha="center", va="center",
                fontsize=9.5, color=NAVY, fontweight="bold")
        for start in range(count):
            left = xs[start] - 0.92
            right = xs[start + n - 1] + 0.92
            y = y_top - (start % lanes) * 0.78
            stage_box(ax, ((left + right) / 2, y), " ".join(words[start:start + n]),
                      facecolor=shade, width=right - left, height=0.62, fontsize=8.8,
                      textcolor=PAPER if shade == BLUE else NAVY)

    ax.text(7.0, 0.3,
            "N=1 broad coverage  ·  N=2 phrases  ·  N=3+ specific patterns.  Higher N = more context, sparser data.",
            ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
    ax.set_title("N-grams: grouping adjacent words to keep local order",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("tfidf_worked_example")
def tfidf_worked_example() -> plt.Figure:
    """TF, IDF and their product for five words in a film review."""
    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 5.8))
    for spine in ax.spines.values():
        spine.set_visible(False)

    rows = [
        ("the", 0.12, 0.01, 0.001, "Low", WARN),
        ("movie", 0.05, 1.20, 0.060, "Medium", GREY),
        ("cinematography", 0.02, 3.80, 0.076, "High", GOOD),
        ("acting", 0.04, 1.50, 0.060, "Medium", GREY),
        ("superb", 0.01, 2.90, 0.029, "Medium", GREY),
    ]

    headers = [("Word", 1.0), ("TF", 5.0), ("IDF", 7.2), ("TF-IDF", 9.6), ("Importance", 12.0)]
    for text, x in headers:
        ax.text(x, 4.8, text, ha="left" if x < 2 else "center", fontsize=10,
                color=NAVY, fontweight="bold")
    ax.plot([0.8, 13.2], [4.5, 4.5], color=NAVY, lw=1.3)

    for i, (word, tf, idf, score, importance, colour) in enumerate(rows):
        y = 4.0 - i * 0.66
        ax.text(1.0, y, word, ha="left", va="center", fontsize=9.5, color=INK, family="monospace")
        ax.text(5.0, y, f"{tf:.2f}", ha="center", va="center", fontsize=9.5, color=INK)
        ax.text(7.2, y, f"{idf:.2f}", ha="center", va="center", fontsize=9.5, color=INK)
        ax.text(9.6, y, f"{score:.3f}", ha="center", va="center", fontsize=9.5,
                color=colour, fontweight="bold")
        ax.text(12.0, y, importance, ha="center", va="center", fontsize=9.2, color=colour,
                fontweight="bold")

    ax.text(7.0, 0.42,
            '"the" has high TF but near-zero IDF -> score ~ 0.   "cinematography" is rare -> highest score.',
            ha="center", fontsize=9.8, color=NAVY, fontweight="bold")
    ax.set_title("TF-IDF finds the distinctive words  (corpus: 1000 movie reviews)",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("embedding_vector_space")
def embedding_vector_space() -> plt.Figure:
    """Semantic clusters and the king - man + woman analogy."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.0, 4.8))

    blank_axes(left, xlim=(0, 10), ylim=(0, 10))
    left.set_title("Similar words cluster together", fontsize=11.5, color=NAVY, fontweight="bold", pad=8)
    clusters = [
        ([(2.0, 8.0, "king"), (2.9, 8.7, "queen"), (1.6, 7.0, "prince")], BLUE, "royalty"),
        ([(7.4, 7.8, "happy"), (8.3, 7.0, "joyful"), (6.8, 6.6, "glad")], GOOD, "emotion"),
        ([(2.4, 2.4, "cat"), (3.4, 1.8, "dog"), (1.8, 1.4, "horse")], WARN, "animals"),
    ]
    for points, colour, name in clusters:
        for x, y, word in points:
            left.scatter([x], [y], s=70, color=colour, zorder=3)
            left.text(x + 0.25, y + 0.28, word, fontsize=9, color=INK, fontweight="bold")
        cx = sum(p[0] for p in points) / len(points)
        cy = sum(p[1] for p in points) / len(points)
        left.scatter([cx], [cy], s=2600, color=colour, alpha=0.10, zorder=1)
        left.text(cx, cy - 1.35, name, ha="center", fontsize=8.8, color=colour, fontweight="bold")
    left.set_xlabel("dimension 1", fontsize=9, color=GREY)
    left.set_ylabel("dimension 2", fontsize=9, color=GREY)

    blank_axes(right, xlim=(0, 10), ylim=(0, 10))
    right.set_title("Arithmetic works", fontsize=11.5, color=NAVY, fontweight="bold", pad=8)
    pts = {"king": (2.2, 7.6), "man": (2.2, 3.2), "woman": (6.6, 3.2), "queen": (6.6, 7.6)}
    for word, (x, y) in pts.items():
        right.scatter([x], [y], s=90, color=BLUE, zorder=3)
        right.text(x, y + 0.55, word, ha="center", fontsize=10, color=NAVY, fontweight="bold")
    right.annotate("", xy=pts["king"], xytext=pts["man"],
                   arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6, linestyle="--"))
    right.annotate("", xy=pts["queen"], xytext=pts["woman"],
                   arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6, linestyle="--"))
    right.annotate("", xy=pts["woman"], xytext=pts["man"],
                   arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.8))
    right.annotate("", xy=pts["queen"], xytext=pts["king"],
                   arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.8))
    right.text(4.4, 2.6, "same offset", ha="center", fontsize=8.8, color=GOOD, fontweight="bold")
    right.text(4.4, 8.35, "same offset", ha="center", fontsize=8.8, color=GOOD, fontweight="bold")
    right.text(5.0, 0.75, '"king" - "man" + "woman"  ~  "queen"', ha="center", fontsize=10,
               color=NAVY, fontweight="bold")

    fig.tight_layout()
    return fig
