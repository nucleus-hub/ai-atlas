"""Figures for Note 03 -- text preprocessing."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, MIST, NAVY, PAPER, SKY, WARN, arrow, blank_axes, figure, stage_box

STEPS = [
    ("Sentence\nSegmentation", '"Hello world. How are you?"\n-> ["Hello world.", "How are you?"]'),
    ("Tokenization", '"I love NLP!"\n-> ["I", "love", "NLP", "!"]'),
    ("Noise Removal\n& Regex", '"Visit http://nlp.ai !!!"\n-> "Visit"'),
    ("Lowercasing", '"Apple" and "apple"\n-> same token "apple"'),
    ("Stopword\nRemoval", '"the cat sat on the mat"\n-> "cat sat mat"'),
    ("Stemming /\nLemmatization", '"running", "runs"\n-> "run"'),
]


@figure("preprocessing_steps")
def preprocessing_steps() -> plt.Figure:
    """The preprocessing chores, each with a worked before/after."""
    fig, ax = plt.subplots(figsize=(13.0, 6.2))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 7.2))
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, (name, example) in enumerate(STEPS):
        y = 6.2 - i * 1.0
        shade = MIST if i % 2 == 0 else SKY
        stage_box(ax, (2.3, y), name, facecolor=shade, width=3.2, height=0.82, fontsize=9.5)
        ax.text(4.35, y, example, ha="left", va="center", fontsize=8.6, color=INK, family="monospace")
        if i:
            arrow(ax, (2.3, y + 0.59), (2.3, y + 0.43), scale=10, linewidth=1.3)

    ax.text(7.0, 0.35,
            "Before cooking you wash, peel and chop -- preprocessing does the same for text.",
            ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_title("Text preprocessing: cleaning raw text before any model sees it",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("stemming_vs_lemmatization")
def stemming_vs_lemmatization() -> plt.Figure:
    """Rough-and-fast suffix stripping versus dictionary-backed lemmatization."""
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 6.2))
    for spine in ax.spines.values():
        spine.set_visible(False)

    stage_box(ax, (3.6, 5.4), "Stemming  (Rough & Fast)\nsuffix-stripping rules, no dictionary",
              facecolor=PAPER, edgecolor=WARN, width=6.2, height=0.95, fontsize=10)
    stage_box(ax, (10.4, 5.4), "Lemmatization  (Smart & Accurate)\ndictionary + grammar rules",
              facecolor=BLUE, edgecolor=NAVY, textcolor=PAPER, width=6.2, height=0.95, fontsize=10)

    rows = [
        ("running", "run", "ok", "run"),
        ("studies", "studi", "not a word", "study"),
        ("better", "better", "no change", "good"),
        ("historical", "histor", "not a word", "historical"),
    ]
    for i, (word, stem, note, lemma) in enumerate(rows):
        y = 4.35 - i * 0.72
        stem_ok = note == "ok"
        ax.text(1.1, y, word, ha="left", va="center", fontsize=9.5, color=INK, family="monospace")
        ax.text(4.3, y, f"-> {stem}", ha="left", va="center", fontsize=9.5,
                color=GOOD if stem_ok else WARN, family="monospace", fontweight="bold")
        ax.text(6.35, y, note, ha="left", va="center",
                fontsize=8.2, color=GOOD if stem_ok else WARN)

        ax.text(8.6, y, word, ha="left", va="center", fontsize=9.5, color=INK, family="monospace")
        ax.text(11.4, y, f"-> {lemma}", ha="left", va="center", fontsize=9.5,
                color=GOOD, family="monospace", fontweight="bold")

    ax.plot([7.5, 7.5], [0.9, 4.9], color=GREY, lw=1.2, linestyle="--")
    ax.text(3.6, 0.95, "Best for: search indexing, large corpora", ha="center", fontsize=8.8, color=GREY)
    ax.text(10.4, 0.95, "Best for: sentiment, QA, chatbots", ha="center", fontsize=8.8, color=GREY)

    ax.text(7.0, 0.3, "Stemming may return nonsense; lemmatization always returns a real dictionary word.",
            ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_title("Stemming vs Lemmatization", fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig
