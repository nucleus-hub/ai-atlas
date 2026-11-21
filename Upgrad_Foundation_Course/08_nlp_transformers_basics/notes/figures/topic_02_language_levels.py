"""Figures for Note 02 -- the four levels of language analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, MIST, NAVY, PAPER, SKY, WARN, arrow, blank_axes, figure, stage_box

LEVELS = [
    ("Pragmatic", "Intent", '"Can you pass the salt?" -> a polite REQUEST', BLUE),
    ("Semantic", "Meaning", "Agent = you, Action = pass, Patient = salt", SKY),
    ("Syntactic", "Structure", 'interrogative; "you" is subject of "pass"', SKY),
    ("Lexical", "Words", "can/AUX  you/PRON  pass/VERB  salt/NOUN", MIST),
]


@figure("language_analysis_levels")
def language_analysis_levels() -> plt.Figure:
    """The four-level stack, each layer resting on the one below it."""
    fig, ax = plt.subplots(figsize=(12.8, 5.6))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 6.4))
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, (name, tag, example, shade) in enumerate(LEVELS):
        y = 1.15 + (len(LEVELS) - 1 - i) * 1.25
        stage_box(ax, (3.1, y), f"{name}\n= {tag}", facecolor=shade, width=3.4, height=1.0, fontsize=10.5,
                  textcolor=PAPER if shade == BLUE else NAVY)
        ax.text(5.2, y, example, ha="left", va="center", fontsize=9.2, color=INK, family="monospace")

    arrow(ax, (0.85, 1.15), (0.85, 4.9), scale=16, linewidth=2.0)
    ax.text(0.45, 3.0, "each level builds on the one below", rotation=90, ha="center", va="center",
            fontsize=9, color=GOOD, fontweight="bold")

    ax.text(7.0, 0.35, "Lexical = Words  ->  Syntactic = Structure  ->  Semantic = Meaning  ->  Pragmatic = Intent",
            ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_title("The four levels of language analysis", fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("dependency_vs_parse_tree")
def dependency_vs_parse_tree() -> plt.Figure:
    """'The dog chased the cat' as a dependency graph and as a parse tree."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 4.8))

    # --- dependency parse -------------------------------------------------
    blank_axes(left, xlim=(0, 10), ylim=(0, 4.6))
    for spine in left.spines.values():
        spine.set_visible(False)
    left.set_title("Dependency Parsing\nword-to-word relations", fontsize=11.5, color=NAVY, fontweight="bold", pad=8)

    words = ["The", "dog", "chased", "the", "cat"]
    xs = [1.0, 2.8, 5.0, 7.2, 8.9]
    for x, w in zip(xs, words):
        left.text(x, 1.0, w, ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold")

    left.text(5.0, 4.05, "ROOT", ha="center", fontsize=10, color=WARN, fontweight="bold")
    arrow(left, (5.0, 3.78), (5.0, 1.45), color=WARN, scale=13, linewidth=1.6)

    arcs = [(5.0, 2.8, "nsubj"), (5.0, 8.9, "dobj"), (2.8, 1.0, "det"), (8.9, 7.2, "det")]
    for x0, x1, label in arcs:
        mid = (x0 + x1) / 2
        top = 1.45 + 0.225 * abs(x1 - x0) + 0.16
        left.annotate("", xy=(x1, 1.45), xytext=(x0, 1.45),
                      arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5,
                                      connectionstyle=f"arc3,rad={0.45 if x1 > x0 else -0.45}"))
        left.text(mid, top, label, ha="center", fontsize=8.6, color=BLUE, fontweight="bold",
                  bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.2))

    left.text(5.0, 0.25, "Arrows flow head -> dependent; all roads lead to ROOT.",
              ha="center", fontsize=8.8, color=GREY)

    # --- parse tree -------------------------------------------------------
    blank_axes(right, xlim=(0, 10), ylim=(0, 6))
    for spine in right.spines.values():
        spine.set_visible(False)
    right.set_title("Parse Tree\nnested phrase structure", fontsize=11.5, color=NAVY, fontweight="bold", pad=8)

    nodes = {
        "S": (5.0, 5.3), "NP1": (2.2, 4.1), "VP": (6.6, 4.1), "NP2": (7.9, 2.9),
        "DT1": (1.2, 2.9), "NN1": (3.2, 2.9), "VBD": (5.3, 2.9),
        "DT2": (7.0, 1.7), "NN2": (8.8, 1.7),
    }
    labels = {"S": "S", "NP1": "NP", "VP": "VP", "NP2": "NP", "DT1": "DT", "NN1": "NN",
              "VBD": "VBD", "DT2": "DT", "NN2": "NN"}
    edges = [("S", "NP1"), ("S", "VP"), ("NP1", "DT1"), ("NP1", "NN1"),
             ("VP", "VBD"), ("VP", "NP2"), ("NP2", "DT2"), ("NP2", "NN2")]
    for a, b in edges:
        right.plot([nodes[a][0], nodes[b][0]], [nodes[a][1] - 0.22, nodes[b][1] + 0.22],
                   color=GREY, lw=1.3, zorder=1)
    for key, (x, y) in nodes.items():
        right.text(x, y, labels[key], ha="center", va="center", fontsize=9.5, color=NAVY, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.25", facecolor=MIST, edgecolor=NAVY, lw=1.1), zorder=3)

    leaves = [("The", 1.2), ("dog", 3.2), ("chased", 5.3), ("the", 7.0), ("cat", 8.8)]
    for text, x in leaves:
        right.text(x, 0.75, text, ha="center", fontsize=9.5, color=INK, fontweight="bold")

    right.text(5.0, 0.2, "Phrases nest recursively -- the VP holds its own NP.",
               ha="center", fontsize=8.8, color=GREY)

    fig.tight_layout()
    return fig


@figure("chunking_flat_phrases")
def chunking_flat_phrases() -> plt.Figure:
    """Flat, non-overlapping chunks versus recursive parse-tree depth."""
    fig, ax = plt.subplots(figsize=(12.8, 4.4))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 5))
    for spine in ax.spines.values():
        spine.set_visible(False)

    rows = [
        (3.5, [("NP", "The big red apple", 3.5), ("VP", "sat", 1.5), ("PP", "on the table", 3.0)]),
        (1.7, [("NP", "Apple Inc.", 2.3), ("VP", "launched", 2.0), ("NP", "a new device", 2.8), ("PP", "in March", 2.2)]),
    ]
    for y, chunks in rows:
        x = 0.7
        for tag, text, width in chunks:
            stage_box(ax, (x + width / 2, y), f"{tag}\n{text}", facecolor=MIST if tag == "NP" else SKY,
                      width=width, height=0.95, fontsize=9)
            x += width + 0.35

    ax.text(7.0, 0.55,
            "Chunks are flat and non-overlapping -- fast regex/ML over POS tags, no recursion.",
            ha="center", fontsize=10, color=NAVY, fontweight="bold")
    ax.set_title("Chunking: flat phrase extraction", fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig
