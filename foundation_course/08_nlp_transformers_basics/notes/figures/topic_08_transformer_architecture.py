"""Figures for Note 08 -- the Transformer architecture."""

from __future__ import annotations

import matplotlib.pyplot as plt

from .core import BLUE, GOOD, GREY, INK, LINE, MIST, NAVY, PAPER, SKY, WARN, arrow, blank_axes, figure, stage_box

HEADS = [
    ("Head 1", "Grammar\nsubject / verb links"),
    ("Head 2", "Reference\nwhat \"it\" points to"),
    ("Head 3", "Meaning\ntopic-level relationships"),
    ("Head 4", "Position\nnearby-word patterns"),
]


@figure("multihead_attention_experts")
def multihead_attention_experts() -> plt.Figure:
    """Several attention heads reading the same sentence for different relationships."""
    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 5.6))
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(7.0, 5.2, 'Same sentence, four heads, four different relationships',
            ha="center", fontsize=10.5, color=NAVY, fontweight="bold")

    xs = [2.0, 5.0, 8.0, 11.0]
    shades = [MIST, SKY, SKY, MIST]
    for x, (title, detail), shade in zip(xs, HEADS, shades):
        stage_box(ax, (x, 3.7), title, facecolor=shade, width=2.6, height=0.75, fontsize=10)
        ax.text(x, 2.75, detail, ha="center", va="center", fontsize=8.6, color=INK)
        arrow(ax, (x, 2.25), (7.0, 1.35), scale=11, linewidth=1.2, color=GREY)

    stage_box(ax, (7.0, 0.85), "Concatenate + Combine", facecolor=BLUE, textcolor=PAPER,
              width=4.6, height=0.7, fontsize=10.5)

    ax.set_title("Multi-head attention: many experts, one sentence",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("attention_variants_overview")
def attention_variants_overview() -> plt.Figure:
    """Self, cross, multi-head and flash attention compared."""
    fig, ax = plt.subplots(figsize=(13.0, 5.6))
    blank_axes(ax, xlim=(0, 14), ylim=(0, 9.2))
    for spine in ax.spines.values():
        spine.set_visible(False)

    rows = [
        ("Self-Attention", "WHAT attends to WHAT", "Words attend to other words in the same sequence.\nQ, K, V all come from the same input.",
         "Every encoder layer; decoder's first attention step", MIST),
        ("Cross-Attention", "WHAT attends to WHAT", "Words in one sequence attend to words in another.\nQ from one side, K & V from the other.",
         "Decoder's second attention step (encoder-decoder models)", SKY),
        ("Multi-Head Attention", "HOW MANY views at once", "Several heads run in parallel -- each can be\nself- or cross-attention -- capturing different relationships.",
         "A multiplier applied to self- or cross-attention, everywhere", SKY),
        ("Flash Attention", "HOW it's computed", "The exact same attention math, computed with a\nhardware-efficient algorithm avoiding wasted memory traffic.",
         "Modern training & inference, for speed and longer context", BLUE),
    ]
    y0, dy = 8.35, 2.05
    for i, (name, axis, detail, use, shade) in enumerate(rows):
        y = y0 - i * dy
        stage_box(ax, (1.9, y), name, facecolor=shade, width=3.4, height=1.55, fontsize=9.6,
                  textcolor=PAPER if shade == BLUE else NAVY)
        ax.text(4.1, y + 0.55, axis, ha="left", fontsize=8.4, color=GREY, fontweight="bold")
        ax.text(4.1, y, detail, ha="left", va="center", fontsize=8.8, color=INK)
        ax.text(4.1, y - 0.65, use, ha="left", fontsize=8.0, color=GOOD, fontweight="bold")

    ax.set_title("Attention isn't one thing -- it's a family of variants",
                 fontsize=12.5, color=NAVY, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


@figure("feedforward_residual_layernorm")
def feedforward_residual_layernorm() -> plt.Figure:
    """Per-word feed-forward network plus the residual + layer-norm safety nets."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.2, 4.8))

    blank_axes(left, xlim=(0, 10), ylim=(0, 8))
    for spine in left.spines.values():
        spine.set_visible(False)
    left.set_title("Feed-Forward Network\napplied per word, in parallel", fontsize=11.2,
                    color=NAVY, fontweight="bold", pad=8)
    words = ["cat", "sat", "tired"]
    for i, w in enumerate(words):
        y = 6.2 - i * 2.3
        stage_box(left, (1.4, y), w, facecolor=MIST, width=1.6, height=0.75, fontsize=9.5)
        arrow(left, (2.25, y), (3.55, y), scale=11)
        stage_box(left, (5.2, y), "small NN\n(thinking...)", facecolor=SKY, width=3.0, height=0.9,
                  fontsize=8.8)
        arrow(left, (6.75, y), (8.05, y), scale=11)
        stage_box(left, (8.9, y), "out", facecolor=BLUE, textcolor=PAPER, width=1.4, height=0.75,
                  fontsize=9.5)
    left.text(5.0, 0.6, "Same tiny network, applied to every position independently.",
              ha="center", fontsize=9, color=NAVY, fontweight="bold")

    blank_axes(right, xlim=(0, 10), ylim=(0, 8))
    for spine in right.spines.values():
        spine.set_visible(False)
    right.set_title("Residual + LayerNorm\nsafety nets for deep stacks", fontsize=11.2,
                     color=NAVY, fontweight="bold", pad=8)
    stage_box(right, (5.0, 6.6), "input x", facecolor=MIST, width=2.6, height=0.75, fontsize=9.5)
    arrow(right, (5.0, 6.2), (5.0, 5.1), scale=12)
    stage_box(right, (5.0, 4.65), "Sublayer(x)\nattention or feed-forward", facecolor=SKY,
              width=4.2, height=0.95, fontsize=8.8)
    # residual skip arrow around the sublayer
    right.annotate("", xy=(7.3, 3.55), xytext=(6.6, 6.6),
                   arrowprops=dict(arrowstyle="-", color=GOOD, lw=1.6,
                                    connectionstyle="arc3,rad=-0.45"))
    right.annotate("", xy=(7.3, 3.55), xytext=(7.35, 3.6),
                   arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.6))
    right.text(8.35, 5.2, "residual\nskip", ha="center", fontsize=8.4, color=GOOD, fontweight="bold")
    arrow(right, (5.0, 4.2), (5.0, 3.9), scale=12)
    stage_box(right, (5.5, 3.55), "x + Sublayer(x)", facecolor=PAPER, edgecolor=GOOD,
              width=3.6, height=0.7, fontsize=9)
    arrow(right, (5.5, 3.2), (5.5, 2.55), scale=12)
    stage_box(right, (5.5, 2.1), "LayerNorm(...)", facecolor=BLUE, textcolor=PAPER,
              width=3.6, height=0.7, fontsize=9.2)
    right.text(5.0, 0.7, "Add & Normalize -- the pattern repeated after every sublayer.",
               ha="center", fontsize=9, color=NAVY, fontweight="bold")

    fig.tight_layout()
    return fig


@figure("transformer_encoder_decoder_stack")
def transformer_encoder_decoder_stack() -> plt.Figure:
    """The assembled encoder and decoder stacks, side by side."""
    fig, (enc, dec) = plt.subplots(1, 2, figsize=(12.6, 7.2))

    enc_layers = ["Multi-Head Self-Attention", "Add & Normalize", "Feed-Forward Network", "Add & Normalize"]
    dec_layers = ["Masked Self-Attention", "Add & Normalize", "Cross-Attention -> Encoder",
                  "Add & Normalize", "Feed-Forward Network", "Add & Normalize"]

    for panel, layers, title, shade_main, inp in [
        (enc, enc_layers, "ENCODER", SKY, "Input embeddings + positions"),
        (dec, dec_layers, "DECODER", MIST, "Output so far (shifted right)"),
    ]:
        blank_axes(panel, xlim=(0, 10), ylim=(0, len(layers) * 1.35 + 4.0))
        for spine in panel.spines.values():
            spine.set_visible(False)
        top = len(layers) * 1.35 + 1.6
        panel.text(5.0, top + 1.9, title, ha="center", fontsize=13, color=NAVY, fontweight="bold")

        # Data flows bottom -> top: input feeds the first sublayer; the final
        # Add & Normalize output is what stacks x N and continues onward.
        bottom_y = top - (len(layers) - 1) * 1.35
        panel.text(5.0, bottom_y - 1.15, inp, ha="center", fontsize=9, color=INK)
        arrow(panel, (5.0, bottom_y - 0.85), (5.0, bottom_y - 0.5), scale=13, color=GOOD)

        y = top
        for i, name in enumerate(layers):
            shade = shade_main if "Add" not in name else PAPER
            edge = NAVY if "Add" not in name else GREY
            stage_box(panel, (5.0, y), name, facecolor=shade, edgecolor=edge, width=6.4,
                      height=0.95, fontsize=9.2, bold="Add" not in name)
            if i < len(layers) - 1:
                arrow(panel, (5.0, y - 0.5), (5.0, y - 1.35 + 0.5), scale=11)
            y -= 1.35
        arrow(panel, (5.0, top + 0.5), (5.0, top + 0.85), scale=13, color=GOOD)
        panel.text(5.0, top + 1.6 - 0.5, "x N layers stacked", ha="center", fontsize=9,
                   color=GOOD, fontweight="bold")

    fig.suptitle("The full Transformer: encoder reads, decoder generates",
                 fontsize=13, color=NAVY, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, "Source: Vaswani et al., \"Attention Is All You Need\" (2017)",
             ha="center", fontsize=8.2, color=GREY)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig
