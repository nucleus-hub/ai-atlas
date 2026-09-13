"""Figures for Note 03 -- Activation Functions & Loss Functions.

Six diagrams:

* ``activation_functions``       -- all five plotted with their ranges
* ``sigmoid_saturation``         -- sigma(z) above its derivative; the 0.25 ceiling
* ``relu_vs_sigmoid_gradient``   -- gradient survival through stacked layers
* ``softmax_mechanics``          -- logits -> exp -> normalise, as bars
* ``loss_functions_compared``    -- MSE's 1.0 ceiling vs BCE's asymptote
* ``ce_vs_mse_gradient``         -- the 203x gradient gap where the model is worst
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .core import (
    BLUE,
    GOOD,
    GREY,
    INK,
    LINE,
    MIST,
    NAVY,
    PAPER,
    SKY,
    WARN,
    figure,
    panel_title,
)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _style(ax, xlabel="z", ylabel="f(z)"):
    ax.axhline(0, color=LINE, linewidth=1.0, zorder=1)
    ax.axvline(0, color=LINE, linewidth=1.0, zorder=1)
    ax.grid(alpha=0.18, color=LINE)
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)


@figure("activation_functions")
def activation_functions() -> plt.Figure:
    """The five activation functions, each in its own panel."""
    z = np.linspace(-5, 5, 400)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))

    panels = [
        ("Linear", z, "f(z) = z", "(-inf, inf)", "Regression output", BLUE),
        ("Sigmoid", _sigmoid(z), "1 / (1 + e^-z)", "(0, 1)", "Binary classification", SKY),
        ("Tanh", np.tanh(z), "tanh(z)", "(-1, 1)", "Hidden (older nets)", GREY),
        ("ReLU", np.maximum(0, z), "max(0, z)", "[0, inf)", "Most deep models", GOOD),
    ]

    for ax, (name, y, formula, rng, use, colour) in zip(axes, panels):
        ax.plot(z, y, color=colour, linewidth=2.6, zorder=3)
        _style(ax)
        ax.set_ylim(-1.6, 3.2) if name in ("Linear", "ReLU") else ax.set_ylim(-1.35, 1.35)
        panel_title(ax, name, formula)
        ax.text(0.5, -0.30, f"range {rng}\n{use}", transform=ax.transAxes,
                ha="center", va="top", fontsize=8, color=INK)

    # Softmax needs a bar panel, not a curve.
    ax = axes[4]
    logits = np.array([2.0, 1.0, 0.1])
    probs = np.exp(logits) / np.exp(logits).sum()
    bars = ax.bar(["A", "B", "C"], probs, color=[BLUE, SKY, MIST],
                  edgecolor=NAVY, linewidth=1.3, zorder=3)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.03, f"{p:.3f}",
                ha="center", fontsize=8.5, fontweight="bold", color=NAVY)
    ax.set_ylim(0, 0.85)
    ax.grid(alpha=0.18, color=LINE, axis="y")
    ax.set_ylabel("probability", fontsize=9, color=INK)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Softmax", "e^zi / sum(e^zj)")
    ax.text(0.5, -0.30, "range (0, 1), sums to 1\nMulti-class classification",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=INK)

    fig.suptitle("The five activation functions", fontsize=13, fontweight="bold",
                 color=NAVY, y=1.04)
    fig.tight_layout()
    return fig


@figure("sigmoid_saturation")
def sigmoid_saturation() -> plt.Figure:
    """Sigmoid above, its derivative below -- the 0.25 ceiling and the collapse."""
    z = np.linspace(-10, 10, 500)
    s = _sigmoid(z)
    d = s * (1 - s)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})

    ax = axes[0]
    ax.plot(z, s, color=SKY, linewidth=2.8, zorder=3)
    for edge in (-6, 6):
        ax.axvspan(edge - 4, edge + 4 if edge < 0 else edge, alpha=0)
    ax.axvspan(-10, -4, color=WARN, alpha=0.07, zorder=1)
    ax.axvspan(4, 10, color=WARN, alpha=0.07, zorder=1)
    ax.text(-7, 0.55, "SATURATED", ha="center", fontsize=9, color=WARN, fontweight="bold")
    ax.text(7, 0.45, "SATURATED", ha="center", fontsize=9, color=WARN, fontweight="bold")
    ax.text(0, 0.12, "useful\nrange", ha="center", fontsize=8.5, color=GOOD, fontweight="bold")
    _style(ax, xlabel="", ylabel="sigma(z)")
    ax.set_ylim(-0.08, 1.12)
    panel_title(ax, "Sigmoid flattens at both ends", "output stops responding to z")

    ax = axes[1]
    ax.plot(z, d, color=WARN, linewidth=2.8, zorder=3)
    ax.axhline(0.25, color=GOOD, linewidth=1.4, linestyle="--", zorder=2)
    ax.text(-9.6, 0.262, "maximum possible gradient = 0.25", fontsize=8.5,
            color=GOOD, fontweight="bold", va="bottom")
    for zz in (0, 2, 6):
        dv = float(_sigmoid(zz) * (1 - _sigmoid(zz)))
        ax.scatter([zz], [dv], s=60, color=NAVY, zorder=5, edgecolor=PAPER, linewidth=1.4)
        ax.annotate(f"z={zz}\n{dv:.4f}", xy=(zz, dv), xytext=(zz + 0.9, dv + 0.035),
                    fontsize=8, color=NAVY, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=NAVY, linewidth=1.1))
    _style(ax, xlabel="z", ylabel="sigma'(z)")
    ax.set_ylim(-0.02, 0.33)
    panel_title(ax, "Its gradient collapses toward zero",
                "at z=6 the signal is 101x weaker than at z=0")

    fig.tight_layout()
    return fig


@figure("relu_vs_sigmoid_gradient")
def relu_vs_sigmoid_gradient() -> plt.Figure:
    """Gradient surviving backwards through stacked layers."""
    layers = np.arange(0, 8)
    sig_best = 0.25 ** layers
    relu = 1.0 ** layers

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    ax.semilogy(layers, sig_best, "o-", color=WARN, linewidth=2.4, markersize=7,
                label="Sigmoid  (0.25 per layer, best case)", zorder=3)
    ax.semilogy(layers, relu, "o-", color=GOOD, linewidth=2.4, markersize=7,
                label="ReLU  (1.0 per layer)", zorder=3)
    ax.set_xlabel("layers travelled backwards", fontsize=9.5, color=INK)
    ax.set_ylabel("gradient magnitude  (log scale)", fontsize=9.5, color=INK)
    ax.grid(alpha=0.2, color=LINE, which="both")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Gradient survival through depth", "log scale -- sigmoid falls off a cliff")
    ax.annotate(f"5 layers: {0.25**5:.5f}", xy=(5, 0.25 ** 5), xytext=(3.1, 4e-4),
                fontsize=8.5, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.2))

    ax = axes[1]
    zz = np.linspace(-5, 5, 400)
    ax.plot(zz, _sigmoid(zz) * (1 - _sigmoid(zz)), color=WARN, linewidth=2.6,
            label="Sigmoid gradient  (peaks at 0.25)", zorder=3)
    ax.plot(zz, (zz > 0).astype(float), color=GOOD, linewidth=2.6,
            label="ReLU gradient  (exactly 1, or 0)", zorder=3)
    ax.fill_between(zz, 0, (zz <= 0).astype(float) * 1.15, color=WARN, alpha=0.06, zorder=1)
    ax.text(-2.5, 0.55, "dying ReLU zone\ngradient exactly 0", ha="center", fontsize=8.2,
            color=WARN, fontweight="bold")
    _style(ax, xlabel="z", ylabel="gradient")
    ax.set_ylim(-0.05, 1.2)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    panel_title(ax, "Why: the per-layer gradient", "ReLU passes 1 through; sigmoid shrinks")

    fig.tight_layout()
    return fig


@figure("softmax_mechanics")
def softmax_mechanics() -> plt.Figure:
    """Logits -> exponentiate -> normalise."""
    names = ["Class A", "Class B", "Class C"]
    logits = np.array([2.0, 1.0, 0.1])
    exps = np.exp(logits)
    probs = exps / exps.sum()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    stages = [
        (logits, "STEP 1 — raw logits", "network output, any value", MIST, "{:.1f}"),
        (exps, "STEP 2 — exponentiate", f"all positive · sum = {exps.sum():.4f}", SKY, "{:.4f}"),
        (probs, "STEP 3 — divide by the sum", "probabilities · sum = 1.000000", BLUE, "{:.4f}"),
    ]

    for ax, (vals, title, subtitle, colour, fmt) in zip(axes, stages):
        bars = ax.bar(names, vals, color=colour, edgecolor=NAVY, linewidth=1.4, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.04,
                    fmt.format(v), ha="center", fontsize=9.5, fontweight="bold", color=NAVY)
        ax.set_ylim(0, max(vals) * 1.28)
        ax.grid(alpha=0.18, color=LINE, axis="y")
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)
        panel_title(ax, title, subtitle)

    axes[2].axhline(1.0, color=GOOD, linewidth=1.3, linestyle="--", zorder=2)
    fig.suptitle("Softmax turns arbitrary scores into a probability distribution",
                 fontsize=11.5, fontweight="bold", color=NAVY, y=1.02)
    fig.tight_layout()
    return fig


@figure("loss_functions_compared")
def loss_functions_compared() -> plt.Figure:
    """MSE is capped at 1.0; cross-entropy is unbounded."""
    yhat = np.linspace(0.001, 0.999, 500)
    mse = (1 - yhat) ** 2
    bce = -np.log(yhat)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9))

    ax = axes[0]
    ax.plot(yhat, mse, color=SKY, linewidth=2.8, label="MSE  = (1 - yhat)^2", zorder=3)
    ax.plot(yhat, bce, color=WARN, linewidth=2.8, label="BCE  = -log(yhat)", zorder=3)
    ax.axhline(1.0, color=SKY, linewidth=1.3, linestyle="--", zorder=2)
    ax.text(0.62, 1.09, "MSE can never exceed 1.0", fontsize=8.5, color=SKY, fontweight="bold")
    for xv, colour in ((0.8, GOOD), (0.4, WARN)):
        ax.axvline(xv, color=GREY, linewidth=1.0, linestyle=":", zorder=1)
    ax.scatter([0.8, 0.4], [0.04, 0.36], s=70, color=SKY, zorder=5,
               edgecolor=PAPER, linewidth=1.5)
    ax.scatter([0.8, 0.4], [-np.log(0.8), -np.log(0.4)], s=70, color=WARN, zorder=5,
               edgecolor=PAPER, linewidth=1.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 4.2)
    ax.set_xlabel("predicted probability for the TRUE class  (y = 1)", fontsize=9, color=INK)
    ax.set_ylabel("loss", fontsize=9, color=INK)
    ax.grid(alpha=0.18, color=LINE)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "Loss as the prediction gets worse", "leftwards = more wrong")

    ax = axes[1]
    tail = np.linspace(0.0001, 0.25, 400)
    ax.semilogy(tail, (1 - tail) ** 2, color=SKY, linewidth=2.8, label="MSE", zorder=3)
    ax.semilogy(tail, -np.log(tail), color=WARN, linewidth=2.8, label="BCE", zorder=3)
    ax.set_xlabel("predicted probability for the TRUE class", fontsize=9, color=INK)
    ax.set_ylabel("loss  (log scale)", fontsize=9, color=INK)
    ax.grid(alpha=0.2, color=LINE, which="both")
    ax.legend(fontsize=8.5, loc="center right", framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    panel_title(ax, "The confidently-wrong tail", "MSE flattens at 1.0 · BCE heads for infinity")
    ax.text(0.09, 1.35, "MSE barely reacts\nto catastrophe", fontsize=8.4, color=SKY,
            fontweight="bold", ha="center")

    fig.tight_layout()
    return fig


@figure("ce_vs_mse_gradient")
def ce_vs_mse_gradient() -> plt.Figure:
    """The gradient each loss delivers through a sigmoid output."""
    z = np.linspace(-8, 4, 500)
    yh = _sigmoid(z)
    mse_grad = np.abs(2 * (yh - 1) * yh * (1 - yh))
    bce_grad = np.abs(yh - 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.plot(z, bce_grad, color=GOOD, linewidth=2.9, label="Cross-Entropy   |dL/dz| = |yhat - y|", zorder=3)
    ax.plot(z, mse_grad, color=WARN, linewidth=2.9, label="MSE                   |dL/dz| = |2(yhat-y)*sigma'(z)|", zorder=3)

    ax.axvspan(-8, -4, color=WARN, alpha=0.07, zorder=1)
    ax.text(-6.9, 0.72, "CONFIDENTLY WRONG\n(true label is 1)", ha="center", fontsize=9,
            color=WARN, fontweight="bold")

    zc = -6.0
    yc = float(_sigmoid(zc))
    g_bce = abs(yc - 1)
    g_mse = abs(2 * (yc - 1) * yc * (1 - yc))
    ax.scatter([zc, zc], [g_bce, g_mse], s=85, color=NAVY, zorder=6,
               edgecolor=PAPER, linewidth=1.6)
    ax.annotate(f"{g_bce:.4f}", xy=(zc, g_bce), xytext=(-4.6, 0.90), fontsize=9,
                color=GOOD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GOOD, linewidth=1.3))
    ax.annotate(f"{g_mse:.4f}  -- almost nothing", xy=(zc, g_mse), xytext=(-3.4, 0.13),
                fontsize=9, color=WARN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=WARN, linewidth=1.3))
    ax.annotate("", xy=(zc, g_bce), xytext=(zc, g_mse),
                arrowprops=dict(arrowstyle="<->", color=NAVY, linewidth=1.8), zorder=5)
    ax.text(zc - 0.28, 0.5, "203x", rotation=90, ha="center", va="center",
            fontsize=11, color=NAVY, fontweight="bold",
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5), zorder=7)

    ax.set_xlabel("z  (output pre-activation)", fontsize=9.5, color=INK)
    ax.set_ylabel("|gradient| delivered to the weights", fontsize=9.5, color=INK)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.18, color=LINE)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)
    ax.set_title("The real reason for Cross-Entropy: gradient where the model is most wrong",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig
