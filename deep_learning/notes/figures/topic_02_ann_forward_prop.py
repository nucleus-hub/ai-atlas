"""Figures for Note 02 -- Artificial Neural Networks & Forward Propagation.

Five diagrams:

* ``ann_layer_structure``  -- the four layer roles and what each one learns
* ``depth_vs_width``       -- deep-narrow vs wide-shallow at equal budget
* ``forward_pass_worked``  -- the 2->3->1 network with real numbers flowing
* ``xor_hidden_transform`` -- XOR before/after the hidden-layer transformation
* ``linear_collapse``      -- stacked linear layers collapse; ReLU does not
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

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
    blank_axes,
    figure,
    node_box,
    panel_title,
)

# The worked network, shared by several figures.
W1 = np.array([[0.5, -0.2, 0.4], [0.3, 0.6, -0.1]])
B1 = np.array([0.1, 0.0, -0.1])
W2 = np.array([[0.7], [-0.3], [0.5]])
B2 = np.array([0.0])


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-z))


@figure("ann_layer_structure")
def ann_layer_structure() -> plt.Figure:
    """Input -> Hidden 1 -> Hidden 2 -> Output, with each layer's job."""
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    blank_axes(ax, xlim=(0, 13), ylim=(0, 6.4))
    for spine in ax.spines.values():
        spine.set_visible(False)

    columns = [
        (1.55, 4, MIST, "1", "Input Layer", "Raw data:\npixels, numbers,\ntext tokens", "no computation"),
        (4.6, 5, SKY, "2", "Hidden Layer 1", "Learns low-level\nfeatures:\nedges, patterns", ""),
        (7.65, 5, SKY, "3", "Hidden Layer 2", "Combines features:\nshapes, structures", ""),
        (10.9, 2, BLUE, "4", "Output Layer", "Final prediction:\nclass label,\nprobability", ""),
    ]

    centres: list[list[tuple[float, float]]] = []
    for x, count, colour, _num, _name, _desc, _foot in columns:
        span = 0.62
        top = 3.55 + (count - 1) * span / 2
        centres.append([(x, top - i * span) for i in range(count)])

    # Connections first so nodes sit on top.
    for left, right in zip(centres, centres[1:]):
        for sx, sy in left:
            for dx, dy in right:
                ax.plot([sx + 0.21, dx - 0.21], [sy, dy], color=LINE, linewidth=0.7, zorder=1)

    for (x, _count, colour, num, name, desc, foot), nodes in zip(columns, centres):
        for cx, cy in nodes:
            ax.add_patch(Circle((cx, cy), 0.21, facecolor=colour, edgecolor=NAVY,
                                linewidth=1.2, zorder=3))
        ax.text(x, 5.95, num, ha="center", va="center", fontsize=11, fontweight="bold",
                color=PAPER, zorder=5,
                bbox=dict(boxstyle="circle,pad=0.32", facecolor=NAVY, edgecolor="none"))
        ax.text(x, 5.35, name, ha="center", va="center", fontsize=10,
                fontweight="bold", color=NAVY)
        ax.text(x, 1.52, desc, ha="center", va="top", fontsize=8.2, color=INK)
        if foot:
            ax.text(x, 0.55, foot, ha="center", va="center", fontsize=8,
                    color=WARN, fontweight="bold", style="italic")

    for x in (3.1, 6.15, 9.25):
        ax.annotate("", xy=(x + 0.3, 3.55), xytext=(x - 0.3, 3.55),
                    arrowprops=dict(arrowstyle="->", color=BLUE, linewidth=1.8), zorder=4)

    ax.text(6.5, 0.1, "Information flows one way:  input → hidden(s) → output",
            ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
    ax.set_title("Building a neural network — layer by layer",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=14)
    fig.tight_layout()
    return fig


@figure("depth_vs_width")
def depth_vs_width() -> plt.Figure:
    """Two networks, similar budget, very different shape."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    def draw_net(ax, layer_sizes, title, subtitle, accent):
        blank_axes(ax, xlim=(0, 10), ylim=(0, 8))
        n_layers = len(layer_sizes)
        xs = np.linspace(1.4, 8.6, n_layers)
        centres = []
        for x, size in zip(xs, layer_sizes):
            span = min(0.95, 6.4 / max(size, 1))
            top = 4.2 + (size - 1) * span / 2
            centres.append([(x, top - i * span) for i in range(size)])
        for left, right in zip(centres, centres[1:]):
            for sx, sy in left:
                for dx, dy in right:
                    ax.plot([sx, dx], [sy, dy], color=LINE, linewidth=0.6, zorder=1)
        for i, nodes in enumerate(centres):
            colour = MIST if i == 0 else (BLUE if i == len(centres) - 1 else accent)
            for cx, cy in nodes:
                ax.add_patch(Circle((cx, cy), 0.17, facecolor=colour, edgecolor=NAVY,
                                    linewidth=1.0, zorder=3))
        panel_title(ax, title, subtitle)

    draw_net(axes[0], [2, 3, 3, 3, 3, 1], "DEEP & NARROW",
             "6 layers · few neurons each", SKY)
    axes[0].text(5.0, 0.75, "Depth buys FEATURE COMPOSITION\neach layer builds on the last",
                 ha="center", fontsize=9, color=GOOD, fontweight="bold")

    draw_net(axes[1], [2, 10, 1], "WIDE & SHALLOW",
             "3 layers · many neurons each", SKY)
    axes[1].text(5.0, 0.75, "Width buys CAPACITY\nmore detail within one stage",
                 ha="center", fontsize=9, color=GOOD, fontweight="bold")

    fig.suptitle("Depth and width are two different dials — similar parameter budget, different behaviour",
                 fontsize=10.5, color=INK, y=0.035)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


@figure("forward_pass_worked")
def forward_pass_worked() -> plt.Figure:
    """The 2 -> 3 -> 1 forward pass with every real number shown."""
    x = np.array([0.9, 0.8])
    z1 = x @ W1 + B1
    a1 = np.maximum(0, z1)
    z2 = float((a1 @ W2 + B2)[0])
    y_hat = float(_sigmoid(z2))

    fig, ax = plt.subplots(figsize=(12.8, 6.0))
    blank_axes(ax, xlim=(0, 13), ylim=(0.2, 7.2))
    for spine in ax.spines.values():
        spine.set_visible(False)

    in_pos = [(1.5, 4.85), (1.5, 2.65)]
    hid_pos = [(6.0, 5.75), (6.0, 3.75), (6.0, 1.75)]
    out_pos = (10.6, 3.75)

    # Edges with weights.
    for i, (sx, sy) in enumerate(in_pos):
        for j, (dx, dy) in enumerate(hid_pos):
            ax.plot([sx + 0.52, dx - 0.62], [sy, dy], color=LINE, linewidth=1.0, zorder=1)
            t = 0.32 + 0.13 * j
            lx, ly = sx + 0.52 + t * (dx - 0.62 - sx - 0.52), sy + t * (dy - sy)
            ax.text(lx, ly, f"{W1[i, j]:+.1f}", ha="center", va="center", fontsize=7.4,
                    color=BLUE, fontweight="bold",
                    bbox=dict(facecolor=PAPER, edgecolor="none", pad=0.9), zorder=5)
    for j, (sx, sy) in enumerate(hid_pos):
        ax.plot([sx + 0.62, out_pos[0] - 0.55], [sy, out_pos[1]], color=LINE,
                linewidth=1.0, zorder=1)
        lx = sx + 0.62 + 0.55 * (out_pos[0] - 0.55 - sx - 0.62)
        ly = sy + 0.55 * (out_pos[1] - sy)
        ax.text(lx, ly, f"{W2[j, 0]:+.1f}", ha="center", va="center", fontsize=7.4,
                color=BLUE, fontweight="bold",
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=0.9), zorder=5)

    # Input nodes.
    labels = [("x1 = 0.9", "study"), ("x2 = 0.8", "sleep")]
    for (cx, cy), (val, meaning) in zip(in_pos, labels):
        ax.add_patch(Circle((cx, cy), 0.52, facecolor=MIST, edgecolor=NAVY,
                            linewidth=1.6, zorder=3))
        ax.text(cx, cy + 0.09, val, ha="center", va="center", fontsize=9,
                fontweight="bold", color=NAVY, zorder=4)
        ax.text(cx, cy - 0.24, meaning, ha="center", va="center", fontsize=7.2,
                color=INK, style="italic", zorder=4)

    # Hidden nodes with z and a.
    for j, (cx, cy) in enumerate(hid_pos):
        ax.add_patch(Circle((cx, cy), 0.62, facecolor=SKY, edgecolor=NAVY,
                            linewidth=1.6, zorder=3))
        ax.text(cx, cy + 0.20, f"z={z1[j]:.2f}", ha="center", va="center", fontsize=8.2,
                color=PAPER, zorder=4)
        ax.text(cx, cy - 0.16, f"a={a1[j]:.2f}", ha="center", va="center", fontsize=9.2,
                color=INK, fontweight="bold", zorder=4)

    # Output node.
    ax.add_patch(Circle(out_pos, 0.55, facecolor=BLUE, edgecolor=NAVY,
                        linewidth=1.6, zorder=3))
    ax.text(out_pos[0], out_pos[1], f"{y_hat:.3f}", ha="center", va="center",
            fontsize=10, fontweight="bold", color=PAPER, zorder=4)

    ax.text(1.5, 6.05, "INPUT", ha="center", fontsize=9.5, fontweight="bold", color=NAVY)
    ax.text(6.0, 6.72, "HIDDEN  (ReLU)", ha="center", fontsize=9.5, fontweight="bold", color=NAVY)
    ax.text(10.6, 4.72, "OUTPUT  (Sigmoid)", ha="center", fontsize=9.5, fontweight="bold", color=NAVY)

    ax.text(10.6, 2.72, f"z2 = {z2:.4f}\nŷ = σ(z2) = {y_hat:.4f}", ha="center", va="top",
            fontsize=8.6, color=INK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PAPER, edgecolor=BLUE, linewidth=1.2))

    ax.text(6.3, 0.62,
            "a1 = [0.79, 0.30, 0.18]   →   z2 = 0.5530   →   ŷ = 0.6348   →   PASS  (true label: Pass)",
            ha="center", fontsize=9.5, color=GOOD, fontweight="bold")

    ax.set_title("A worked forward pass — 2 inputs → 3 hidden (ReLU) → 1 output (Sigmoid)",
                 fontsize=12.5, fontweight="bold", color=NAVY, pad=12)
    fig.tight_layout()
    return fig


@figure("xor_hidden_transform")
def xor_hidden_transform() -> plt.Figure:
    """XOR is not linearly separable -- until the hidden layer moves the points."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    Wx = np.array([[1.0, 1.0], [1.0, 1.0]])
    bx = np.array([0.0, -1.0])
    H = np.maximum(0, X @ Wx + bx)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))

    # -- Left: original space -------------------------------------------------
    ax = axes[0]
    for (px, py), t in zip(X, y):
        ax.scatter(px, py, s=320, zorder=4,
                   color=BLUE if t else GREY, edgecolor=PAPER, linewidth=2)
        ax.text(px, py, str(t), ha="center", va="center", fontsize=11,
                fontweight="bold", color=PAPER, zorder=5)
    for angle, off in ((1.0, 0.5), (1.0, 1.5), (-1.0, 0.5)):
        xs = np.linspace(-0.45, 1.45, 50)
        ax.plot(xs, off - angle * xs, color=WARN, linewidth=1.3, linestyle="--", alpha=0.55, zorder=2)
    ax.text(0.5, -0.33, "no straight line can separate them",
            ha="center", fontsize=9, color=WARN, fontweight="bold")
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(-0.45, 1.45)
    ax.set_xlabel("x1", fontsize=9.5, color=INK)
    ax.set_ylabel("x2", fontsize=9.5, color=INK)
    ax.grid(alpha=0.18, color=LINE)
    panel_title(ax, "BEFORE — original input space", "the two 1s sit on opposite corners")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)

    # -- Right: hidden space --------------------------------------------------
    ax = axes[1]
    xs = np.linspace(-0.45, 2.45, 50)
    ax.plot(xs, (xs - 0.5) / 2.0, color=GOOD, linewidth=2.2, zorder=3,
            label="h1 − 2·h2 = 0.5")
    ax.fill_between(xs, (xs - 0.5) / 2.0, -0.8, color=GOOD, alpha=0.08, zorder=1)

    seen: dict[tuple[float, float], list[str]] = {}
    for (hx, hy), (ox, oy) in zip(H, X):
        seen.setdefault((hx, hy), []).append(f"({int(ox)},{int(oy)})")
    for (hx, hy), origins in seen.items():
        t = y[[i for i, h in enumerate(H) if tuple(h) == (hx, hy)][0]]
        ax.scatter(hx, hy, s=340, zorder=4,
                   color=BLUE if t else GREY, edgecolor=PAPER, linewidth=2)
        ax.text(hx, hy, str(t), ha="center", va="center", fontsize=11,
                fontweight="bold", color=PAPER, zorder=5)
        ax.annotate(" & ".join(origins), xy=(hx, hy), xytext=(hx, hy + 0.30),
                    ha="center", fontsize=8, color=INK, fontweight="bold", zorder=6)

    ax.annotate("both 1-cases land\non the SAME point",
                xy=(1.0, -0.05), xytext=(0.62, -0.42), fontsize=8.2, color=GOOD,
                fontweight="bold", ha="center",
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5),
                arrowprops=dict(arrowstyle="->", color=GOOD, linewidth=1.4))
    ax.text(2.28, 0.52, "1-side", ha="right", va="center", fontsize=8,
            color=GOOD, fontweight="bold", style="italic")
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-0.78, 1.45)
    ax.set_xlabel("h1  (OR-ish)", fontsize=9.5, color=INK)
    ax.set_ylabel("h2  (AND-ish)", fontsize=9.5, color=INK)
    ax.grid(alpha=0.18, color=LINE)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    panel_title(ax, "AFTER — hidden-layer space", "now ONE straight line works")
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)

    fig.suptitle("The hidden layer does not classify — it RESHAPES the problem until a line suffices",
                 fontsize=10.5, color=INK, y=0.035)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


@figure("linear_collapse")
def linear_collapse() -> plt.Figure:
    """Without an activation, stacked layers collapse into a single line."""
    rng = np.random.default_rng(7)
    n = 120
    pts = rng.uniform(-2.2, 2.2, size=(n, 2))
    labels = (pts[:, 1] > 0.75 * np.sin(1.7 * pts[:, 0]) + 0.25 * pts[:, 0]).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    grid_x, grid_y = np.meshgrid(np.linspace(-2.4, 2.4, 260), np.linspace(-2.4, 2.4, 260))
    grid = np.c_[grid_x.ravel(), grid_y.ravel()]

    A = np.array([[1.1, -0.7], [0.4, 1.3], [-0.9, 0.5]])
    ba = np.array([0.3, -0.2, 0.15])
    Bm = np.array([[0.8, -1.2, 0.6]])
    bb = np.array([0.1])

    def plot_panel(ax, zz, title, subtitle, note, colour):
        ax.contourf(grid_x, grid_y, zz.reshape(grid_x.shape) > 0,
                    levels=[-0.5, 0.5, 1.5], colors=[MIST, "#FFE1DC"], alpha=0.55, zorder=1)
        ax.contour(grid_x, grid_y, zz.reshape(grid_x.shape), levels=[0],
                   colors=[colour], linewidths=2.4, zorder=3)
        ax.scatter(pts[labels == 1, 0], pts[labels == 1, 1], s=22, color=BLUE,
                   edgecolor=PAPER, linewidth=0.5, zorder=4)
        ax.scatter(pts[labels == 0, 0], pts[labels == 0, 1], s=22, color=GREY,
                   edgecolor=PAPER, linewidth=0.5, zorder=4)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        ax.set_xticks([])
        ax.set_yticks([])
        panel_title(ax, title, subtitle)
        ax.text(0, -2.18, note, ha="center", fontsize=8.6, color=colour, fontweight="bold",
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=2), zorder=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(LINE)

    z_lin = (Bm @ (A @ grid.T + ba[:, None]) + bb[:, None]).ravel()
    plot_panel(axes[0], z_lin, "NO activation", "two stacked linear layers",
               "still ONE straight line — the layers collapsed", WARN)

    z_relu = (Bm @ np.maximum(0, A @ grid.T + ba[:, None]) + bb[:, None]).ravel()
    plot_panel(axes[1], z_relu, "WITH ReLU activation", "the same two layers",
               "a bent boundary — depth now means something", GOOD)

    fig.suptitle("W2(W1x + b1) + b2  =  (W2W1)x + (W2b1 + b2)   —   verified identical in NumPy",
                 fontsize=10, color=INK, y=0.035)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig
