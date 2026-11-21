"""Shared visual style and drawing helpers for the self-notes figures.

Copied from software-engineering-atlas so both sites share one visual language.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

NAVY = "#0B3D91"
BLUE = "#1E6FEB"
SKY = "#5B9BFF"
MIST = "#C9DCFF"
PALE = "#EAF1FF"
INK = "#12213D"
GREY = "#7183A3"
LINE = "#D5DEEC"
RED = "#C0392B"
GREEN = "#1E8449"
AMBER = "#D68910"
WHITE = "#FFFFFF"
# Soft fills that pair with the matching edge colour above.
AMBER_FACE = "#FFF5DC"
RED_FACE = "#FDEDEC"
GREEN_FACE = "#E9F7EF"
DEAD = "#F2F2F2"
DPI = 200
ASSETS_ROOT = Path(__file__).resolve().parent.parent.parent / "assets"


def render(figures: dict[str, Callable[[], plt.Figure]], topic: str,
           names: list[str]) -> None:
    """Render the named figures (all if none) into assets/<topic>/."""
    targets = names or list(figures)
    unknown = sorted(set(targets) - set(figures))
    if unknown:
        raise SystemExit(f"Unknown figure(s): {', '.join(unknown)}. "
                         f"Known: {', '.join(figures)}")
    folder = ASSETS_ROOT / topic
    folder.mkdir(parents=True, exist_ok=True)
    for name in targets:
        fig = figures[name]()
        fig.savefig(folder / f"{name}.png", dpi=DPI, bbox_inches="tight",
                    facecolor=WHITE)
        plt.close(fig)
        print(f"wrote assets/{topic}/{name}.png")


def caps(ax: plt.Axes, x: float, y: float, text: str, *, color: str = GREY,
         ha: str = "center", size: float = 8.5) -> None:
    """Bold ALL-CAPS column or zone label."""
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, fontweight="bold",
            color=color)


def note(ax: plt.Axes, x: float, y: float, text: str, *, color: str = INK,
         size: float = 8.6, ha: str = "center") -> None:
    """Plain explanatory text, multi-line friendly."""
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color,
            linespacing=1.45)


def badge(ax: plt.Axes, x: float, y: float, text: str, *, color: str = NAVY,
          size: float = 7.5) -> None:
    """Draw a numbered disc that is round on any aspect ratio and the same
    diameter whatever the label length ("1" and "FM" match)."""
    ax.plot(x, y, marker="o", markersize=size * 2.3, color=color, zorder=3)
    ax.text(x, y, text, ha="center", va="center", fontsize=size,
            fontweight="bold", color=WHITE, zorder=4)


def canvas(width: float = 14, height: float = 6) -> tuple[plt.Figure, plt.Axes]:
    """Return a clean drawing surface using percentage-like coordinates."""
    fig, ax = plt.subplots(figsize=(width, height), facecolor=WHITE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def heading(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    """Draw the standard navy title and optional italic subtitle."""
    ax.text(50, 96, title, ha="center", va="top", fontsize=19,
            fontweight="bold", color=NAVY)
    if subtitle:
        ax.text(50, 89, subtitle, ha="center", va="top", fontsize=10.5,
                color=INK, style="italic")


def box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str = "",
    *,
    face: str = PALE,
    edge: str = BLUE,
    title_color: str = NAVY,
    radius: float = 1.4,
    linewidth: float = 1.8,
) -> None:
    """Draw a rounded concept box with a title and optional subtitle."""
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.5,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth,
    )
    ax.add_patch(patch)
    cy = y + height / 2
    ax.text(x + width / 2, cy + (2.2 if subtitle else 0), title,
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color=title_color)
    if subtitle:
        ax.text(x + width / 2, cy - 3.1, subtitle,
                ha="center", va="center", fontsize=8.2, color=INK)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = BLUE,
    label: str = "",
    curve: float = 0,
    dashed: bool = False,
) -> None:
    """Draw a directional arrow with an optional readable label."""
    patch = FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.8, color=color,
        connectionstyle=f"arc3,rad={curve}",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(patch)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 3, label, ha="center", va="center", fontsize=8,
                color=color, bbox=dict(facecolor=WHITE, edgecolor="none", pad=1))


def stage_row(
    ax: plt.Axes,
    stages: list[tuple[str, str, str]],
    xs: list[float],
    *,
    y: float,
    width: float,
    height: float,
    highlight: tuple[int, ...] = (),
) -> None:
    """Draw numbered (badge, title, subtitle) stages left to right, linked by arrows.

    Highlighted indexes get a deeper fill to mark where the real work happens.
    """
    for index, ((badge, title, subtitle), x) in enumerate(zip(stages, xs)):
        box(ax, x, y, width, height, title, subtitle,
            face=MIST if index in highlight else PALE)
        centre = (x + width / 2, y + height + 6)
        ax.add_patch(Circle(centre, 3.3, facecolor=NAVY, edgecolor=NAVY))
        ax.text(*centre, badge, ha="center", va="center", fontsize=8,
                fontweight="bold", color=WHITE)
        if index < len(xs) - 1:
            arrow(ax, (x + width + 0.5, y + height / 2), (xs[index + 1] - 0.5, y + height / 2))


def card_row(
    ax: plt.Axes,
    cards: list[tuple[str, str, str, str, str, str]],
    labels: tuple[str, str],
    *,
    gap: float = 2.2,
    arrows: bool = False,
) -> None:
    """Draw a row of tall cards: (badge, name, first text, second text, colour, face).

    Each card shows a coloured badge, a navy name, and two labelled sections,
    e.g. ASK/SMELL or USE WHEN/COST. Width adapts to the number of cards, so
    this works for a row of four, five, or six. Set arrows=True (with a wider
    gap) when the cards are steps in order.
    """
    margin = 3
    width = (100 - 2 * margin - (len(cards) - 1) * gap) / len(cards)
    step = width + gap
    for index, (badge_text, name, first, second, color, face) in enumerate(cards):
        x = margin + index * step
        centre_x = x + width / 2
        if arrows and index < len(cards) - 1:
            arrow(ax, (x + width + 0.3, 40), (x + step - 0.3, 40), color=color)
        box(ax, x, 8, width, 64, "", face=face, edge=color)
        badge(ax, centre_x, 79, badge_text, color=color, size=14)
        ax.text(centre_x, 64, name, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=NAVY)
        ax.plot([x + 3, x + width - 3], [58, 58], color=LINE, linewidth=1.2)
        for label, text, top in ((labels[0], first, 51), (labels[1], second, 30)):
            ax.text(centre_x, top, label, ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color=GREY)
            ax.text(centre_x, top - 9, text, ha="center", va="center", fontsize=8.8,
                    color=INK, linespacing=1.45)
