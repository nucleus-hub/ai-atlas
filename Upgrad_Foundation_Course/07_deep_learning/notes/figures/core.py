"""Shared plumbing for the Deep Learning Basics note figures.

Holds the palette, the figure registry, and the render loop -- so each
``figures/topic_NN_*.py`` module only has to describe what it draws.

Deliberately mirrors ``06_machine_learning_02/notes/figures/core.py`` so every
note set in the atlas renders with one identical visual language.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# Shared palette -- matches the notes' heading colours.
NAVY = "#0B3D91"
BLUE = "#1E6FEB"
SKY = "#5B9BFF"
MIST = "#A9C9FF"
INK = "#12213D"
PAPER = "#FFFFFF"
GREY = "#8A9BB8"
LINE = "#D5DEEC"
WARN = "#C0392B"
GOOD = "#1E8449"

DPI = 200

REGISTRY: dict[str, Callable[[], plt.Figure]] = {}


def figure(name: str) -> Callable[[Callable[[], plt.Figure]], Callable[[], plt.Figure]]:
    """Register a figure-building function under ``name``."""

    def decorator(func: Callable[[], plt.Figure]) -> Callable[[], plt.Figure]:
        REGISTRY[name] = func
        return func

    return decorator


def blank_axes(ax: plt.Axes, xlim: tuple[float, float] = (0, 5), ylim: tuple[float, float] = (0, 5)) -> None:
    """Strip ticks and soften the frame -- used by most conceptual panels."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(LINE)


def panel_title(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    """Apply the standard two-line panel heading."""
    text = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(text, fontsize=12, fontweight="bold", color=NAVY, pad=10)


def node_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    text: str,
    facecolor: str = PAPER,
    edgecolor: str = BLUE,
    textcolor: str = INK,
    width: float = 1.5,
    height: float = 0.6,
    fontsize: int = 9,
    bold: bool = False,
) -> None:
    """Draw a rounded node box centred on ``xy`` -- the tree-diagram workhorse."""
    from matplotlib.patches import FancyBboxPatch

    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.6,
        zorder=3,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontweight="bold" if bold else "normal",
        zorder=4,
    )


def connect(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str = "",
    color: str = GREY,
    fontsize: int = 8,
) -> None:
    """Draw a labelled edge between two node centres."""
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-", color=color, linewidth=1.4),
        zorder=2,
    )
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(
            mx,
            my,
            label,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=INK,
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.5),
            zorder=5,
        )


def render(names: list[str] | None = None) -> None:
    """Render the requested figures (all of them when ``names`` is empty)."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    targets = names or list(REGISTRY)

    unknown = [n for n in targets if n not in REGISTRY]
    if unknown:
        raise SystemExit(f"Unknown figure(s): {', '.join(unknown)}. Known: {', '.join(REGISTRY)}")

    for name in targets:
        fig = REGISTRY[name]()
        out = ASSETS / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=PAPER)
        plt.close(fig)
        print(f"wrote assets/{out.name}")
