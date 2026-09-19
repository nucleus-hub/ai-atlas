"""Shared plumbing for the Basic Machine Learning I note figures.

Holds the palette, the figure registry, and the render loop -- so each
``figures/topic_NN_*.py`` module only has to describe what it draws.
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
