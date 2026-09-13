"""Figure modules for the Deep Learning Basics notes.

Importing this package registers every topic's figures in ``core.REGISTRY``.
"""

from . import topic_01_intro_neuron  # noqa: F401
from .core import REGISTRY, render

__all__ = ["REGISTRY", "render"]
