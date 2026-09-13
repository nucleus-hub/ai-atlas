"""Figure modules for the Basic Machine Learning II notes.

Importing this package registers every topic's figures in ``core.REGISTRY``.
"""

from . import topic_01_decision_trees  # noqa: F401
from .core import REGISTRY, render

__all__ = ["REGISTRY", "render"]
