"""Figure modules for the Basic Machine Learning II notes.

Importing this package registers every topic's figures in ``core.REGISTRY``.
"""

from . import (  # noqa: F401
    topic_01_decision_trees,
    topic_02_random_forest,
)
from .core import REGISTRY, render

__all__ = ["REGISTRY", "render"]
