"""Figure modules for the Basic Machine Learning I notes.

Importing this package registers every topic's figures in ``core.REGISTRY``.
"""

from . import topic_01_foundations, topic_02_data_prep, topic_03_metrics  # noqa: F401
from .core import REGISTRY, render

__all__ = ["REGISTRY", "render"]
