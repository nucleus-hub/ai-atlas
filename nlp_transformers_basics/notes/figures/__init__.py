"""Figure modules for the NLP & Transformers notes.

Importing this package registers every topic's figures in ``core.REGISTRY``.
"""

from . import topic_01_intro_pipeline, topic_02_language_levels  # noqa: F401
from .core import REGISTRY, render

__all__ = ["REGISTRY", "render"]
