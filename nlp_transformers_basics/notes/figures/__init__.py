"""Figure modules for the NLP & Transformers notes.

Importing this package registers every topic's figures in ``core.REGISTRY``.
"""

from . import (  # noqa: F401
    topic_01_intro_pipeline,
    topic_02_language_levels,
    topic_03_preprocessing,
    topic_04_text_representation,
    topic_05_demo1_walkthrough,
    topic_06_embeddings,
    topic_07_self_attention,
)
from .core import REGISTRY, render

__all__ = ["REGISTRY", "render"]
