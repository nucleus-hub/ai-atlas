"""Figure modules for the Deep Learning Basics notes.

Importing this package registers every topic's figures in ``core.REGISTRY``.
"""

from . import (  # noqa: F401
    topic_01_intro_neuron,
    topic_02_ann_forward_prop,
)
from .core import REGISTRY, render

__all__ = ["REGISTRY", "render"]
