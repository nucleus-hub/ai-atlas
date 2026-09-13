"""Figure modules for the Deep Learning Basics notes.

Importing this package registers every topic's figures in ``core.REGISTRY``.
"""

from . import (  # noqa: F401
    topic_01_intro_neuron,
    topic_02_ann_forward_prop,
    topic_03_activations_losses,
    topic_04_backprop_gradient_descent,
    topic_05_optimizers,
    topic_06_regularization,
    topic_07_cnn,
)
from .core import REGISTRY, render

__all__ = ["REGISTRY", "render"]
