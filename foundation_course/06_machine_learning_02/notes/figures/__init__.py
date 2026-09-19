"""Figure modules for the Basic Machine Learning II notes.

Importing this package registers every topic's figures in ``core.REGISTRY``.
"""

from . import (  # noqa: F401
    topic_01_decision_trees,
    topic_02_ensembles_random_forest,
    topic_03_clustering_kmeans,
    topic_04_clustering_evaluation,
    topic_05_feature_engineering,
    topic_06_overfitting_underfitting,
    topic_07_regularization_cross_validation,
    topic_08_optimization_loss_functions,
)
from .core import REGISTRY, render

__all__ = ["REGISTRY", "render"]
