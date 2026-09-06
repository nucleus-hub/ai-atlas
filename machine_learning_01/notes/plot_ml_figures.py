"""Regenerate every diagram used by the Basic Machine Learning I notes.

Figures live in the ``figures/`` package, one module per topic, sharing the
palette and registry in ``figures/core.py``. Run with the repo virtualenv:

    cd machine_learning_01/notes && ../../.venv/bin/python plot_ml_figures.py

Pass figure names to rebuild only some of them:

    ../../.venv/bin/python plot_ml_figures.py overfitting scaling_comparison
"""

from __future__ import annotations

import sys

from figures import render

if __name__ == "__main__":
    render(sys.argv[1:])
