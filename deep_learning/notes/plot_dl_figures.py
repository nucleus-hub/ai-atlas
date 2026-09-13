"""Regenerate every diagram used by the Deep Learning Basics notes.

Figures live in the ``figures/`` package, one module per topic, sharing the
palette and registry in ``figures/core.py``. Run with the repo virtualenv:

    cd deep_learning/notes && ../../.venv/bin/python plot_dl_figures.py

Pass figure names to rebuild only some of them:

    ../../.venv/bin/python plot_dl_figures.py neuron_anatomy weights_bias_effect
"""

from __future__ import annotations

import sys

from figures import render

if __name__ == "__main__":
    render(sys.argv[1:])
