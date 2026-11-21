"""Regenerate every diagram used by the NLP & Transformers notes.

Figures live in the ``figures/`` package, one module per topic, sharing the
palette and registry in ``figures/core.py``.

    cd foundation_course/08_nlp_transformers_basics/notes && python plot_nlp_figures.py

Pass figure names to rebuild only some of them:

    python plot_nlp_figures.py nlp_pipeline_stages
"""

from __future__ import annotations

import sys

from figures import render

if __name__ == "__main__":
    render(sys.argv[1:])
