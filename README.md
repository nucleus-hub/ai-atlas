# ai-atlas

> A personal, long-term **AI learning notebook, experimentation space, and quick-reference knowledge base** — starting from Python fundamentals and growing toward Math, ML, Deep Learning, and LLMs.

This repository is my map of the AI landscape. It grows with me: I add sections as I reach them, so the structure stays lean and intuitive instead of full of empty folders.

---

## Purpose

- **Learn** — structured, hands-on material, built up gradually.
- **Reference** — a quick-lookup knowledge base for concepts, snippets, and gotchas.
- **Track progress** — a living roadmap of what's done and what's next.

---

## Learning Roadmap

```
Python  ->  Math Foundations  ->  Machine Learning  ->  Deep Learning  ->  LLMs / GenAI
  |              |                     |                    |                |
 code the    understand the       train & evaluate     build neural     prompt, fine-tune,
 tools       math behind ML          models              networks         deploy
```

I build this repo one section at a time. New top-level folders get added when I actually start that topic — no empty scaffolding.

---

## Repository Structure (current)

```
ai-atlas/
├── README.md                 <- you are here
├── requirements.in / .txt    <- shared Python dependencies
├── .env.example              <- template for API keys (never commit real .env)
│
├── python_basics/            <- Python learning (01-15 demos) + sample data
├── math_foundations/         <- math notes & examples (to be built up)
└── ai_environment_setup/     <- LLM clients (cloud/local) + env/key tools
```

**Planned sections** (added when I get there): `machine_learning/`, `deep_learning/`,
`generative_ai/`, `ai_concepts/`, `experiments/`, `notes/`, `projects/`.

---

## Section Guide

| Section | What lives here |
|---------|-----------------|
| **python_basics** | 15 hands-on demo scripts: basics, collections, functions, strings, files, exceptions, OOP, APIs, comprehensions, modules/lambdas, threading, sync, async, env config, logging. |
| **math_foundations** | Notes and worked examples for linear algebra, probability, statistics, and calculus — the math ML is built on. |
| **ai_environment_setup** | LLM chat clients (`llm/cloud/` Gemini, `llm/local/` Qwen) plus environment/key utilities (`tools/check_keys.py`). |

---

## How Python Concepts Connect to AI/ML

| Python concept (in `python_basics/`) | Why it matters for AI/ML |
|--------------------------------------|--------------------------|
| Collections, comprehensions | Data wrangling, feature engineering, batching |
| Functions & lambdas | Transformations, custom loss/metric functions |
| OOP | Modeling datasets, layers, estimators (sklearn/PyTorch style) |
| File I/O | Loading datasets, saving checkpoints |
| Exceptions | Robust data pipelines |
| Threading / async | Parallel data loading, concurrent API calls to LLMs |
| Env config & logging | Reproducible experiments, secret management, tracking runs |
| APIs (requests) | Calling model endpoints (Gemini, OpenAI, HF) |

---

## Progress Tracker

| Section | Status | Notes |
|---------|:------:|-------|
| python_basics | In progress | 15 demos complete |
| math_foundations | Not started | folder ready |
| ai_environment_setup | Started | cloud + local chat loops working |
| machine_learning | Planned | |
| deep_learning | Planned | |
| generative_ai | Planned | |
| ai_concepts | Planned | |

---

## Getting Started

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up secrets (never commit the real .env!)
cp .env.example .env             # then fill in your API keys
python ai_environment_setup/tools/check_keys.py   # verify keys are loaded

# 4. Run a Python basics demo
python python_basics/01_python_basics_demo.py
```

---

## Useful Resources

- **Python**: [Official Docs](https://docs.python.org/3/) · [Real Python](https://realpython.com/)
- **Math**: [3Blue1Brown](https://www.3blue1brown.com/) · [Khan Academy](https://www.khanacademy.org/math)
- **ML**: [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- **Deep Learning**: [d2l.ai](https://d2l.ai/) · [fast.ai](https://www.fast.ai/)
- **LLMs / GenAI**: [Hugging Face](https://huggingface.co/docs) · [Google AI Studio](https://ai.google.dev/)

---

*Maintained by [@nucleus-hub](https://github.com/nucleus-hub). A living document — update the tracker and add sections as the atlas grows.*
