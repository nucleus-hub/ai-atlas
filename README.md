#  ai-atlas

> A personal, long-term **AI learning notebook, experimentation space, and quick-reference knowledge base** — from Python fundamentals all the way to LLMs and Generative AI.

This repository is my map of the AI landscape. It grows with me: every concept I learn, every experiment I run, and every note worth keeping lands here in a structure that stays intuitive for years.

---

##  Purpose

- **Learn** — structured, hands-on material from Python → Math → ML → DL → LLMs.
- **Experiment** — a safe scratch space for trying ideas without polluting "clean" material.
- **Reference** — a quick-lookup knowledge base for concepts, snippets, and gotchas.
- **Track progress** — a living roadmap of what's done, in-progress, and next.

---

##  Learning Roadmap

```
Python  →  Math Foundations  →  Machine Learning  →  Deep Learning  →  LLMs / GenAI
  │              │                     │                   │                │
 code the    understand the       train & evaluate    build neural     fine-tune,
 tools       math behind ML         models             networks        prompt, deploy
```

**Recommended progression:**

1. **`python_basics/`** — Get fluent in Python (data structures, OOP, concurrency, I/O, logging, env config). *You can't do AI without solid Python.*
2. **`math_foundations/`** — Linear algebra, probability, statistics, calculus. The language ML is written in.
3. **`machine_learning/`** — Fundamentals + classic algorithms (regression, trees, clustering, evaluation).
4. **`deep_learning/`** — Neural nets, backprop, CNNs, RNNs, transformers.
5. **`llm/`** — Working with large language models (cloud APIs + local inference).
6. **`generative_ai/`** — Diffusion, image/audio generation, multimodal, agents.
7. **`ai_concepts/`** — Cross-cutting theory: evaluation, ethics, MLOps, RAG, etc.

Support sections used **throughout** the journey:
- **`experiments/`** — throwaway trials and proofs-of-concept.
- **`notes/`** — distilled study notes (markdown).
- **`projects/`** — larger, "real" builds worth keeping.
- **`tools/`** — reusable dev utilities.

---

##  Repository Structure

```
ai-atlas/
├── README.md                 ← you are here
├── requirements.in / .txt    ← shared Python dependencies
├── .env.example              ← template for API keys (never commit real .env)
│
├── python_basics/            ← Python learning (01–15 demos) + sample data
├── math_foundations/         ← linear_algebra / probability / statistics / calculus
├── machine_learning/         ← fundamentals / algorithms
├── deep_learning/            ← neural networks & architectures
├── llm/                      ← cloud/ (API) + local/ (on-device inference)
├── generative_ai/            ← diffusion, multimodal, agents
├── ai_concepts/              ← cross-cutting theory & practice
├── experiments/              ← scratch space
├── notes/                    ← study notes
├── projects/                 ← larger builds
└── tools/                    ← dev utilities (e.g. env key checker)
```

---

##  Section Guide

| Section | What lives here |
|---------|-----------------|
| **python_basics** | 15 hands-on demo scripts covering basics, collections, functions, strings, files, exceptions, OOP, APIs, comprehensions, modules/lambdas, threading, sync, async, env config, logging. |
| **math_foundations** | The math ML is built on — organized by discipline. |
| **machine_learning** | Classic ML: theory (`fundamentals/`) + implementations (`algorithms/`). |
| **deep_learning** | Neural networks: from perceptrons to transformers. |
| **llm** | Chatting with LLMs — `cloud/` (Gemini/OpenAI APIs) and `local/` (Qwen via transformers). |
| **generative_ai** | Beyond text: images, audio, multimodal, agentic systems. |
| **ai_concepts** | Evaluation, RAG, prompting, MLOps, ethics, and other cross-cutting topics. |
| **experiments** | Rapid, disposable prototyping. |
| **notes** | Distilled markdown notes for quick recall. |
| **projects** | End-to-end builds worth showcasing. |
| **tools** | Small utilities (e.g. `check_keys.py`). |

---

##  How Python Concepts Connect to AI/ML

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

##  Progress Tracker

| Section | Status | Notes |
|---------|:------:|-------|
| python_basics |  In progress | 15 demos complete |
| math_foundations |  Not started | |
| machine_learning |  Not started | |
| deep_learning |  Not started | |
| llm |  Started | cloud + local chat loops working |
| generative_ai |  Not started | |
| ai_concepts |  Not started | |

> Legend:  solid ·  started ·  not started ·  needs revisit

---

##  Getting Started

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up secrets (never commit the real .env!)
cp .env.example .env             # then fill in your API keys
python tools/check_keys.py       # verify keys are loaded
```

---

##  Useful Resources

- **Python**: [Official Docs](https://docs.python.org/3/) · [Real Python](https://realpython.com/)
- **Math**: [3Blue1Brown](https://www.3blue1brown.com/) · [Khan Academy](https://www.khanacademy.org/math)
- **ML**: [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) · Andrew Ng's ML course
- **Deep Learning**: [d2l.ai](https://d2l.ai/) · [fast.ai](https://www.fast.ai/)
- **LLMs / GenAI**: [Hugging Face](https://huggingface.co/docs) · [Google AI Studio](https://ai.google.dev/)

---

*Maintained by [@nucleus-hub](https://github.com/nucleus-hub). This is a living document — update the tracker and sections as the atlas grows.*
