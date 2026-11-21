# AI Atlas — Self Notes

Concise, visual reference notes for the IIT Kharagpur Executive PG Certificate in Generative & Agentic AI.

| Topic | Covers |
|---|---|
| [Topic 5 — LLM Models & APIs](Topic5_LLM_Models_APIs_Self_Notes.md) | LLM fundamentals, prompt engineering, the OpenAI and Anthropic APIs, Hugging Face, RAG vs fine-tuning |

## Building the site

From the `ai-atlas` root:

```bash
# Live preview at http://127.0.0.1:8001
uv run --no-project --with mkdocs-material \
  --index-url https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/external-pypi/simple \
  --allow-insecure-host pypi.ci.artifacts.walmart.com \
  mkdocs serve -f mkdocs-self-notes.yml -a 127.0.0.1:8001

# Static build into site/self-notes/ (opens directly from disk)
uv run --no-project --with mkdocs-material \
  --index-url https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/external-pypi/simple \
  --allow-insecure-host pypi.ci.artifacts.walmart.com \
  mkdocs build -f mkdocs-self-notes.yml
```

Figures are generated, not hand-drawn. Regenerate them with `_support/tools/plot_topic_05_figures.py` (the run command is in its docstring).
