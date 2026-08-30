# AI Environment Setup

Everything for getting an AI/LLM workspace running: environment/key utilities and
runnable LLM chat clients (cloud + local). Grouped here so setup and experimentation
live in one place, separate from the pure learning material.

## Structure

```
ai_environment_setup/
├── tools/
│   └── check_keys.py        <- verifies API keys load from .env
└── llm/
    ├── cloud/chat_gemini.py <- streaming chat via Google Gemini API
    └── local/chat_local.py  <- local inference (Qwen2.5-1.5B via transformers)
```

## Quick start

Run these from the **repo root** (so `.env` is picked up from the project root):

```bash
# 1. Keys (never commit the real .env)
cp .env.example .env
python ai_environment_setup/tools/check_keys.py

# 2. Cloud LLM (needs GOOGLE_API_KEY)
python ai_environment_setup/llm/cloud/chat_gemini.py

# 3. Local LLM (first run downloads the model; needs torch/transformers)
python ai_environment_setup/llm/local/chat_local.py
```

Both chat loops support `/reset` and `/quit`.

See `tools/README.md` and `llm/README.md` for details.
