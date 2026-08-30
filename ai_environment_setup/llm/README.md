#  LLM

Working with Large Language Models — both hosted APIs and local inference.

## Structure

- **`cloud/`** — LLMs via hosted APIs (require keys in `.env`).
  - `chat_gemini.py` — streaming chat loop against Google's Gemini API.
- **`local/`** — LLMs running on your own machine.
  - `chat_local.py` — local inference with Qwen2.5-1.5B via  transformers (auto-detects Apple Silicon MPS / CPU).

## Setup

```bash
cp ../../.env.example ../../.env   # add GOOGLE_API_KEY, HF_TOKEN, etc.
python ../tools/check_keys.py       # verify keys load

# Cloud (Gemini)
python cloud/chat_gemini.py

# Local (Qwen) — first run downloads the model
python local/chat_local.py
```

Both loops support `/reset` (clear conversation) and `/quit`.

## Ideas to explore next
- [ ] Add an OpenAI-based cloud client
- [ ] RAG over local documents
- [ ] Function/tool calling
