"""Render the diagrams for Topic 5: LLM Models & APIs.

Run from the 00_self_notes directory:

    uv run --with matplotlib \
      --index-url https://pypi.ci.artifacts.walmart.com/artifactory/api/pypi/external-pypi/simple \
      --allow-insecure-host pypi.ci.artifacts.walmart.com \
      python _support/tools/plot_topic_05_figures.py

Pass figure names to render only selected assets.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt

from figure_style import (
    AMBER,
    AMBER_FACE,
    BLUE,
    GREEN,
    GREEN_FACE,
    GREY,
    INK,
    MIST,
    NAVY,
    PALE,
    WHITE,
    arrow,
    box,
    canvas,
    caps,
    card_row,
    heading,
    note,
    render,
    stage_row,
)

TOPIC = "topic-05"
MONO = {"family": "monospace"}


def llm_inference_pipeline() -> plt.Figure:
    fig, ax = canvas(14, 5.6)
    heading(ax, "How an LLM turns a prompt into a response",
            "Generation is a loop: each predicted token is appended and fed back in")
    stage_row(ax, [
        ("1", "Prompt", "\"The capital of\nFrance is\""),
        ("2", "Tokenize", "text -> token IDs"),
        ("3", "Embed", "IDs -> vectors"),
        ("4", "Transformer", "attention layers\nweigh the context"),
        ("5", "Next token", "probabilities ->\n\"Paris\" (likely)"),
    ], [3, 23, 43, 63, 83], y=40, width=14, height=22, highlight=(3, 4))
    arrow(ax, (90, 38.5), (10, 38.5), curve=-0.12, dashed=True, color=GREY)
    note(ax, 50, 19, "Append the chosen token to the input and repeat", color=GREY, size=9)
    note(ax, 50, 9, "Generation stops at an end-of-sequence token or when the "
         "max-token limit is reached.", size=8.6)
    return fig


def llm_evolution() -> plt.Figure:
    fig, ax = canvas(14, 5)
    heading(ax, "How we got to modern LLMs",
            "Each step replaced hand-written knowledge with patterns learned from data")
    stage_row(ax, [
        ("1", "Rule-based AI", "hand-written rules,\nbrittle outside them"),
        ("2", "Statistical ML", "patterns learned\nfrom data"),
        ("3", "Deep learning", "neural nets +\nword embeddings"),
        ("4", "Transformers", "attention-based,\n2017 onward"),
        ("5", "Foundation models", "instruction-tuned\nassistants + tools"),
    ], [2, 22, 42, 62, 82], y=34, width=16, height=24, highlight=(3, 4))
    note(ax, 50, 12, "Milestone: \"Attention Is All You Need\" (2017) introduced the "
         "Transformer, the backbone of almost every modern LLM.", size=8.8)
    return fig


def llm_training_stages() -> plt.Figure:
    fig, ax = canvas(14, 5.6)
    heading(ax, "From raw data to a usable assistant",
            "Training changes the model's weights; inference only uses them")
    xs = [2, 16, 30, 44, 58, 72, 86]
    stage_row(ax, [
        ("1", "Data prep", "collect +\nfilter"),
        ("2", "Tokenize", "text -> IDs"),
        ("3", "Pretrain", "predict the\nnext token"),
        ("4", "Instruction", "follow\ninstructions"),
        ("5", "Alignment", "prefer helpful,\nsafe answers"),
        ("6", "Evaluate", "quality +\nsafety tests"),
        ("7", "Inference", "answer your\nprompts"),
    ], xs, y=36, width=11, height=22, highlight=(2, 6))
    ax.plot([2, 83], [77, 77], color=BLUE, linewidth=2)
    caps(ax, 42.5, 81, "BUILD THE MODEL  (done once, by the provider)", color=BLUE)
    ax.plot([86, 97], [77, 77], color=GREEN, linewidth=2)
    caps(ax, 91.5, 81, "USE THE MODEL", color=GREEN)
    note(ax, 50, 14, "Only stages 3-5 update the weights. Stages 4 and 5 are common, "
         "not universal:\nthe exact recipe varies from model to model.", color=GREY, size=8.8)
    return fig


def prompt_anatomy() -> plt.Figure:
    fig, ax = canvas(14, 5.8)
    heading(ax, "The five building blocks of a clear prompt",
            "Use only the blocks a task needs: a simple question may need just the task")
    card_row(ax, [
        ("1", "Role", "Who should answer?", "Act as an experienced\nPython developer.",
         NAVY, PALE),
        ("2", "Task", "What must be done?", "Explain how\nREST APIs work.", BLUE, MIST),
        ("3", "Context", "What is the situation?", "I know basic Python\nbut am new to APIs.",
         BLUE, PALE),
        ("4", "Constraints", "What are the limits?", "Beginner-friendly,\nunder 300 words.",
         AMBER, AMBER_FACE),
        ("5", "Output format", "What shape is the answer?", "Analogy, table, and\na Python example.",
         GREEN, GREEN_FACE),
    ], ("ANSWERS", "EXAMPLE"))
    return fig


def api_request_flow() -> plt.Figure:
    fig, ax = canvas(14, 6.2)
    heading(ax, "What happens when your code calls an LLM API",
            "The SDK wraps an authenticated HTTPS request; the model runs on the provider's servers")
    box(ax, 4, 44, 15, 18, "Your Python app", "prompt + settings", face=WHITE, edge=NAVY)
    box(ax, 27, 44, 15, 18, "SDK client", "openai / anthropic")
    box(ax, 51, 44, 17, 18, "Provider API", "auth, rate limits,\nbilling")
    box(ax, 77, 44, 17, 18, "Model", "generates tokens", face=MIST)
    box(ax, 26, 70, 17, 9, "API key (env var)", "", face=AMBER_FACE, edge=AMBER,
        title_color=AMBER)
    arrow(ax, (34.5, 69.5), (34.5, 62.5), color=AMBER)
    arrow(ax, (19.5, 53), (26.5, 53))
    arrow(ax, (42.5, 53), (50.5, 53), label="HTTPS")
    arrow(ax, (68.5, 53), (76.5, 53))
    arrow(ax, (85.5, 43.5), (11.5, 43.5), curve=-0.18, color=GREEN)
    note(ax, 49, 25, "Response travels back through the API and SDK: "
         "generated text, tool-call requests, token usage", color=GREEN, size=9)
    ax.text(14, 13, "OpenAI:     client.responses.create(model, instructions, input)"
            "  ->  response.output_text", ha="left", fontsize=8.6, color=INK, **MONO)
    ax.text(14, 6, "Anthropic:  client.messages.create(model, max_tokens, system, messages)"
            "  ->  message.content[0].text", ha="left", fontsize=8.6, color=INK, **MONO)
    return fig


def provider_landscape() -> plt.Figure:
    fig, ax = canvas(14, 6)
    heading(ax, "Three ways to put a model in your application",
            "Two hosted model providers and one open ecosystem of models from many publishers")
    card_row(ax, [
        ("OA", "OpenAI", "Its own models,\nserved by a hosted API",
         "SDK: openai\nResponses API\nOPENAI_API_KEY", NAVY, PALE),
        ("A", "Anthropic", "Claude models,\nserved by a hosted API",
         "SDK: anthropic\nMessages API\nANTHROPIC_API_KEY", BLUE, MIST),
        ("HF", "Hugging Face", "A hub of models, datasets\nand tools from many publishers",
         "transformers pipeline()\nrun locally or hosted\ncheck license + hardware",
         AMBER, AMBER_FACE),
    ], ("WHAT IT IS", "HOW YOU USE IT"), gap=5)
    return fig


def rag_vs_finetuning() -> plt.Figure:
    fig, ax = canvas(14, 6.4)
    heading(ax, "RAG versus fine-tuning",
            "RAG changes what the model sees; fine-tuning changes the model itself")
    lanes = [
        (80, 60, "RAG: GIVE THE MODEL INFORMATION  (weights unchanged)", BLUE, PALE,
         ["Company docs", "Retrieve passages", "Add to prompt", "Generate answer"],
         "Best when knowledge is\nprivate, changing, or large"),
        (44, 24, "FINE-TUNING: CHANGE THE MODEL  (weights updated)", AMBER, AMBER_FACE,
         ["Labeled examples", "Further training", "Adapted model", "Generate answer"],
         "Best when behavior, style,\nor format must change"),
    ]
    xs = [3, 22, 41, 60]
    for label_y, box_y, label, color, face, steps, best_when in lanes:
        caps(ax, 3, label_y, label, color=color, ha="left", size=9)
        for index, (x, step) in enumerate(zip(xs, steps)):
            box(ax, x, box_y, 15, 11, step, face=face, edge=color, title_color=NAVY)
            if index < len(xs) - 1:
                arrow(ax, (x + 15.5, box_y + 5.5), (xs[index + 1] - 0.5, box_y + 5.5),
                      color=color)
        note(ax, 89, box_y + 5.5, best_when, size=8.8)
    note(ax, 50, 8, "The two are not rivals: many systems fine-tune for behavior and "
         "use RAG for knowledge.", color=GREY, size=8.8)
    return fig


def doc_qa_assistant() -> plt.Figure:
    fig, ax = canvas(14, 6)
    heading(ax, "Putting it together: a document Q&A assistant",
            "Retrieval, prompt engineering, and an LLM API combine into one request path")
    box(ax, 2, 50, 12, 15, "User", "\"How does our\nauth service work?\"", face=WHITE,
        edge=NAVY)
    box(ax, 21, 50, 14, 15, "Retriever", "search by meaning\n(embeddings)")
    box(ax, 21, 16, 14, 14, "Document store", "architecture docs", face=MIST)
    box(ax, 42, 50, 15, 15, "Prompt builder", "question + passages\n+ instructions")
    box(ax, 64, 50, 13, 15, "LLM API", "OpenAI or Claude", face=MIST)
    box(ax, 84, 50, 14, 15, "Grounded answer", "summary + cited\nsources",
        face=GREEN_FACE, edge=GREEN, title_color=GREEN)
    arrow(ax, (14.5, 57.5), (20.5, 57.5))
    arrow(ax, (25, 49.5), (25, 30.5), label="query")
    arrow(ax, (31, 30.5), (31, 49.5), color=GREEN, label="passages")
    arrow(ax, (35.5, 57.5), (41.5, 57.5))
    arrow(ax, (57.5, 57.5), (63.5, 57.5))
    arrow(ax, (77.5, 57.5), (83.5, 57.5), color=GREEN)
    ax.plot([21, 57], [76, 76], color=BLUE, linewidth=2)
    caps(ax, 39, 80, "RAG  (Section 9)", color=BLUE)
    ax.plot([64, 77], [76, 76], color=NAVY, linewidth=2)
    caps(ax, 70.5, 80, "API CALL  (Sections 5-6)", color=NAVY)
    note(ax, 70, 22, "Prompt engineering decides how the passages,\nthe question, and the "
         "rules are laid out\n(Sections 4 and 7).", size=8.8)
    return fig


FIGURES = {
    "llm_inference_pipeline": llm_inference_pipeline,
    "llm_evolution": llm_evolution,
    "llm_training_stages": llm_training_stages,
    "prompt_anatomy": prompt_anatomy,
    "api_request_flow": api_request_flow,
    "provider_landscape": provider_landscape,
    "rag_vs_finetuning": rag_vs_finetuning,
    "doc_qa_assistant": doc_qa_assistant,
}


if __name__ == "__main__":
    render(FIGURES, TOPIC, sys.argv[1:])
