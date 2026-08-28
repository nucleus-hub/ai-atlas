"""The same chat loop, against a model running on this machine."""

from threading import Thread

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM = "You are a concise, helpful assistant. Keep answers short."


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU

    return "cpu"


def main() -> None:
    device = pick_device()

    print(
        f"loading {MODEL_ID} on {device} — "
        "first run downloads the model …"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    ).to(device)

    history = [
        {
            "role": "system",
            "content": SYSTEM,
        }
    ]

    print("ready. /reset to start over, /quit to leave.\n")

    while True:
        try:
            user = input("you > ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not user:
            continue

        if user == "/quit":
            print("bye")
            return

        if user == "/reset":
            history = [
                {
                    "role": "system",
                    "content": SYSTEM,
                }
            ]

            print("(conversation cleared)\n")
            continue

        history.append(
            {
                "role": "user",
                "content": user,
            }
        )

        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        thread = Thread(
            target=model.generate,
            kwargs=dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=300,
            ),
        )

        thread.start()

        print("bot > ", end="", flush=True)

        reply = ""

        for piece in streamer:
            print(piece, end="", flush=True)
            reply += piece

        thread.join()

        print("\n")

        history.append(
            {
                "role": "assistant",
                "content": reply,
            }
        )


if __name__ == "__main__":
    main()