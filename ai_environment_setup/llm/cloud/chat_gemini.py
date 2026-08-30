"""A minimal streaming chat loop against the Gemini API."""
import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL = "gemini-2.0-flash"  # small, fast, inexpensive — right for learning
SYSTEM = "You are a concise, helpful assistant. Keep answers short."

def main() -> None:
    load_dotenv()
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("GOOGLE_API_KEY is not set. Copy .env.example to .env and fill it in.")

    client = genai.Client(api_key=key)
    
    # Initialize a clean chat session with system instructions
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        max_output_tokens=400, # a ceiling on the expensive half of the bill
    )
    chat = client.chats.create(model=MODEL, config=config)

    print(f"Talking to {MODEL}. /reset to start over, /quit to leave.\n")

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
            chat = client.chats.create(model=MODEL, config=config)
            print("(conversation cleared)\n")
            continue

        print("bot > ", end="", flush=True)
        
        # Send message and stream response chunks back
        response_stream = chat.send_message_stream(user)
        
        for chunk in response_stream:
            if chunk.text:
                print(chunk.text, end="", flush=True)
                
        print("\n")

if __name__ == "__main__":
    main()
