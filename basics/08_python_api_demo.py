"""Simple example of calling a live Gemini API endpoint."""

import os

import requests
from dotenv import load_dotenv


def call_live_gemini_api():
    """Send a prompt to Gemini and display the API response."""
    # Load the API key from the local environment configuration.
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY not found in your .env file!")
        return

    print("--- 1. Environment Status ---")
    print("GOOGLE_API_KEY loaded successfully.")

    # Build the endpoint, headers, and JSON request body.
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3.6-flash:generateContent?key={api_key}"
    )

    custom_headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": "Explain the concept of API in one short sentence."}
                ]
            }
        ]
    }

    print("\n--- 2. Printing Outgoing Request Metadata ---")
    print("Target Endpoint: https://googleapis.com")
    print(f"Headers:         {custom_headers}")
    print(f"Payload Body:    {payload}")

    print("\n--- 3. Executing Live Network Request ---")
    try:
        # A timeout prevents the request from waiting indefinitely.
        response = requests.post(
            url,
            json=payload,
            headers=custom_headers,
            timeout=10,
        )

        print(f"HTTP Status Code: {response.status_code}")

        # Raise HTTPError for unsuccessful 4xx or 5xx responses.
        response.raise_for_status()

        response_json = response.json()
        print("\n--- 4. Full Raw Success Response JSON ---")
        print(response_json)

        # Extract the generated text from Gemini's nested response structure.
        try:
            ai_text_reply = response_json["candidates"][0]["content"]["parts"][0]["text"]
            print("\n--- 5. Extracted AI Reply Text ---")
            print(ai_text_reply.strip())
        except (KeyError, IndexError, TypeError):
            print(
                "\nWarning: The API responded successfully, but the text "
                "structure was unexpected."
            )

    except requests.exceptions.Timeout:
        print(
            "Error: The connection timed out. Google's server took too long "
            "to reply."
        )
    except requests.exceptions.HTTPError as http_error:
        print(f"HTTP Error occurred: {http_error}")
        if http_error.response is not None:
            print(f"Server response details: {http_error.response.text}")
    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not establish a connection to the server. "
            "Verify your internet status."
        )
    except Exception as error:
        print(f"An unexpected script failure occurred: {error}")


# Run the live API demonstration when this file is executed directly.
if __name__ == "__main__":
    call_live_gemini_api()
