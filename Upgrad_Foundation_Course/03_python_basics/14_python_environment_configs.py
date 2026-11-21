"""Demonstrate loading application configuration from environment variables."""

import os

from dotenv import load_dotenv


def demonstrate_environment_loading():
    """Load secrets and typed application settings from the environment."""
    print("--- Environment Variable Configuration ---")

    # Load key-value pairs from a nearby .env file into the process environment.
    # Existing environment variables are not overwritten unless override=True.
    load_dotenv()

    # getenv() returns the variable's string value, or None when it is undefined.
    # Keep secrets in environment variables instead of hard-coding them in source.
    api_credential = os.getenv("GOOGLE_API_KEY")
    token_credential = os.getenv("HF_TOKEN")

    # Optional settings can provide defaults. Environment values always begin
    # as strings, so applications must explicitly convert Booleans and numbers.
    app_environment = os.getenv("APP_ENV", "development")

    debug_text = os.getenv("DEBUG", "false").strip().lower()
    if debug_text in {"1", "true", "yes", "on"}:
        debug_enabled = True
    elif debug_text in {"0", "false", "no", "off"}:
        debug_enabled = False
    else:
        debug_enabled = False
        print(f"Warning: Invalid DEBUG value {debug_text!r}; using False.")

    timeout_text = os.getenv("REQUEST_TIMEOUT_SECONDS", "30")
    try:
        request_timeout = float(timeout_text)
        if request_timeout <= 0:
            raise ValueError
    except ValueError:
        request_timeout = 30.0
        print(
            f"Warning: Invalid REQUEST_TIMEOUT_SECONDS value {timeout_text!r}; "
            "using 30.0 seconds."
        )

    print("\nApplication settings (safe to display):")
    print(f"  Environment: {app_environment}")
    print(f"  Debug enabled: {debug_enabled}")
    print(f"  Request timeout: {request_timeout:.1f} seconds")

    # Check configuration availability without printing any part of a secret.
    print("\nCredential availability (values remain hidden):")
    if api_credential:
        print("Success: GOOGLE_API_KEY is configured (value hidden).")
    else:
        print(
            "Warning: GOOGLE_API_KEY was not found in the .env file "
            "or process environment."
        )

    if token_credential:
        print("Success: HF_TOKEN is configured (value hidden).")
    else:
        print(
            "Warning: HF_TOKEN was not found in the .env file "
            "or process environment."
        )


if __name__ == "__main__":
    demonstrate_environment_loading()
