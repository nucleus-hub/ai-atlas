"""Configure application-wide logging using environment-based settings."""

import logging
import os

from dotenv import load_dotenv


def configure_system_logging():
    """Configure the root logger once for console and file output."""
    # Deployment-specific values belong in .env; handler and formatter details
    # remain in Python. Existing process variables take precedence over .env.
    load_dotenv()
    log_level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    log_file = os.getenv("LOG_FILE", "automated_system_logs.log").strip()

    # Convert a level name such as "INFO" into logging.INFO. Fall back safely
    # instead of failing application startup when configuration is invalid.
    log_level = getattr(logging, log_level_name, None)
    if not isinstance(log_level, int):
        print(f"Warning: Invalid LOG_LEVEL {log_level_name!r}; using INFO.")
        log_level = logging.INFO

    if not log_file:
        print("Warning: LOG_FILE is empty; using 'automated_system_logs.log'.")
        log_file = "automated_system_logs.log"

    # Configure the root logger early in application startup. Modules that use
    # logging.getLogger(__name__) inherit this configuration automatically.
    logging.basicConfig(
        level=log_level,
        # Include the event time, severity, source line, and message in each record.
        format="%(asctime)s [%(levelname)s] (Line: %(lineno)d) - %(message)s",
        handlers=[
            # Display records immediately in the terminal.
            logging.StreamHandler(),
            # Append the same records to a file in the current working directory.
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return log_file


def run_production_simulation():
    """Generate example INFO, WARNING, and ERROR log records."""
    logging.info("System initializing production pipeline operations...")

    try:
        user_input = "not_a_number"
        logging.warning("Validating potentially unstable data: %r", user_input)

        # This intentional invalid conversion demonstrates exception logging.
        int(user_input)

    except ValueError as format_error:
        # exception() emits an ERROR record and includes the full stack trace.
        logging.exception("Conversion failed with traceback: %s", format_error)

        # error() emits an ERROR record without automatically adding a traceback.
        logging.error("Handled conversion failure; using no value: %s", format_error)

    logging.info("Pipeline lifecycle completed cleanly.")


if __name__ == "__main__":
    configured_log_file = configure_system_logging()
    run_production_simulation()

    print("\nNotice: Check the current working directory.")
    print(f"{configured_log_file!r} has been created or updated.")
