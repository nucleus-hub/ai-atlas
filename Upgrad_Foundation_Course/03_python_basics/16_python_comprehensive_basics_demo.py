# basics/python_comprehensive_basics_demo.py

import os
import logging
import threading
import requests
from dotenv import load_dotenv

# =====================================================================
# CONCEPT 1: Professional Logging Setup
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("master_system_logs.log")
    ]
)

# =====================================================================
# CONCEPT 2: Custom Exception Definition
# =====================================================================
class AgentInitializationError(Exception):
    """Custom exception raised when an agent lacks proper operational credentials."""
    pass

# =====================================================================
# CONCEPT 3: Thread Synchronization Lock & Data Collections
# =====================================================================
file_write_lock = threading.Lock()

# Collections Matrix: Lists, Dicts, Tuples, and Sets working together
completed_agent_names = []  # List (Ordered tracker)
unique_topics_processed = set()  # Set (Strict duplicate removal)
final_system_snapshot = ()  # Tuple (Data integrity snapshot)


# =====================================================================
# CONCEPT 4: Object-Oriented Programming (Parent Class Blueprint)
# =====================================================================
class BaseAgent:
    def __init__(self, agent_name, model_type):
        self.name = agent_name
        self.model = model_type
        # Dictionary Collection mapping properties
        self.config = {"status": "offline", "queries_run": 0}

    def announce(self):
        return f"Agent '{self.name}' built on top of target: {self.model}."


# =====================================================================
# CONCEPT 5: OOP Inheritance (Child Class Subclassing)
# =====================================================================
class ResearchAIAgent(BaseAgent):
    def __init__(self, agent_name, model_type, field_expertise):
        # CONCEPT 6: super().__init__() Construction Bridge
        super().__init__(agent_name, model_type)
        self.field = field_expertise
        self.config["status"] = "ready"

    # CONCEPT 7: Method Overriding (Polymorphism)
    def announce(self):
        return f"🔬 [Specialized] {self.name} is a designated Expert in {self.field} leveraging {self.model}."


# =====================================================================
# CONCEPT 8: Complex Function Arguments (*args and **kwargs)
# =====================================================================
def audit_agent_parameters(agent_instance, *metadata_tags, **execution_flags):
    """
    *metadata_tags packs variable positional arguments into a Tuple.
    **execution_flags packs variable keyword arguments into a Dictionary.
    """
    logging.info(f"Auditing hyper-parameters for {agent_instance.name}...")

    # Loop Collection matching tags
    for tag in metadata_tags:
        logging.info(f" -> Applying compliance label: #{tag}")

    temp_setting = execution_flags.get("temperature", 0.7)
    logging.info(f" -> Verified execution creativity temperature metric: {temp_setting}")


# =====================================================================
# CONCEPT 9: Concurrent Live External API Execution
# =====================================================================
def run_live_gemini_agent_task(agent, prompt, api_key):
    """
    Target worker function deployed inside concurrent background threads.
    Handles network requests, exception sandboxes, file streams, and lock controls.
    """
    # CONCEPT 10: Control Flow (If-Elif-Else matching)
    if not api_key:
        # Raising our custom runtime exception inside validation
        raise AgentInitializationError(f"Agent '{agent.name}' structural error: GOOGLE_API_KEY is null.")
    elif len(prompt) == 0:
        logging.warning(f"[{agent.name}] Target execution prompt cannot be completely blank.")
        return
    else:
        logging.info(f"[{agent.name}] Preparing outbound transaction payload...")

    # CONCEPT 11: Comprehensive Try-Catch Exception Blocks
    try:
        # Use the same working Gemini endpoint and authentication style as the
        # standalone API demonstration in 08_python_api_demo.py.
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-3.6-flash:generateContent?key={api_key}"
        )
        custom_headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        # Enforcing a 10-second request timeout limit to prevent forever hanging sockets
        response = requests.post(url, json=payload, headers=custom_headers, timeout=10)

        logging.info(f"[{agent.name}] Received Server response with HTTP Code: {response.status_code}")
        response.raise_for_status()  # Automatically jumps to HTTPError block if code is 4xx or 5xx

        # candidates and parts are JSON arrays, so select their first entries.
        response_json = response.json()
        ai_raw_text = response_json["candidates"][0]["content"]["parts"][0]["text"]

        # CONCEPT 12: String Manipulation & Slicing
        clean_reply = ai_raw_text.strip().replace("\n", " ")
        # Slicing token string sequences cleanly to block key logs leaking to disk reports
        masked_secret_key = api_key[:10] + "****************"

        # CONCEPT 13: Repetition Control (While Loop tracking progress down steps)
        write_attempts = 1
        while write_attempts > 0:

            # Thread Synchronization Lock: Stops background workers from mixing rows in the file
            file_write_lock.acquire()
            try:
                # CONCEPT 14: Local File Handling with clean automatic Context Managers
                with open("agent_transaction_ledger.txt", "a", encoding="utf-8") as ledger:
                    ledger.write(f"Agent: {agent.name} | MaskedKey: {masked_secret_key} | Response Summary: {clean_reply[:80]}...\n")

                # Adding data safely into our memory data collections matrix trackers
                completed_agent_names.append(agent.name)
                unique_topics_processed.add(agent.field)

            finally:
                # Always release keys safely inside a finally block to prevent Deadlocks
                file_write_lock.release()

            write_attempts -= 1

        logging.info(f"[{agent.name}] Lifecycle transaction wrapped up safely.")

    # Catching specific network-related exception profiles
    except requests.exceptions.Timeout:
        logging.error(f"[{agent.name}] Network transaction aborted. Google's server timed out.")
    except requests.exceptions.HTTPError as http_err:
        # Do not log the exception's URL because its query string contains the API key.
        status_code = (
            http_err.response.status_code
            if http_err.response is not None
            else "unknown"
        )
        logging.error(f"[{agent.name}] Target connection rejected with HTTP status: {status_code}")
    except Exception as unexpected_err:
        logging.critical(f"[{agent.name}] Script execution runtime block exception crash: {unexpected_err}")


# =====================================================================
# CONCEPT 15: Main Execution Engine Block & Module Guards
# =====================================================================
if __name__ == "__main__":
    logging.info("==============================================")
    logging.info("Starting Comprehensive Demo System Initialization...")
    logging.info("==============================================")

    # 1. Environment Parsing Configuration variables
    load_dotenv()
    secure_google_key = os.getenv("GOOGLE_API_KEY")

    # 2. Object Instantiations
    researcher_alpha = ResearchAIAgent("GenomicsBot", "gemini-3.6-flash", "Bioinformatics")
    researcher_beta = ResearchAIAgent("AstrophysicsBot", "gemini-3.6-flash", "Cosmology")

    print(researcher_alpha.announce())
    print(researcher_beta.announce())

    # 3. Dynamic parameter evaluations
    audit_agent_parameters(researcher_alpha, "tier-1", "async-capable", temperature=0.2, maximum_tokens=150)

    # 4. Lambda Function single line shortcut definitions
    generate_prompt_template = lambda agent, query: f"Explain {query} relative to {agent.field} in one super short sentence."

    prompt_a = generate_prompt_template(researcher_alpha, "CRISPR gene drive sequences")
    prompt_b = generate_prompt_template(researcher_beta, "Dark matter string density anomalies")

    # 5. Spawning Multi-Threaded Concurrency Background Workers
    thread_worker_1 = threading.Thread(target=run_live_gemini_agent_task, args=(researcher_alpha, prompt_a, secure_google_key), name="Worker-Alpha")
    thread_worker_2 = threading.Thread(target=run_live_gemini_agent_task, args=(researcher_beta, prompt_b, secure_google_key), name="Worker-Beta")

    # Launching both background thread workers side-by-side simultaneously
    thread_worker_1.start()
    thread_worker_2.start()

    logging.info("The primary Main-Thread UI loop stays active and responsive while network calls run!")

    # Merge background processes gracefully back into main timeline execution blocks
    thread_worker_1.join()
    thread_worker_2.join()

    # 6. Advanced List Comprehensions looping data maps
    system_execution_manifest = [f"Processed Successfully by: {name.upper()}" for name in completed_agent_names]

    print("\n--- Framework Aggregated Execution Output Matrix ---")
    print(f"List Comprehension Output Rows: {system_execution_manifest}")
    print(f"Unique Fields Map Set Collection:  {unique_topics_processed}")

    # Locking final statistics tracking records safely into an unchangeable Tuple
    final_system_snapshot = (len(completed_agent_names), "Execution Complete")
    print(f"Immutable System Tuple Status Snapshot:  {final_system_snapshot}")

