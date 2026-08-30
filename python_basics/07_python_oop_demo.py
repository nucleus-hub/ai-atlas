"""Simple examples of Python classes, inheritance, and polymorphism."""


class AIModelBot:
    """Represent a simple AI assistant with its own chat history."""

    def __init__(self, bot_name, language_model):
        """Initialize a bot with a name, model, and empty chat history."""
        # Each object receives its own attributes and history list.
        self.name = bot_name
        self.model = language_model
        self.chat_history = []

    def generate_response(self, user_message):
        """Generate a simulated reply and record both messages."""
        self.chat_history.append(f"User: {user_message}")

        # Build the reply using this object's name and language model.
        bot_reply = (
            f"[{self.name} processing via {self.model}]: "
            f"I received your message: '{user_message}'"
        )

        self.chat_history.append(f"Bot: {bot_reply}")
        return bot_reply

    def show_history(self):
        """Display the conversation history for this bot instance."""
        print(f"\n--- Chat History Logs for {self.name} ---")
        for log in self.chat_history:
            print(log)


class CodingBot(AIModelBot):
    """Represent a specialized AI bot for a programming language."""

    def __init__(self, bot_name, language_model, primary_language):
        """Initialize inherited bot data and a preferred language."""
        # Reuse the parent constructor before adding subclass-specific data.
        super().__init__(bot_name, language_model)
        self.fav_language = primary_language

    def generate_response(self, user_message):
        """Override the parent method with a coding-specific response."""
        self.chat_history.append(f"User: {user_message}")

        bot_reply = (
            f"[{self.name} - {self.fav_language} Expert]: "
            f"Here is your optimized code block for: '{user_message}'\n"
            f"```python\n"
            f"# Written by {self.name} via {self.model}\n"
            f"print('Hello World')\n"
            f"```"
        )

        self.chat_history.append(f"Bot: {bot_reply}")
        return bot_reply

    def review_syntax(self):
        """Display a syntax review message for the preferred language."""
        print(
            f"🔧 {self.name} is now verifying syntax rules for "
            f"{self.fav_language}..."
        )


# Run the object-oriented demonstration when this file is executed directly.
if __name__ == "__main__":
    # Step 1: create two independent object instances
    print("--- Creating Object Instances from the Blueprint ---")

    # Python passes each new instance as self automatically.
    flash_bot = AIModelBot(bot_name="Flashy", language_model="Gemini-Flash")
    pro_bot = AIModelBot(bot_name="ProBot", language_model="Gemini-Pro")

    # Step 2: access object attributes
    print(f"Bot 1 is named: {flash_bot.name}")
    print(f"Bot 2 is named: {pro_bot.name}")

    # Step 3: call methods on each object
    print("\n--- Interacting with the Objects ---")

    reply1 = flash_bot.generate_response("Hello! What is Python?")
    print(reply1)

    flash_bot.generate_response("Got it, thanks.")

    reply2 = pro_bot.generate_response("Can you run a simulation?")
    print(reply2)

    # Step 4: display each object's independent history
    flash_bot.show_history()
    pro_bot.show_history()

    # Step 5: compare parent and child class behavior
    print("\n--- 1. Testing Parent Class Objects ---")
    parent_bot = AIModelBot(bot_name="Flashy", language_model="Gemini-Flash")
    print(parent_bot.generate_response("What is Python?"))

    print("\n--- 2. Testing Inherited Child Class Object ---")
    coder = CodingBot(
        bot_name="DevBot",
        language_model="Gemini-Pro",
        primary_language="Python",
    )

    # The child object has attributes initialized by the parent class.
    print(f"Created bot {coder.name} running on model {coder.model}")

    # Call the method overridden by CodingBot.
    print("\nRunning overridden method:")
    print(coder.generate_response("Write a simple print script"))

    # Call a method available only on CodingBot.
    print("\nRunning unique subclass method:")
    coder.review_syntax()

    # Reuse show_history() inherited unchanged from AIModelBot.
    coder.show_history()
