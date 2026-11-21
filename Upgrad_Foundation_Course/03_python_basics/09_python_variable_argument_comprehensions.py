"""Examples of comprehensions, unpacking, *args, and **kwargs."""


def demonstrate_list_comprehensions():
    """Build and filter lists using loops and list comprehensions."""
    print("--- 1. List Comprehensions ---")

    prices = [10, 20, 30, 40]

    # Build a discounted list with a traditional loop.
    discounted_prices = []
    for price in prices:
        discounted_prices.append(price * 0.9)
    print(f"Traditional Loop Result:  {discounted_prices}")

    # Produce the same result with a concise list comprehension.
    comprehension_prices = [price * 0.9 for price in prices]
    print(f"Comprehension Result:    {comprehension_prices}")

    # Add a condition to include only matching items.
    expensive_only = [price for price in prices if price > 25]
    print(f"Filtered (Price > 25):   {expensive_only}")


def demonstrate_unpacking_operators():
    """Demonstrate iterable and dictionary unpacking with * and **."""
    print("\n--- 2. Unpacking Operators (* and **) ---")

    # * expands a list into separate positional arguments.
    coordinates = [12.97, 77.59]
    print(f"Raw list print: {coordinates}")
    print("Unpacked print:", *coordinates)

    def render_bot_config(name, model, temperature):
        print(f"Configuring {name} ({model}) with temp: {temperature}")

    settings = {
        "name": "OmniBot",
        "model": "gemini-2.5-flash",
        "temperature": 0.2,
    }

    # ** maps dictionary entries to matching keyword parameters.
    render_bot_config(**settings)


def send_group_message(sender, *recipients):
    """Send a message to any number of recipients using *args."""
    print(f"\nMessage Sender: {sender}")

    # *recipients collects extra positional arguments into a tuple.
    print(f"Recipients Tuple Structure: {recipients}")

    for person in recipients:
        print(f" -> Sending notification payload to: {person}")


def initialize_llm_agent(agent_name, **model_configurations):
    """Initialize an agent with flexible model settings using **kwargs."""
    print(f"\nInitializing Agent: {agent_name}")

    # **model_configurations collects keyword arguments into a dictionary.
    print(f"Configuration Dictionary Structure: {model_configurations}")

    # get() supplies a default when temperature is not provided.
    temperature = model_configurations.get("temperature", 0.7)
    print(f"Extracted execution temperature: {temperature}")


# Run each demonstration when this file is executed directly.
if __name__ == "__main__":
    # Step 1: list comprehensions
    demonstrate_list_comprehensions()

    # Step 2: unpacking operators
    demonstrate_unpacking_operators()

    # Step 3: variable positional arguments
    print("\n--- 3. Testing *args (Variable Positional Arguments) ---")
    send_group_message("SystemAdmin", "alice@test.com", "bob@test.com")
    send_group_message(
        "SystemAdmin",
        "user1@test.com",
        "user2@test.com",
        "user3@test.com",
    )

    # Step 4: variable keyword arguments
    print("\n--- 4. Testing **kwargs (Variable Keyword Arguments) ---")
    initialize_llm_agent(
        "CustomerSupportBot",
        model="gemini-2.5-flash",
        temperature=0.2,
    )
    initialize_llm_agent(
        "CreativeWriterBot",
        top_k=40,
        temperature=0.9,
        presence_penalty=1.5,
    )
