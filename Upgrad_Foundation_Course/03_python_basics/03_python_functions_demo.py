"""Simple examples of Python functions, parameters, and return values."""


def calculate_total_price(price, quantity):
    """Calculate and return the total price for a quantity of items."""
    total = price * quantity
    return total


def greet_user(username, greeting="Hello"):
    """Return a personalized greeting with an optional greeting word."""
    return f"{greeting}, {username}!"


def get_min_and_max(numbers_list):
    """Return the lowest and highest values from a list of numbers."""
    lowest = min(numbers_list)
    highest = max(numbers_list)
    return lowest, highest


def configure_bot(name, model_type, region="US-Central", is_active=True):
    """Display bot settings provided through different parameter types."""
    print(
        f"Bot Name: {name:<8} | Model: {model_type:<13} | "
        f"Region: {region:<10} | Active: {is_active}"
    )


# Run each demonstration when this file is executed directly.
if __name__ == "__main__":
    # Step 1: arguments and return values
    print("--- 1. Testing Arguments & Return Values ---")
    receipt_total = calculate_total_price(15.50, 3)
    print(f"Total Bill: ${receipt_total}")

    # Step 2: default and keyword arguments
    print("\n--- 2. Testing Default Arguments ---")
    print(greet_user("Alice"))
    print(greet_user("Bob", greeting="Welcome"))
    print(greet_user("Bob", "Somehow"))

    # Step 3: multiple return values and tuple unpacking
    print("\n--- 3. Testing Multiple Return Values ---")
    scores = [78, 92, 45, 88, 99, 61]
    lowest_score, highest_score = get_min_and_max(scores)
    print(f"Lowest Score: {lowest_score}")
    print(f"Highest Score: {highest_score}")

    # Step 4: positional and keyword arguments
    print("\n--- 4. Deep Dive: Positional vs. Keyword Arguments ---")

    # Positional arguments are assigned in the order they are provided.
    print("Example A (Positional - Correct Order):")
    configure_bot("Chatty", "Gemini-Flash")

    print("Example A (Positional - Swapped Order Mix-up):")
    configure_bot("Gemini-Flash", "Chatty")

    # Keyword arguments are matched by name, so their order does not matter.
    print("\nExample B (Keywords - Swapped Order behaves perfectly):")
    configure_bot(model_type="Gemini-Flash", name="Chatty")

    # Positional arguments must come before keyword arguments when mixed.
    print("\nExample C (Mixing both types):")
    configure_bot("HelperBot", "Claude-3", is_active=False)
