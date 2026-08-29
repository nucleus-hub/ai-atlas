"""Simple examples of Python lists, dictionaries, tuples, and sets."""

# Python Collections Reference
#
# | Type       | Ordered? | Mutable? | Allows duplicates?    | Syntax       |
# |------------|----------|----------|-----------------------|--------------|
# | List       | Yes      | Yes      | Yes                   | [item, ...]  |
# | Tuple      | Yes      | No       | Yes                   | (item, ...)  |
# | Set        | No       | Yes      | No                    | {item, ...}  |
# | Dictionary | Yes*     | Yes      | Keys: No; Values: Yes | {key: value} |
#
# Key features:
# - List: Stores general-purpose sequences of data.
# - Tuple: Protects data from accidental modification.
# - Set: Provides fast membership checks and removes duplicates.
# - Dictionary: Maps unique keys to descriptive values.
#
# * Dictionaries preserve insertion order in Python 3.7 and later.


def demonstrate_lists():
    """Demonstrate indexed access and common list operations."""
    print("--- 1. Python Lists Demo ---")

    # Create a list of programming languages.
    programming_languages = ["Java", "JavaScript", "C++"]
    print(f"Initial list: {programming_languages}")

    # Access the first item using index 0.
    print(f"First language: {programming_languages[0]}")

    # Add an item to the end of the list.
    programming_languages.append("Python")
    print(f"After adding Python: {programming_languages}")

    # Replace an item at a specific index.
    programming_languages[2] = "C#"
    print(f"After changing C++ to C#: {programming_languages}")

    # Remove an item by its value.
    programming_languages.remove("JavaScript")
    print(f"After removing JavaScript: {programming_languages}")


def demonstrate_dictionaries():
    """Demonstrate dictionary lookup, insertion, updating, and safe access."""
    print("\n--- 2. Python Dictionaries Demo ---")

    llm_model = {
        "name": "Gemini 1.5 Flash",
        "developer": "Google",
        "context_window": 1000000,
        "is_multimodal": True,
    }
    print(f"Initial dictionary: {llm_model}")

    print(f"Model Name: {llm_model['name']}")
    print(f"Developer: {llm_model['developer']}")

    # Key assignment adds a missing key or updates an existing one.
    llm_model["release_year"] = 2024
    llm_model["context_window"] = 2000000
    print(f"Updated dictionary: {llm_model}")

    # get() returns the fallback value when the requested key is missing.
    pricing = llm_model.get("cost_per_token", "Price not listed")
    print(f"Token pricing look-up: {pricing}")


def demonstrate_tuples_and_sets():
    """Demonstrate immutable tuples and sets of unique items."""
    print("\n--- 3. Python Tuples Demo ---")

    # Create a tuple to store a fixed pair of coordinates.
    api_server_coordinates = (37.7749, -122.4194)
    print(f"Server Location Coordinates: {api_server_coordinates}")

    # Access a tuple item by its index.
    print(f"Latitude only: {api_server_coordinates[0]}")

    # Tuple items cannot be changed after the tuple is created.

    print("\n--- 4. Python Sets Demo ---")

    # Create a set of unique roles.
    allowed_roles = {"admin", "developer", "user"}
    print(f"Initial set of roles: {allowed_roles}")

    # Add a new item to the set.
    allowed_roles.add("guest")
    print(f"After adding 'guest': {allowed_roles}")

    # Adding a duplicate leaves the set unchanged.
    allowed_roles.add("admin")
    print(f"After trying to add 'admin' again: {allowed_roles}")

    # Check whether a value exists in the set.
    is_developer = "developer" in allowed_roles
    print(f"Is 'developer' in the allowed roles? {is_developer}")


# Run each demonstration when this file is executed directly.
if __name__ == "__main__":
    # Step 1: lists
    demonstrate_lists()

    # Step 2: dictionaries
    demonstrate_dictionaries()

    # Step 3: tuples and sets
    demonstrate_tuples_and_sets()