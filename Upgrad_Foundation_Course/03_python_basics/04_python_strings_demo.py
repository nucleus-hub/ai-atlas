"""Simple examples of Python string manipulation and slicing."""


def demonstrate_string_manipulation():
    """Demonstrate common operations for modifying and cleaning text."""
    print("--- 1. String Manipulation Demo ---")

    # Start with text that contains extra surrounding spaces.
    raw_input = "   learning python is awesome!   "
    print(f"Original text: '{raw_input}'")

    # Remove whitespace from both ends of the string.
    cleaned_input = raw_input.strip()
    print(f"Stripped text: '{cleaned_input}'")

    # Convert the cleaned text to different letter cases.
    print(f"Uppercase:     '{cleaned_input.upper()}'")
    print(f"Title Case:    '{cleaned_input.title()}'")

    # Replace a matching part of the string.
    replaced_text = cleaned_input.replace("awesome", "fantastic")
    print(f"Replaced text: '{replaced_text}'")

    # Split the string into a list of words.
    words_list = cleaned_input.split(" ")
    print(f"Split into words: {words_list}")


def demonstrate_string_slicing():
    """Demonstrate indexing and extracting text with string slices."""
    print("\n--- 2. String Slicing Demo ---")

    sample_text = "PythonAI"
    print(f"Sample word: '{sample_text}'")

    # String indexes start at zero.
    print(f"First character (index 0): {sample_text[0]}")

    # A slice includes the start index but excludes the stop index.
    print(f"Slice [0:6] (Extract 'Python'): {sample_text[0:6]}")

    # Omitting start begins the slice at the first character.
    print(f"Slice [:6]  (Extract 'Python'): {sample_text[:6]}")

    # Omitting stop continues the slice through the final character.
    print(f"Slice [6:]  (Extract 'AI'):     {sample_text[6:]}")

    # Negative indexes count backward from the end of the string.
    print(f"Last character (index -1):   {sample_text[-1]}")


# Run each demonstration when this file is executed directly.
if __name__ == "__main__":
    # Step 1: string manipulation
    demonstrate_string_manipulation()

    # Step 2: string slicing
    demonstrate_string_slicing()
