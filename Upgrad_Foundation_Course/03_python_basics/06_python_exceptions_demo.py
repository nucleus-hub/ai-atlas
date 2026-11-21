"""Simple examples of handling exceptions in Python."""


def demonstrate_basic_exception(num1, num2):
    """Divide two numbers and handle division by zero."""
    print(f"--- Dividing {num1} by {num2} ---")

    try:
        result = num1 / num2
        print(f"Success! Result is: {result}")
    except ZeroDivisionError:
        print("Error: You cannot divide a number by zero!")


def demonstrate_multiple_exceptions(user_input):
    """Process user input and handle several possible exceptions."""
    print(f"\n--- Processing input: {user_input} ---")

    try:
        # Converting nonnumeric text raises ValueError.
        number = int(user_input)

        # Dividing by zero raises ZeroDivisionError.
        final_value = 100 / number
        print(f"Calculation output: {final_value}")
    except ValueError:
        print("Error: That is not a valid number! Please enter digits only.")
    except ZeroDivisionError:
        print("Error: Entered number zero caused a division crash.")
    except Exception as error:
        # Report any other unexpected exception without stopping the program.
        print(f"An unexpected error occurred: {error}")
    finally:
        # The finally block runs whether an exception occurs or not.
        print("Cleanup: Finished exception verification check.")


# Run each demonstration when this file is executed directly.
if __name__ == "__main__":
    # Step 1: handle a specific exception
    demonstrate_basic_exception(10, 2)
    demonstrate_basic_exception(10, 0)

    # Step 2: handle valid, nonnumeric, zero, and unexpected inputs
    demonstrate_multiple_exceptions("25")
    demonstrate_multiple_exceptions("abc")
    demonstrate_multiple_exceptions("0")
    demonstrate_multiple_exceptions(None)
