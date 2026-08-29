"""Simple examples of custom module imports and lambda functions."""

from importlib import import_module

# Choose the import path based on whether this file runs directly or as a module.
FUNCTIONS_MODULE = (
    f"{__package__}.03_python_functions_demo"
    if __package__
    else "03_python_functions_demo"
)
functions_demo = import_module(FUNCTIONS_MODULE)
greet_user = functions_demo.greet_user


def demonstrate_imports():
    """Call a function imported from another project module."""
    print("--- 1. Custom Module Imports ---")

    # Reuse greet_user() without redefining it in this file.
    borrowed_message = greet_user("Developer", greeting="Welcome back")
    print(f"Result from imported module function: {borrowed_message}")


def demonstrate_lambdas():
    """Compare a regular function with lambdas used for simple operations."""
    print("\n--- 2. Lambda Functions ---")

    # Define the same square operation as a regular function and a lambda.
    def traditional_square(x):
        return x * x

    lambda_square = lambda x: x * x

    print(f"Traditional function output (5^2): {traditional_square(5)}")
    print(f"Lambda function output (5^2):      {lambda_square(5)}")

    bot_models = [
        {"name": "Gemini-Pro", "speed_score": 85},
        {"name": "Gemini-Flash", "speed_score": 98},
        {"name": "Gemini-Ultra", "speed_score": 60},
    ]

    # Use a lambda as the key function for descending speed-score order.
    bot_models.sort(key=lambda model: model["speed_score"], reverse=True)
    print(f"Sorted bots by speed score: {bot_models}")


# Run each demonstration when this file is executed directly.
if __name__ == "__main__":
    # Step 1: custom module import
    demonstrate_imports()

    # Step 2: lambda functions
    demonstrate_lambdas()
