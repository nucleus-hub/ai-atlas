"""Simple examples of Python control flow and loops."""


def demonstrate_if_elif_else(score):
    """Print a grade based on a numerical score using if, elif, and else."""
    print(f"--- Checking score: {score} ---")

    # Check thresholds from highest to lowest so only one grade is selected.
    if score >= 90:
        print("Grade: A (Excellent work!)")
    elif score >= 80:
        print("Grade: B (Good job!)")
    elif score >= 70:
        print("Grade: C (Fair)")
    else:
        print("Grade: F (Needs improvement)")


def demonstrate_match_case(day_name):
    """Identify a day as a weekday or weekend using match-case."""
    print(f"\n--- Checking day: {day_name} ---")

    # Normalize the input, then use | to match any day in each group.
    match day_name.lower():
        case "monday" | "tuesday" | "wednesday" | "thursday" | "friday":
            print("It's a weekday. Time to work!")
        case "saturday" | "sunday":
            print("It's the weekend! Time to relax.")
        case _:
            print("Invalid day entered.")


def demonstrate_while_loop(countdown_start):
    """Count down to zero to demonstrate a while loop."""
    print(f"\n--- Starting While Loop Countdown from {countdown_start} ---")

    count = countdown_start

    # Decreasing count on each iteration ensures the loop terminates.
    while count > 0:
        print(f"Countdown: {count}")
        count -= 1

    print("Blast off! The while loop has finished.")


def demonstrate_for_loops():
    """Demonstrate for loops by iterating over a range and a list."""
    print("\n--- Starting For Loops ---")

    # The stop value in range is exclusive, so this produces 1, 2, and 3.
    print("Example A: Counting from 1 to 3 using range()")
    for number in range(1, 4):
        print(f"Number: {number}")

    print()

    print("Example B: Iterating through a Python list")
    fruits = ["apple", "banana", "cherry"]
    for fruit in fruits:
        print(f"I like eating {fruit}")


# Run each demonstration when this file is executed directly.
if __name__ == "__main__":
    # Step 1: if, elif, and else
    demonstrate_if_elif_else(95)
    demonstrate_if_elif_else(82)
    demonstrate_if_elif_else(50)

    # Step 2: match-case
    demonstrate_match_case("Monday")
    demonstrate_match_case("Saturday")
    demonstrate_match_case("Holiday")

    # Step 3: while loop
    demonstrate_while_loop(5)

    # Step 4: for loops
    demonstrate_for_loops()