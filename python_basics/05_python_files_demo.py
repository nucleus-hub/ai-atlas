"""Simple examples of writing to and reading from a text file."""


def write_to_file(filename):
    """Write sample text to a file, replacing any existing content."""
    print(f"--- Writing data to {filename} ---")

    # Write mode creates the file or overwrites it if it already exists.
    with open(filename, "w") as file:
        file.write("Line 1: Hello from Python AI starter kit.\n")
        file.write("Line 2: Learning file handling is crucial.\n")
        file.write("Line 3: This data is stored locally on disk.\n")

    print("File successfully created and written.")


def read_from_file(filename):
    """Read and display a text file one line at a time."""
    print(f"\n--- Reading data from {filename} ---")

    try:
        # Iterate over the file and remove each line's trailing newline.
        with open(filename, "r") as file:
            for line in file:
                print(f"Read row: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: The file {filename} does not exist.")


# Run each demonstration when this file is executed directly.
if __name__ == "__main__":
    target_file = "learning_notes.txt"

    # Step 1: create and write the file
    write_to_file(target_file)

    # Step 2: read the file
    read_from_file(target_file)
