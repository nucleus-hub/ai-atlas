"""Compare sequential and multithreaded execution of I/O-bound tasks."""

import threading
import time


def simulate_slow_network_call(task_name, delay):
    """Simulate waiting for a response from a remote server."""
    current_thread = threading.current_thread().name
    print(f"📡 [{current_thread}] '{task_name}' started...")

    # sleep() represents I/O waiting. During this pause, another thread can run.
    time.sleep(delay)

    print(f"✅ [{current_thread}] '{task_name}' finished!")


if __name__ == "__main__":
    print(f"Main thread: {threading.current_thread().name}")

    # Sequential execution: the second task starts only after the first finishes.
    print("\n--- 1. Sequential Execution ---")
    sequential_start = time.perf_counter()

    simulate_slow_network_call("Fetch Profile", 2)
    simulate_slow_network_call("Load Images", 2)

    sequential_time = time.perf_counter() - sequential_start
    print(f"Sequential time: {sequential_time:.2f} seconds")

    # Multithreaded execution: both tasks can wait for I/O at the same time.
    print("\n--- 2. Multithreaded Execution ---")
    threaded_start = time.perf_counter()

    # target specifies the function; args supplies its positional arguments.
    # Creating a Thread object does not start it.
    worker1 = threading.Thread(
        target=simulate_slow_network_call,
        args=("Fetch Profile", 2),
        name="Network-Worker-1",
    )
    worker2 = threading.Thread(
        target=simulate_slow_network_call,
        args=("Load Images", 2),
        name="Network-Worker-2",
    )

    # start() schedules each target function to run on its worker thread.
    worker1.start()
    worker2.start()

    # The main thread continues immediately while both workers are active.
    print("💬 [MainThread] Both network requests are now in progress.")

    # join() blocks the main thread until the corresponding worker finishes.
    worker1.join()
    worker2.join()

    threaded_time = time.perf_counter() - threaded_start
    print(f"Multithreaded time: {threaded_time:.2f} seconds")

    print("\n--- 3. Comparison ---")
    print(f"Time saved: {sequential_time - threaded_time:.2f} seconds")
    print(
        "Threads improve this example because the tasks spend time waiting for I/O. "
        "They generally do not speed up CPU-bound Python code due to the GIL."
    )
