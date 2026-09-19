"""Demonstrate concurrent I/O-bound work with Python's asyncio event loop."""

import asyncio
import time


def simulate_sync_api_call(call_id, wait_time):
    """Simulate a blocking API request and return its response data."""
    print(f"🚀 Sync Call-{call_id}: Outbound request triggered...")

    # time.sleep() blocks the current thread, so no other call can progress here.
    time.sleep(wait_time)

    print(f"🎯 Sync Call-{call_id}: Response data received after {wait_time}s!")
    return f"Data Package {call_id}"


async def simulate_async_api_call(call_id, wait_time):
    """Simulate an asynchronous API request and return its response data."""
    print(f"🚀 Async Task-{call_id}: Outbound request triggered...")

    # await pauses only this coroutine while it waits for I/O. Control returns
    # to the event loop, which can then make progress on other ready tasks.
    await asyncio.sleep(wait_time)

    print(f"🎯 Async Task-{call_id}: Response data received after {wait_time}s!")
    return f"Data Package {call_id}"


async def main():
    """Compare sequential blocking calls with concurrent asynchronous calls."""
    print("--- 1. Synchronous (Blocking) Execution ---")
    sync_start = time.perf_counter()

    # Each call blocks until it finishes, so the three waits occur one by one:
    # 3 seconds + 3 seconds + 3 seconds = approximately 9 seconds.
    sync_results = [
        simulate_sync_api_call(1, 3),
        simulate_sync_api_call(2, 3),
        simulate_sync_api_call(3, 3),
    ]

    sync_time = time.perf_counter() - sync_start
    print(f"\nCollected outputs: {sync_results}")
    print(f"Synchronous total run time: {sync_time:.2f} seconds")

    print("\n--- 2. Asynchronous (Concurrent) Execution ---")
    async_start = time.perf_counter()

    # Calling an async function creates a coroutine object. Its body begins
    # running only when the coroutine is awaited or scheduled by the event loop.
    task1 = simulate_async_api_call(1, 3)
    task2 = simulate_async_api_call(2, 3)
    task3 = simulate_async_api_call(3, 3)

    # gather() schedules the coroutines concurrently and waits for all of them.
    # It returns results in input order, even if tasks finish in another order.
    # asyncio normally provides this concurrency on one event-loop thread.
    results = await asyncio.gather(task1, task2, task3)

    async_time = time.perf_counter() - async_start
    print(f"\nCollected outputs: {results}")
    print(f"Asyncio total run time: {async_time:.2f} seconds")

    print("\n--- 3. Comparison ---")
    print(f"Time saved with asyncio: {sync_time - async_time:.2f} seconds")
    print(
        "Synchronous calls wait one after another (about 9 seconds), while "
        "asyncio overlaps the three waits (about 3 seconds)."
    )


if __name__ == "__main__":
    # Create an event loop, run main() until completion, and then close the loop.
    asyncio.run(main())
