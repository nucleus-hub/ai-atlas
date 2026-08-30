"""Demonstrate how a lock protects shared data from concurrent updates."""

import threading
import time

# Both worker threads read and update this shared balance.
secure_database_balance = 100

# A lock allows only one thread at a time to execute the protected section.
database_lock = threading.Lock()


def process_secure_withdrawal(user_name, extract_amount):
    """Withdraw funds while holding a lock around the shared balance update."""
    global secure_database_balance
    print(f"👤 {user_name} is checking balance...")

    # Acquire the lock before the check-and-update operation. If another thread
    # already holds it, this thread waits here until the lock becomes available.
    database_lock.acquire()
    try:
        # The balance check and deduction form one critical section. Protecting
        # both operations prevents two threads from spending the same balance.
        if secure_database_balance >= extract_amount:
            # Simulate processing latency while access to the balance is exclusive.
            time.sleep(0.5)
            secure_database_balance -= extract_amount
            print(
                f"💰 {user_name} withdrawal success! "
                f"Balance remaining: ${secure_database_balance}"
            )
        else:
            print(
                f"❌ {user_name} transaction denied. "
                f"Insufficient balance: ${secure_database_balance}"
            )
    finally:
        # Always release the lock, even if an exception occurs in the critical
        # section; otherwise, waiting threads could remain blocked indefinitely.
        database_lock.release()


if __name__ == "__main__":
    # Both users concurrently request $60 from the same $100 balance. The lock
    # ensures that one request succeeds and the other sees the updated balance.
    user_a = threading.Thread(
        target=process_secure_withdrawal,
        args=("User-A", 60),
    )
    user_b = threading.Thread(
        target=process_secure_withdrawal,
        args=("User-B", 60),
    )

    # start() runs both withdrawal requests on separate worker threads.
    user_a.start()
    user_b.start()

    # Wait for both transactions before reporting the final shared balance.
    user_a.join()
    user_b.join()

    print(f"\n--- Final Confirmed System Balance: ${secure_database_balance} ---")
