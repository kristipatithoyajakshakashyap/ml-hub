### Multithreading
# When to use the multithreading.
## Multithreading is useful when you have I/O-bound tasks, such as: (network request, file operations).
## Concurrent operations: Improve the throughput of application by performing multiple operations concurrently.

import threading
import time

def print_numbers():
    for i in range(5):
        time.sleep(2)
        print(f"Number: {i}")

def print_letters():
    for letter in "abcde":
        time.sleep(2)
        print(f"Letter: {letter}")

## Create 2 threads
t1 = threading.Thread(target=print_numbers)
t2 = threading.Thread(target=print_letters)

t = time.time()
## Start the threads
t1.start()
t2.start()
## Wait for both threads to complete
t1.join()
t2.join()

finished_time = time.time() - t
print(f"Finished in {finished_time} seconds")