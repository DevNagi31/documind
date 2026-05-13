# Python and the Global Interpreter Lock (GIL)

The Global Interpreter Lock, or GIL, is a mutex that protects access to Python
objects, preventing multiple threads from executing Python bytecode at the
same time. The GIL is necessary in CPython because the language's memory
management is not thread-safe.

## Implications for threading

For **CPU-bound** tasks, threads do not provide a real speedup in CPython:
only one thread executes Python bytecode at a time. The recommended approach
is to use the `multiprocessing` module, which spawns separate processes that
each have their own interpreter and memory space, sidestepping the GIL.

For **I/O-bound** tasks, threads still work well. The GIL is released during
blocking I/O operations (network calls, disk reads), so other Python threads
can run in the meantime. This is why `threading` is still useful for things
like concurrent HTTP requests.

## Alternatives

- `multiprocessing` — true parallelism via separate processes.
- `asyncio` — cooperative concurrency for I/O-bound code without OS threads.
- `concurrent.futures.ProcessPoolExecutor` — a high-level interface over multiprocessing.
