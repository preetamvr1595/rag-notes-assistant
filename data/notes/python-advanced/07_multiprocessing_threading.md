# Python Multiprocessing vs Threading vs GIL

- **Global Interpreter Lock (GIL)**: Mutex preventing multiple native threads from executing Python bytecode simultaneously.
- **Threading**: Best for I/O-bound concurrency (GIL is released during I/O).
- **Multiprocessing**: Spawns separate OS processes with independent GILs, ideal for CPU-bound computation.
